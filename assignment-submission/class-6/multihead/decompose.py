"""
Myanmar Syllable Component Decomposer
=====================================
Decomposes each Myanmar syllable in syl.txt into structural components:

  base    : the leading consonant or independent vowel (1 codepoint)
  medials : a tuple of medial codepoints (U+103B, U+103C, U+103D, U+103E)
  vowels  : a tuple of dependent vowel sign codepoints (U+102B..U+1032, U+1036)
  asat    : whether U+103A is present
  stack   : the stacked consonant after U+1039 virama, or None

The decomposition is based on Unicode position rules from Hosken's
"Representing Myanmar in Unicode" (UTN#11) and Unicode chapter 16.

Each component head gets a small label vocabulary, so the model has hundreds
of training examples per component-class even when individual syllables have
only 2 samples.

Why this works:
  4413 full-syllable labels   -> 2 training samples per label   (too few)
  ~35 base labels             -> ~250 training samples per label (plenty)
  ~16 medial-combo labels     -> ~550 training samples per label
  ~30 vowel-combo labels      -> ~290 training samples per label

The model predicts all heads jointly and we recompose the syllable at
inference time.
"""

import json
from collections import Counter
from typing import List, Tuple, Dict


# -------------------------------------------------------------------
# Unicode codepoint categories for Myanmar script (U+1000..U+109F)
# -------------------------------------------------------------------

# Independent letters that can be the "base" of a syllable.
# Includes consonants U+1000..U+1021 and the special bases U+1023, U+1025,
# U+1027, U+1029, U+103F, U+104E that appear in syl.txt.
BASE_CHARS = set(range(0x1000, 0x1022)) | {
    0x1023, 0x1025, 0x1027, 0x1029, 0x103F, 0x104E
}

# Medial consonants (modifiers attached after the base consonant)
MEDIAL_CHARS = {0x103B, 0x103C, 0x103D, 0x103E}

# Dependent vowel signs (and anusvara, which functions as a final nasal)
VOWEL_CHARS = {
    0x102B, 0x102C, 0x102D, 0x102E, 0x102F, 0x1030,
    0x1031, 0x1032, 0x1036,
}

# Tone / final marks
TONE_CHARS = {0x1037, 0x1038}     # dot below, visarga
ASAT       = 0x103A               # vowel killer
VIRAMA     = 0x1039               # stacking marker
KINZI      = 0x1004               # "Nga", forms kinzi with ASAT+VIRAMA

# -------------------------------------------------------------------
# Decomposer
# -------------------------------------------------------------------

def decompose(syl: str) -> Dict:
    """
    Walk the codepoints of a Myanmar syllable and bucket them into structural
    components. We are lenient: if we cannot identify something, we put it in
    'extra' and the syllable still gets a usable decomposition.

    Returns a dict with keys: base, medials, vowels, final, asat, stack, tones, extra.
    'final' is the closing consonant in a closed syllable like နိုင် (= နို + င + ်):
    a consonant that appears AFTER the vowels but BEFORE the asat.
    """
    cps = [ord(c) for c in syl]
    base    = None
    medials = []
    vowels  = []
    tones   = []
    asat    = False
    final   = None      # closing consonant before ASAT
    stack   = None      # the consonant after virama, if any
    extra   = []

    i = 0
    # First codepoint should be a base
    if cps and cps[0] in BASE_CHARS:
        base = cps[0]
        i = 1
    elif cps:
        base = cps[0]
        i = 1

    while i < len(cps):
        c = cps[i]

        # Virama starts a stacked consonant: skip the virama, capture the next.
        if c == VIRAMA:
            if i + 1 < len(cps) and cps[i + 1] in BASE_CHARS:
                stack = cps[i + 1]
                i += 2
                continue
            i += 1
            continue

        if c == ASAT:
            asat = True
            i += 1
            continue

        if c in MEDIAL_CHARS:
            medials.append(c)
            i += 1
            continue

        if c in VOWEL_CHARS:
            vowels.append(c)
            i += 1
            continue

        if c in TONE_CHARS:
            tones.append(c)
            i += 1
            continue

        # An unhandled BASE_CHAR appearing AFTER vowels is a closed-syllable
        # final consonant (e.g. ng in မင်း, ait in အိတ်). Record it.
        if c in BASE_CHARS and final is None:
            final = c
            i += 1
            continue

        # Anything else: treat as extra.
        extra.append(c)
        i += 1

    medials = tuple(sorted(set(medials)))
    vowels  = tuple(sorted(set(vowels)))
    tones   = tuple(sorted(set(tones)))

    return {
        "base":    base,
        "medials": medials,
        "vowels":  vowels,
        "final":   final,
        "asat":    asat,
        "stack":   stack,
        "tones":   tones,
        "extra":   tuple(extra),
    }


def recompose(d: Dict) -> str:
    """
    Reconstruct a Myanmar syllable string from a decomposition dict.

    Canonical Unicode storage order for vowels: U+1031 (ေ) comes FIRST,
    then other vowel signs in numeric order. We must un-sort the tuple
    we stored to put U+1031 ahead of any other vowel codepoint that
    accompanies it.
    """
    out = []
    if d["base"] is not None:
        out.append(chr(d["base"]))
    for m in d["medials"]:
        out.append(chr(m))

    # Canonical vowel order: U+1031 first, then the rest in code order
    vowels = list(d["vowels"])
    if 0x1031 in vowels:
        vowels.remove(0x1031)
        vowels = [0x1031] + vowels
    for v in vowels:
        out.append(chr(v))

    if d.get("final") is not None:
        out.append(chr(d["final"]))
    if d["asat"]:
        out.append(chr(ASAT))
    if d["stack"] is not None:
        out.append(chr(VIRAMA))
        out.append(chr(d["stack"]))
    for t in d["tones"]:
        out.append(chr(t))
    for c in d.get("extra", ()):
        out.append(chr(c))
    return "".join(out)


# -------------------------------------------------------------------
# Vocabulary builder
# -------------------------------------------------------------------

def build_vocabularies(syls: List[str]):
    """
    Walk all syllables and build label vocabularies for each head.
    Each head's vocabulary contains a special "<none>" / "<unk>" label at 0.
    Returns a dict of {head_name: list_of_labels}.
    """
    base_counts    = Counter()
    medials_counts = Counter()
    vowels_counts  = Counter()
    final_counts   = Counter()
    stack_counts   = Counter()
    tones_counts   = Counter()

    for s in syls:
        d = decompose(s)
        base_counts[d["base"]]      += 1
        medials_counts[d["medials"]] += 1
        vowels_counts[d["vowels"]]   += 1
        final_counts[d["final"]]     += 1
        stack_counts[d["stack"]]     += 1
        tones_counts[d["tones"]]     += 1

    # Order vocabularies deterministically: by frequency, then by key.
    def order(counter, none_value):
        # none_value gets index 0
        items = [k for k in counter if k != none_value]
        items.sort(key=lambda k: (-counter[k], str(k)))
        return [none_value] + items

    vocabs = {
        "base":    order(base_counts,    None),
        "medials": order(medials_counts, ()),
        "vowels":  order(vowels_counts,  ()),
        "final":   order(final_counts,   None),
        "stack":   order(stack_counts,   None),
        "tones":   order(tones_counts,   ()),
        # asat is a 2-class binary head
        "asat":    [False, True],
    }
    return vocabs


# -------------------------------------------------------------------
# Syllable -> per-head label indices
# -------------------------------------------------------------------

def syllable_to_label_indices(syl: str, vocabs) -> Dict[str, int]:
    """Convert a syllable to a dict of {head: label_index_into_vocab}."""
    d = decompose(syl)
    out = {}
    for head in ("base", "medials", "vowels", "final", "stack", "tones"):
        key = d[head]
        vocab = vocabs[head]
        try:
            out[head] = vocab.index(key)
        except ValueError:
            out[head] = 0   # fall back to "<none>" if unseen
    out["asat"] = 1 if d["asat"] else 0
    return out


# -------------------------------------------------------------------
# CLI: stats for our actual syl.txt
# -------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Decompose syl.txt and print stats")
    ap.add_argument("--syl_file", default="syl.txt")
    ap.add_argument("--out_json", default=None,
                    help="If set, save vocabs + decompositions to this file")
    args = ap.parse_args()

    with open(args.syl_file, "r", encoding="utf-8") as f:
        syls = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(syls)} syllables from {args.syl_file}")

    vocabs = build_vocabularies(syls)
    print()
    for head, vocab in vocabs.items():
        print(f"{head:>8}: {len(vocab):>4} labels")

    # Show top examples
    print()
    print("First 10 syllables decomposed:")
    for s in syls[:10]:
        d = decompose(s)
        rebuilt = recompose(d)
        ok = "OK" if rebuilt == s else f"DIFF (got {rebuilt!r})"
        print(f"  {s!r:>10}  base={d['base']}  med={d['medials']}  "
              f"vow={d['vowels']}  asat={d['asat']}  stack={d['stack']}  "
              f"tones={d['tones']}  [{ok}]")

    # Check round-trip on the whole set
    mismatches = 0
    for s in syls:
        if recompose(decompose(s)) != s:
            mismatches += 1
    print(f"\nRound-trip mismatches: {mismatches} / {len(syls)} "
          f"({mismatches/len(syls)*100:.2f}%)")

    if args.out_json:
        # vocabs contain ints/None/tuples; convert tuples to lists for JSON
        json_safe = {}
        for k, v in vocabs.items():
            json_safe[k] = [
                list(x) if isinstance(x, tuple) else x for x in v
            ]
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"vocabs": json_safe, "n_syllables": len(syls)},
                      f, ensure_ascii=False, indent=2)
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
