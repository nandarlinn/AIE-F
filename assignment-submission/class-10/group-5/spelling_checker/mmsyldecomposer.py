import json
import re
import argparse
from collections import defaultdict

class MyanmarSyllableDecomposer:
    def __init__(self):
        self.consonants = r"က-အ"
        self.medials = r"ျြွှ"
        self.vowels = r"ေ-ဲာ-ှု"
        self.asats = r"်"
        self.tones = r"း့"
        self.syllable_pattern = re.compile(
            r"([%s][%s]*[%s]*[%s]?[%s]?)" % (
                self.consonants, self.medials, self.vowels, self.asats, self.tones
            )
        )

    def decompose(self, text):
        syllables = self.syllable_pattern.findall(text)
        decomposed_data = []
        for syl in syllables:
            decomposed_data.append({
                "syllable": syl,
                "c": re.findall(f"[{self.consonants}]", syl),
                "m": re.findall(f"[{self.medials}]", syl),
                "v": re.findall(f"[{self.vowels}]", syl),
                "a": "်" if "်" in syl else None
            })
        return decomposed_data

def generate_rules_from_file(input_filepath, output_filepath):
    decomposer = MyanmarSyllableDecomposer()
    syllable_rules = defaultdict(lambda: defaultdict(set))
    word_rules = {}

    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            words = line.strip().split()
            for word in words:
                if not word: continue
                
                syllables = decomposer.decompose(word)
                word_components = []

                for syl in syllables:
                    c = "".join(syl['c'])
                    m = "".join(syl['m'])
                    a = syl['a'] if syl['a'] else ""
                    
                    if not c: continue
                    syllable_rules[c][m].add(a)
                    word_components.append(f"{c}+{m}+{a}")

                if word not in word_rules:
                    word_rules[word] = word_components

    formatted_syllable_kb = {}
    for c, medials in syllable_rules.items():
        formatted_syllable_kb[c] = {m: list(asats) for m, asats in medials.items()}

    final_kb = {
        "word_level_rules": word_rules,
        "syllable_level_rules": formatted_syllable_kb
    }

    # Save to file
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(final_kb, outfile, indent=4, ensure_ascii=False)

    print(f"{output_filepath} file has been successfully created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Syllable Decomposer Rules from Segmented Corpus")
    parser.add_argument("--input", default="segmented_corpus.txt", help="Path to the segmented corpus file")
    parser.add_argument("--output", default="rdr_level2_rules.json", help="Path to output the generated JSON rules")
    
    args = parser.parse_args()
    generate_rules_from_file(args.input, args.output)