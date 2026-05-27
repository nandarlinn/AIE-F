import json
import re
from rabbit import Rabbit as rabbit
from myword import WordTokenizer
from myanmartools import ZawgyiDetector

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
                "a": "်" if "်" in syl else None,
                "t": re.findall(f"[{self.tones}]", syl)
            })
        return decomposed_data


class MyanmarSpellingChecker:
    def __init__(self, rule_json_path, ngram_model_path=None):
        self.rule_json_path = rule_json_path
        # Load the saved Knowledge Base
        try:
            with open(rule_json_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
        except FileNotFoundError:
            print(f"Error: {rule_json_path} ကို မတွေ့ပါ။ အရင်ဆုံး Knowledge Base ကို generate လုပ်ပါ။")
            self.rules = {}
            
        self.word_rules = self.rules.get("word_level_rules", {})
        self.syllable_rules = self.rules.get("syllable_level_rules", {})
        self.decomposer = MyanmarSyllableDecomposer()
        
        # Load N-gram Model
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        if ngram_model_path:
            try:
                with open(ngram_model_path, 'r', encoding='utf-8') as f:
                    ngram_data = json.load(f)
                    self.unigrams = ngram_data.get("unigrams", {})
                    self.bigrams = ngram_data.get("bigrams", {})
                    self.trigrams = ngram_data.get("trigrams", {})
                print(f"Loaded N-gram model: {len(self.unigrams)} unigrams, {len(self.bigrams)} bigrams, {len(self.trigrams)} trigrams.")
            except FileNotFoundError:
                print(f"Warning: N-gram model '{ngram_model_path}' မတွေ့ပါ။")

        self.blacklist_words = {
            "ယဥ်": "ယဉ်",
            "ယာဥ်": "ယာဉ်",
            "စဥ်": "စဉ်",
            "ဥ်": "ဉ်"
        }

        # Confusable words group for N-gram Context Suggestion
        self.confusables = {
            "ယဉ်": ["ယာဉ်", "ယဉ်"],
            "ယာဉ်": ["ယဉ်", "ယာဉ်"],
            "စဉ်": ["စဉ့်", "စဉ်", "စင်"],
            "စင်": ["စဉ်", "စင်", "စဉ့်"],
            "သစ်": ["သစ်", "သစ်ပင်", "စစ်"],
            "စစ်": ["သစ်", "စစ်", "စစ်တပ်"]
        }

        self.bigram_corrections = {
            "လေ ယဥ်": "လေယာဉ်",
            "လေ ယာဉ်": "လေယာဉ်",
            "လေ ယာဥ်": "လေယာဉ်"
        }

    def save_rules(self):
        """Save updated rules back to JSON"""
        self.rules["word_level_rules"] = self.word_rules
        self.rules["syllable_level_rules"] = self.syllable_rules
        with open(self.rule_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, indent=4, ensure_ascii=False)

    def add_word_to_kb(self, new_word):
        """Add a new valid word to the knowledge base"""
        syllables = self.decomposer.decompose(new_word)
        word_components = []

        for syl in syllables:
            c = "".join(syl['c'])
            m = "".join(syl['m'])
            a = syl['a'] if syl['a'] else ""

            if not c: continue

            word_components.append(f"{c}+{m}+{a}")

            if c not in self.syllable_rules:
                self.syllable_rules[c] = {}
            if m not in self.syllable_rules[c]:
                self.syllable_rules[c][m] = []
            if a not in self.syllable_rules[c][m]:
                self.syllable_rules[c][m].append(a)

        self.word_rules[new_word] = word_components
        self.save_rules()

        self.blacklist = {
            "ယဥ်": "ယဉ်",
            "ယာဥ်": "ယာဉ်",
            "စဥ်": "စဉ်",
            "ဥ်": "ဉ်"
        }

        self.bigram_corrections = {
            "လေ ယဥ်": "လေယာဉ်",
            "လေ ယာဉ်": "လေယာဉ်",
            "လေ ယာဥ်": "လေယာဉ်"
        }
        

    def check_syllable_structure(self, c, m, a):
        """Check if C+M+A combination exists in Syllable Rules"""
        if c in self.syllable_rules:
            if m in self.syllable_rules[c]:
                if a in self.syllable_rules[c][m]:
                    return True
        return False

    def check_word(self, word):
        """Level 2 Check: Verify the whole word and its syllables"""
        # Step 1: Check Exact Match in Word Dictionary
        if word in self.word_rules:
            return {"word": word, "status": "VALID", "message": "မှန်ကန်ပါသည်။"}

        # Step 2: Check individual syllables if word is not found
        syllables = self.decomposer.decompose(word)
        invalid_syllables = []
        
        for syl in syllables:
            c = "".join(syl['c'])
            m = "".join(syl['m'])
            a = syl['a'] if syl['a'] else ""
            
            if not c: continue 
            
            is_valid_syl = self.check_syllable_structure(c, m, a)
            if not is_valid_syl:
                invalid_syllables.append(syl['syllable'])

        # Step 3: Return analysis
        if invalid_syllables:
            return {
                "word": word, 
                "status": "INVALID_SYLLABLE", 
                "message": f"သတ်ပုံဖွဲ့စည်းပုံ မှားယွင်းနေပါသည်: {', '.join(invalid_syllables)}"
            }
        else:
            return {
                "word": word, 
                "status": "CONTEXT_ERROR", 
                "message": "Syllable မှန်သော်လည်း စကားလုံးတွဲဖက်မှု အဘိဓာန်တွင် မရှိပါ။"
            }

    def check_sentence(self, segmented_sentence, interactive=False):
        """Check a space-separated segmented sentence"""
        words = segmented_sentence.strip().split()
        results = []
        for i, word in enumerate(words):
            # Skip pure punctuation
            if re.match(r'^[။၊\s\.]+$', word):
                continue

            # Step 1: Bigram Check (ရှေ့စကားလုံးနှင့် တွဲစစ်ခြင်း)
            if i > 0:
                prev_word = words[i-1]
                bigram = f"{prev_word} {word}"
                if bigram in self.bigram_corrections:
                    # အရှေ့က "လေ" ရဲ့ result ကိုပါ ပြင်ဆင်ခြင်း
                    if results:
                        results[-1]['status'] = "BIGRAM_ERROR"
                        results[-1]['message'] = f"နောက်စကားလုံးနှင့်တွဲ၍ '{self.bigram_corrections[bigram]}' ဟု ပြင်ပါ။"
                    
                    results.append({
                        "word": word,
                        "status": "BIGRAM_ERROR",
                        "message": f"'{bigram}' အစား '{self.bigram_corrections[bigram]}' ဟု ပြင်ပါ။"
                    })
                    continue

            # Step 2: Blacklist Check (တားမြစ်စကားလုံး စစ်ခြင်း)
            if word in self.blacklist_words:
                results.append({
                    "word": word,
                    "status": "BLACKLISTED",
                    "message": f"'{self.blacklist_words[word]}' ဟု ရေးရပါမည်။"
                })
                continue
            
            res = self.check_word(word)
            
            # Step 3: Statistical N-gram Context Check
            # If the word is grammatically valid but forms an unusual bigram, suggest corrections
            if res["status"] == "VALID" and i > 0 and self.bigrams:
                prev_word = words[i-1]
                current_bigram = f"{prev_word} {word}"
                
                # Check if this bigram is unknown to the model
                if current_bigram not in self.bigrams:
                    # Let's check if there is a more probable confusable word
                    best_candidate = None
                    highest_prob = 0
                    
                    candidates = self.confusables.get(word, [])
                    for candidate in candidates:
                        cand_bigram = f"{prev_word} {candidate}"
                        freq = self.bigrams.get(cand_bigram, 0)
                        if freq > highest_prob:
                            highest_prob = freq
                            best_candidate = candidate
                    
                    if best_candidate and best_candidate != word:
                        res["status"] = "CONTEXT_WARNING"
                        res["message"] = f"ရှေ့နောက်စကားလုံးတွဲဖက်မှု အားနည်းပါသည်။ '{best_candidate}' ဖြစ်နိုင်ပါသလား? (အကြိမ်အရေအတွက်: {highest_prob})"
                    elif not best_candidate:
                        res["status"] = "CONTEXT_UNKNOWN"
                        res["message"] = f"'{current_bigram}' တွဲဖက်မှုကို Corpus တွင် မတွေ့ရပါ။"
            
            # Interactive flow for OOV words
            if interactive and res["status"] == "CONTEXT_ERROR":
                print(f"\n[?] '{word}' သည် အဘိဓာန်တွင် မရှိပါ။ (Syllable မှန်ကန်ပါသည်)")
                choice = input(f"'{word}' ကို စကားလုံးအသစ်အနေဖြင့် Knowledge Base သို့ ထည့်မည်လား? (y/n): ").strip().lower()
                if choice == 'y':
                    self.add_word_to_kb(word)
                    res["status"] = "VALID_NEW_WORD"
                    res["message"] = "Knowledge Base သို့ အသစ်ထည့်သွင်းလိုက်ပါသည်။"
                    print(f"'{word}' ကို Knowledge Base ထဲသို့ အောင်မြင်စွာ ထည့်သွင်းပြီးပါပြီ။\n")
                    
            results.append(res)
        return results

# ==========================================
# TEST THE CHECKER
# ==========================================
if __name__ == "__main__":
    # If you have built ngram_model.json, pass it here. Otherwise, pass None or omit.
    checker = MyanmarSpellingChecker('rdr_level2_rules.json', 'ngram_model.json')
    detector = ZawgyiDetector()
    words = WordTokenizer()
    
    raw_test_sentence = "စိတ်ကူးယဥ်အိပ်မက်တွေ"
    # raw_test_sentence = "မမရေ မမရေ မမရေ... ကျတော့်ကို တံခါးဖွင့်ပေးပါ... သွေးရူးသွေးတန်း ကိန်းဂဏန်းတွေ ကိုင်မြှောက်ပြ အကြိမ်တစ်ရာရှိပါပြီ"
    
    # 1. Zawgyi to Unicode
    if detector.get_zawgyi_probability(raw_test_sentence) > 0.95:
        unicode_sentence = rabbit.zg2uni(raw_test_sentence)
    else:
        unicode_sentence = raw_test_sentence
        
    # 2. Segment using myword
    tokens = words.tokenize(unicode_sentence)
    test_sentence = " ".join([t.strip() for t in tokens])
    
    print(f"Checking Sentence: {test_sentence}\n")
    print("-" * 60)
    
    # Change interactive=True to enable the interactive prompt
    output = checker.check_sentence(test_sentence, interactive=True)
    
    for res in output:
        print(f"Word: {res['word']:<10} | Status: {res['status']:<18} | Info: {res['message']}")