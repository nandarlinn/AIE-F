python preprocessing.py --input_dir corpus/
python filter_garbled_corpus.py --input_dir segmented_corpus/ --output_dir filtered_corpus/
python mmsyldecomposer.py --input_dir filtered_corpus/ --output rdr_level2_rules.json
python build_ngram_model.py --input_dir filtered_corpus/ --output_file ngram_model.json
python grammer_spelling_checker.py
