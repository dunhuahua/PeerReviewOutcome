import os
import csv
import json
import gzip
from collections import Counter


def build_word_frequency_corpus(corpus_dir, output_file):
    """
    Builds a word frequency counter from a directory of text files,
    including gzipped JSONs (like VILA) and plain text.
    Crucially, applies basic ligature fixes to corpus content before counting words
    to ensure full words like "unfortunately" are counted even if they come from
    hyphenated/ligated forms in the raw corpus source.
    """

    assert os.path.exists(corpus_dir)
    
    word_counts = Counter()

    print(f"Building word frequency corpus from '{corpus_dir}'...")

    files = os.listdir(corpus_dir)

    for file_counter, filename in enumerate(files):
        filepath = os.path.join(corpus_dir, filename)

        # Skip if empty: nothing to read
        if os.stat(filepath).st_size == 0:
            continue

        assert filename.endswith("json.gz")

        print('%d: %s' % (file_counter, filename))
        
        # Read in content of file
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            content = data.get("symbols", "")
            
        # Count words in lower-cased content
        word_list = content.lower().split()
        word_counts.update(word_list)

        #if file_counter > 10:
        #    break
        
    # Save to file so it doesn't need re-creating
    with open(output_file, "w") as fp:
        w = csv.writer(fp)
        w.writerows(word_counts.items())
    

OPEN_REVIEW_DIR = os.path.join("/mnt", "xdrive", "preiss_group", "Shared", "openreview")
assert os.path.exists(OPEN_REVIEW_DIR)
    
corpus_dir = os.path.join(OPEN_REVIEW_DIR, "extracted_mmda")
output_file = "simple_freq_file.csv"

build_word_frequency_corpus(corpus_dir, output_file)
