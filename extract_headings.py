import os
import re
import csv
import sys
import gzip
import json
import random
import pandas as pd
from lxml import etree
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

random.seed(234)


def normalize_for_matching(text):
    """Normalize text for comparison: lowercase, remove non-alphanumeric, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_word_frequency_corpus(corpus_dir: Path, output_file: str) -> Counter:
    """
    Builds a word frequency counter from a directory of text files,
    including gzipped JSONs (like VILA) and plain text.
    Crucially, applies basic ligature fixes to corpus content before counting words
    to ensure full words like "unfortunately" are counted even if they come from
    hyphenated/ligated forms in the raw corpus source.
    """

    if os.path.exists(output_file):
        # Read in the data
        print('Reading word counts from file: %s' % output_file)
        wordDF = pd.read_csv(output_file, header=None)
        word_counts = dict(zip(wordDF[0], wordDF[1]))
        print(f"Loaded {len(word_counts)} unique words.")
        return word_counts
            
    ## Create new word counts
    
    word_counts = Counter()
    if not corpus_dir.exists():
        print(f"Warning: Corpus path '{corpus_dir}' does not exist. Cannot build frequency corpus.", file=sys.stderr)
        return word_counts

    print(f"Building word frequency corpus from '{corpus_dir}'...")
    file_count = 0
    for root, _, files in os.walk(corpus_dir):
        for file_name in files:
            file_path = Path(root) / file_name

            # Skip if empty: nothing to read
            if os.stat(file_path).st_size == 0:
                continue
            
            content = ""
            try:
                if file_path.suffix == '.gz':
                    if 'json' in file_path.name.lower():
                         with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                             data = json.load(f)
                             content = data.get("symbols", "")
                    else:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            content = f.read()
                elif file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = data.get("symbols", "")
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                # --- NEW: Apply basic text cleaning to corpus content BEFORE counting ---
                # This ensures words like "unfortunately" get into the corpus
                # even if they were originally "Unfor-\ntunately" or had ligatures.
                temp_cleaned_content = content.replace('ï¬', 'ffi')
                temp_cleaned_content = temp_cleaned_content.replace('ï¬‚', 'ffl')
                temp_cleaned_content = temp_cleaned_content.replace('â€¢', '•')

                # A more aggressive dehyphenation for corpus building (might not be needed if this fix works)
                # This is a simpler dehyphenation just for populating the corpus, not the main dehyphenator.
                temp_cleaned_content = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', temp_cleaned_content)

                words = re.findall(r'\b[a-z]+\b', temp_cleaned_content.lower()) # Count lowercased words
                word_counts.update(words)
                file_count += 1
            except Exception as e:
                print(f"Error reading corpus file '{file_path}' for frequency building: {e}", file=sys.stderr)
    print(f"Finished building corpus from {file_count} files. Loaded {len(word_counts)} unique words.")

    print(word_counts)
    
    # Save to file so it doesn't need re-creating
    with open(output_file, "w") as fp:
        w = csv.writer(fp)
        w.writerows(word_counts.items())
    
    return word_counts


def post_process_text(lines):
    """
    Remove clearly incorrect lines:
    1. Lines with a single digit on (probably page number)
    2. Under review as a conference paper at ICLR [0-9]+["]?[,]?
    3. Published as a conference paper at ICLR [0-9]+["]?[,]?
    """

    new_lines = []

    num_under_review = 0
    num_published = 0
    num_page_numbers = 0
    
    for line in lines:
        
        if line.strip() in [str(x) for x in list(range(1, 10))]:
            # Do not keep this line
            #print('Potential page number line: %s' % line)
            num_page_numbers += 1
            continue
        elif re.match(r'^Under review as a conference paper at ICLR [0-9]+["]?[,]?$', line.strip()):
            # Do not keep this line
            num_under_review += 1
            continue
        elif re.match(r'Published as a conference paper at ICLR [0-9]+["]?[,]?', line.strip()):
            # Do not keep this line
            num_published += 1
            continue
        else:
            new_lines.append(line)

    print('Number of under review lines removed: %d' % num_under_review)
    print('Number of published lines removed: %d' % num_published)
    print('Number of page number lines removed: %d' % num_page_numbers)
    
    return new_lines


def dehyphenate_text_with_corpus(text: str, word_frequencies: Counter) -> str:
    """
    Dehyphenates text by checking word frequencies for hyphenated words at line breaks.
    Attempts to preserve original casing, especially for sentence-starting words.
    Includes a fallback for zero frequencies if the word looks like a clear dehyphenation candidate.
    """
    if not isinstance(text, str):
        return ""

    if not word_frequencies:
        print("Warning: Word frequency corpus not loaded. Dehyphenation will rely on basic heuristics.", file=sys.stderr)
        # If no corpus, we'll try a simpler dehyphenation
        return re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)


    lines = text.split('\n')
    processed_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]

        if (i + 1) < len(lines):
            next_line = lines[i+1]

            # JP: just sentence final one to worry about
            potential_hyphen = re.match(r'(.*)\s([a-zA-Z]+)-(\s*)$', current_line)
            if potential_hyphen:
                # Check freqs with and without hyphen
                next_word = next_line.split()[0]
                word_without_hyphen = potential_hyphen.group(2) + next_word
                word_with_hyphen = potential_hyphen.group(2) + "-" + next_word

                # Set default to 0
                freq_without = word_frequencies.get(word_without_hyphen.lower(), 0)
                freq_with = word_frequencies.get(word_with_hyphen.lower(), 0)

                # Decide regarding dehyphenation
                if freq_without > freq_with:
                    # Dehyphenate: merge with next line
                    new_line = potential_hyphen.group(1) + " " + potential_hyphen.group(2) + next_word + next_line[len(next_word):]

                    #print('Hyphen remove: %s vs %s (%d vs %d)' % (word_without_hyphen, word_with_hyphen, freq_without, freq_with))
                    #print(new_line)
                else:
                    new_line = current_line + next_line
                    #print('Hyphen leave: %s vs %s (%s vs %s)' % (word_without_hyphen, word_with_hyphen, freq_without, freq_with))
                    #print(new_line)

                # Don't append anything until we are sure this one
                # doesn't have a hyphen on the end either (or we've
                # reached the end of file)
                lines[i + 1] = new_line
            else:
                processed_lines.append(current_line)
        else:
            # Append the last line
            processed_lines.append(current_line)
            
        i += 1

    processed_lines = post_process_text(processed_lines)
        
    return '\n'.join(processed_lines)


def extract_vila_sections(json_content):
    if not isinstance(json_content, str):
        return []
    sections = []
    pattern = re.compile(
        r'^\s*(?P<number>\d+(?:\.\d+)*\s*)?'
        r'(?P<heading_text>[A-Z][A-Z\s]+(?:[A-Z]|\d)*)'
        r'\s*\n+'
        r'(?P<content>.*?)'
        r'(?=\n^\s*(?:\d+(?:\.\d+)*\s*)?[A-Z][A-Z\s]+(?:[A-Z]|\d)*\s*\n+|\Z)',
        re.MULTILINE | re.DOTALL
    )
    for match in pattern.finditer(json_content):
        heading = match.group('heading_text').strip()
        text = match.group('content').strip()
        num_part = match.group('number')
        level = num_part.count('.') + 1 if num_part else 0
        sections.append((level, heading, text))
    return sections


def extract_grobid_headings(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    try:
        try:
            with open(file_path, 'rb') as f:
                tree = etree.parse(f)
        except etree.XMLSyntaxError:
            with gzip.open(file_path, 'rb') as f:
                tree = etree.parse(f)
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        headings = []
        for head_element in tree.xpath('//tei:body//tei:head[@n]', namespaces=ns):
            n_value = head_element.get('n')
            head_text_elements = head_element.xpath('./text()', namespaces=ns)
            head_text = head_text_elements[0].strip() if head_text_elements else ''

            #print('n_value: %s' % n_value)
            #print('head_text_elements: %s' % head_text_elements)
            #print('head_text: %s' % head_text)
            
            if not head_text:
                continue
            parts = n_value.split('.')
            level = len(parts)

            #print('level: %d' % level)
            
            if level == 1:
                headings.append({
                    "level": level,
                    "n_value": n_value,
                    "text": head_text,
                    "subheadings": []
                })
            else:

                #print('Current headings: %s' % headings)   
                parent_n = '.'.join(parts[:-1])
                #print('parent_n: %s' % parent_n)
                found_parent = False

                if level == 2:
                    for main_heading in headings:
                        #print('Checking main_heading: %s' % main_heading)
                        if main_heading['n_value'] == parent_n:
                            main_heading['subheadings'].append({
                                "level": level,
                                "n_value": n_value,
                                "text": head_text,
                                "subheadings": []
                            })
                            found_parent = True
                            break
                elif level == 3:
                    for main_heading in headings:
                        for subheading in main_heading['subheadings']:
                            if subheading['n_value'] == parent_n:
                                subheading['subheadings'].append({
                                    "level": level,
                                    "n_value": n_value,
                                    "text": head_text,
                                    "subheadings": []
                                    })
                                found_parent = True
                                break
                else:
                    assert False, "Assuming max depth is 3"
                    
                if not found_parent:
                    print('WARNING: Parent of %s NOT found' % n_value)
                    headings.append({
                        "level": level,
                        "n_value": n_value,
                        "text": head_text,
                        "subheadings": []
                    })
        return headings
    except Exception as e:
        print(f"Error processing GROBID file '{file_path}': {e}", file=sys.stderr)
        return []


def process_files_to_csv(vila_dir, grobid_dir, output_csv, corpus_for_dehyphenation, freq_file, min_score=0.6):
    vila_dir = Path(vila_dir)
    grobid_dir = Path(grobid_dir)
    global_word_frequencies = Counter()
    if corpus_for_dehyphenation:
        global_word_frequencies = build_word_frequency_corpus(Path(corpus_for_dehyphenation), freq_file)
        print(f"Successfully loaded {len(global_word_frequencies)} unique words for dehyphenation.")
    else:
        print("No corpus path provided for dehyphenation. Dehyphenation will be skipped.", file=sys.stderr)

    vila_files = sorted(list(vila_dir.glob("*.json.gz")))
    grobid_files = sorted(list(grobid_dir.glob("*.tei*")))

    grobid_map = {}
    for g_path in grobid_files:
        base_name = g_path.name.split('.grobid')[0]
        grobid_map[base_name] = g_path

    all_matched_records = []
    processed_paper_ids = set()

    print(f"\nStarting to process {len(vila_files)} VILA files...")
    for v_path in vila_files:
        paper_id = v_path.stem.split('.')[0]

        if paper_id in processed_paper_ids:
            print(f"Skipping already processed paper: {paper_id}", file=sys.stderr)
            continue

        if (not 'cO1IH43yUF' in paper_id) and (not '-3Qj7Jl6UP5' in paper_id) and (not '-4hMlsXK4st' in paper_id) and (not 'cO1IH43yUF' in paper_id) and (not '-0LuSWi6j4' in paper_id):
            continue

        print(f"Processing paper: {paper_id}")
        
        if paper_id not in grobid_map:
            #print(f"Warning: No matching GROBID file found for '{paper_id}'. Skipping.", file=sys.stderr)
            print(f"Warning: No matching GROBID file found for '{paper_id}'. Skipping.")
            continue

        g_path = grobid_map[paper_id]

        try:
            with gzip.open(v_path, 'rt', encoding='utf-8') as f:
                vila_data = json.load(f)
                vila_raw_text = vila_data.get("symbols", "")

                vila_processed_text = vila_raw_text.replace('ï¬', 'ffi')
                vila_processed_text = vila_processed_text.replace('ï¬‚', 'ffl')
                vila_processed_text = vila_processed_text.replace('â€¢', '•')

                if global_word_frequencies:
                    print(f"--- Dehyphenating for paper {paper_id} ---")
                    vila_processed_text = dehyphenate_text_with_corpus(vila_processed_text, global_word_frequencies)
                    print(f"--- Dehyphenation complete for paper {paper_id} ---")

                vila_sections = extract_vila_sections(vila_processed_text)
                if not vila_sections:
                    #print(f"Warning: No VILA sections extracted for {paper_id}. Skipping matching.", file=sys.stderr)
                    print(f"Warning: No VILA sections extracted for {paper_id}. Skipping matching.")
                    continue

                for (v1, v2, v3) in vila_sections:                
                    print('vila_sections: %s %s' % (v1, v2))
                
            grobid_headings = extract_grobid_headings(g_path)
            if not grobid_headings:
                print(f"Warning: No GROBID headings extracted for {paper_id}. Skipping matching.", file=sys.stderr)
                continue
            print('Grobid headings: %s' % grobid_headings)

            for grobid_main_heading in grobid_headings:
                grobid_heading_text = grobid_main_heading['text']
                normalized_grobid_heading = normalize_for_matching(grobid_heading_text)

                print('Searching for: %s (%s)' % (grobid_heading_text, normalized_grobid_heading))

                best_vila_match = None
                highest_match_score = 0

                for vila_level, vila_heading_text, vila_section_content in vila_sections:
                    normalized_vila_heading = normalize_for_matching(vila_heading_text)

                    match_score = SequenceMatcher(
                        None,
                        normalized_vila_heading,
                        normalized_grobid_heading
                    ).ratio()

                    if match_score > highest_match_score and match_score >= min_score:
                        highest_match_score = match_score
                        best_vila_match = (vila_heading_text, vila_section_content)

                subheadings_str = "; ".join(
                    f"{sub['n_value']}: {sub['text']}"
                    for sub in grobid_main_heading['subheadings']
                )

                if best_vila_match:

                    #print('Best VILA match: %s' % best_vila_match[1])
                    print('Best VILA match FOUND')
                    
                    all_matched_records.append({
                        "paper_id": paper_id,
                        "section_name_grobid": grobid_heading_text,
                        "section_content_vila": best_vila_match[1],
                        "subheadings_grobid": subheadings_str
                    })

            processed_paper_ids.add(paper_id)

        except Exception as e:
            print(f"Critical error processing paper '{paper_id}': {e}", file=sys.stderr)

    if all_matched_records:
        df = pd.DataFrame(all_matched_records)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nProcessing complete! Results saved to '{output_csv}'")
        print(f"Total unique papers with matched sections: {len(processed_paper_ids)}")
        print(f"Total sections matched and exported: {len(df)}")
    else:
        print("\nNo matching data found to export after processing all files.", file=sys.stderr)

        
if __name__ == "__main__":

    OPEN_REVIEW_DIR = os.path.join("/mnt", "xdrive", "preiss_group", "Shared", "openreview")
    assert os.path.exists(OPEN_REVIEW_DIR)
    
    VILA_INPUT_DIR = os.path.join(OPEN_REVIEW_DIR, "extracted_mmda")
    GROBID_INPUT_DIR = os.path.join(OPEN_REVIEW_DIR, "extracted_grobid")
    OUTPUT_CSV_FILE = "matched_sections_cleaned_final.csv"
    OUTPUT_FREQ_FILE = "frequency_file.csv"
    OUTPUT_FREQ_FILE = "simple_freq_file.csv"
    MATCHING_MIN_SCORE = 0.6
    CORPUS_FOR_DEHYPHENATION = VILA_INPUT_DIR
    
    #OUTPUT_CSV = "matched_results.csv.gz" # change to desired csv name
    #OUTPUT_CSV = "test_output.csv.gz"
    
    print("\n--- Starting Text Processing Script ---")
    process_files_to_csv(
        vila_dir=VILA_INPUT_DIR,
        grobid_dir=GROBID_INPUT_DIR,
        output_csv=OUTPUT_CSV_FILE,
        corpus_for_dehyphenation=CORPUS_FOR_DEHYPHENATION,
        freq_file=OUTPUT_FREQ_FILE,
        min_score=MATCHING_MIN_SCORE
    )

