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


def tidy_vila_text(json_content):
    vila_text = json_content['symbols']
    vila_span_groups = json_content['vila_span_groups']
    new_text = []
    for group in vila_span_groups:
        if not group['metadata']['type'] in ['Header', 'Equation', 'Figure', 'Caption', 'Table', 'Bibliography']:
            #print('Adding group[metadata][type]: %s' % group['metadata']['type'])
            #print('group[spans][start]: %s' % group['spans'][0]['start'])
            assert len(group['spans']) == 1
            start = int(group['spans'][0]['start'])
            end = int(group['spans'][0]['end'])
            #print('Interval: %d to %d' % (start, end))
            if group['metadata']['type'] == 'Section':
                print('Section: %s' % vila_text[start:end])
                if vila_text[start:end].replace(' ', '').lower() == 'references':
                    print('Finishing text extraction')
                    break
            new_text.append(vila_text[start:end])
        
    return '\n'.join(new_text)
    

def remove_page_numbers(text):

    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        
        if line.strip() in [str(x) for x in list(range(1, 10))]:
            # Do not keep this line
            #print('Potential page number line: %s' % line)
            continue
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

        
def post_process_text(lines):
    """
    Remove clearly incorrect lines:
    # JP: removed 1. as won't match sections!
    1. Lines with a single digit on (probably page number)
    2. Under review as a conference paper at ICLR [0-9]+["]?[,]?
    3. Published as a conference paper at ICLR [0-9]+["]?[,]?
    """

    new_lines = []

    num_under_review = 0
    num_published = 0
    #num_page_numbers = 0
    
    for line in lines:
        
        #if line.strip() in [str(x) for x in list(range(1, 10))]:
        #    # Do not keep this line
        #    #print('Potential page number line: %s' % line)
        #    num_page_numbers += 1
        #    continue
        #el
        if re.match(r'^Under review as a conference paper at ICLR [0-9]+["]?[,]?$', line.strip()):
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
    #print('Number of page number lines removed: %d' % num_page_numbers)
    
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
        #print('current_line: %s' % current_line)
        
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
    print('Num processed lines: %d' % len(processed_lines))
    
    return '\n'.join(processed_lines)


def extract_vila_sections(json_content):

    if not isinstance(json_content, str):
        return []

    sections = []
    
    pattern = re.compile(r'\n([0-9]+)(\.[0-9]+)?\n([^\n]*[^\n0-9][^\n]*)\n',
                         re.MULTILINE | re.DOTALL | re.IGNORECASE)
    
    for result in pattern.finditer(json_content):
        if result.group(2) == None:
            heading = result.group(3).strip()
            level = result.group(1).strip()
            start_offset = result.start()
            end_offset = result.end()
        else:
            heading = result.group(3).strip()
            level = result.group(1).strip() + result.group(2).strip()
            start_offset = result.start()
            end_offset = result.end()
        #print('Potential heading: %s %s %s' % (level, heading, end_offset))
        print('Potential heading: %s %s %s (%s)' % (level, heading, end_offset, json_content[end_offset:(end_offset+20)]))
        sections.append((level, heading, (start_offset, end_offset)))
    
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
            if not head_text:
                continue
            parts = n_value.split('.')
            level = len(parts)
            if level == 1:
                headings.append({
                    'level': level,
                    'n_value': n_value,
                    'text': head_text,
                    'subheadings': []
                })
            else:
                parent_n = '.'.join(parts[:-1])
                found_parent = False
                # Try to find the immediate parent
                for main_heading in headings:
                    if main_heading['n_value'] == parent_n:
                        main_heading['subheadings'].append({
                            'level': level,
                            'n_value': n_value,
                            'text': head_text
                        })
                        found_parent = True
                        break
                if not found_parent:
                    # If parent not found, add it as a new top-level heading.
                    # This might happen if a parent heading was not marked with @n or was malformed.
                    # Or if Grobid output isn't strictly hierarchical.
                    headings.append({
                        'level': level,
                        'n_value': n_value,
                        'text': head_text,
                        'subheadings': []
                    })
        return headings
    except Exception as e:
        print(f"Error processing GROBID file '{file_path}': {e}", file=sys.stderr)
        return []


def read_csv_to_dict(freq_file):
    assert os.path.exists(freq_file)
    with open(freq_file, 'r') as fp:
        reader = csv.reader(fp)
        global_word_frequencies = {row[0]:int(row[1]) for row in reader}
    return global_word_frequencies
        
    
def process_files_to_csv(vila_dir, grobid_dir, output_csv, freq_file, min_score=0.6):

    # Read in global_word_frequencies
    global_word_frequencies = read_csv_to_dict(freq_file)
    print(f"Successfully loaded {len(global_word_frequencies)} unique words for dehyphenation.")

    vila_files = sorted(list(Path(vila_dir).glob("*.json.gz")))
    grobid_files = sorted(list(Path(grobid_dir).glob("*.tei*")))

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

        # Explore problematic papers only
        #if (not 'cO1IH43yUF' in paper_id) and (not '-3Qj7Jl6UP5' in paper_id) and (not '-4hMlsXK4st' in paper_id) and (not 'cO1IH43yUF' in paper_id) and (not '-0LuSWi6j4' in paper_id):
        #if (not '-4hMlsXK4st' in paper_id):
        #if (not '-0LuSWi6j4' in paper_id):
        #    continue

        print(f"Processing paper: {paper_id}")
        single_record = []
        
        if paper_id not in grobid_map:
            #print(f"Warning: No matching GROBID file found for '{paper_id}'. Skipping.", file=sys.stderr)
            print(f"Warning: No matching GROBID file found for '{paper_id}'. Skipping.")
            continue

        g_path = grobid_map[paper_id]

        # Check if file is empty
        if os.path.getsize(v_path) == 0:
            print(f"Warning: empty {v_path}. Skipping.")
            continue
        
        with gzip.open(v_path, 'rt', encoding='utf-8') as f:
            vila_data = json.load(f)
            vila_raw_text = tidy_vila_text(vila_data)

            vila_processed_text = vila_raw_text.replace('ï¬', 'ffi')
            vila_processed_text = vila_processed_text.replace('ï¬‚', 'ffl')
            vila_processed_text = vila_processed_text.replace('â€¢', '•')
                
            if global_word_frequencies:
                print(f"--- Dehyphenating for paper {paper_id} ---")
                vila_processed_text = dehyphenate_text_with_corpus(vila_processed_text, global_word_frequencies)
                print(f"--- Dehyphenation complete for paper {paper_id} ---")
            #print('dehyphenated_text')
            #print(vila_processed_text)

            vila_sections = extract_vila_sections(vila_processed_text)
            if not vila_sections:
                #print(f"Warning: No VILA sections extracted for {paper_id}. Skipping matching.", file=sys.stderr)
                print(f"Warning: No VILA sections extracted for {paper_id}. Skipping matching.")
                continue

        grobid_headings = extract_grobid_headings(g_path)
        if not grobid_headings:
            #print(f"Warning: No GROBID headings extracted for {paper_id}. Skipping matching.", file=sys.stderr)
            print(f"Warning: No GROBID headings extracted for {paper_id}. Skipping matching.")
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

                #print('normalized_vila_heading: %s vs %s (%s vs %s and %s)' % (normalized_grobid_heading, normalized_vila_heading, match_score, highest_match_score, min_score))
                    
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
                    
                single_record.append({
                    "paper_id": paper_id,
                    "section_name_grobid": grobid_heading_text,
                    "section_header_offsets": best_vila_match[1],
                    "subheadings_grobid": subheadings_str
                })

        if len(single_record) > 0:
            # Fill in section_content_vila (based on offsets)
            for i in range(len(single_record) - 1):
                start_offset = single_record[i]['section_header_offsets'][1]
                end_offset = single_record[i+1]['section_header_offsets'][0]
                print('Extracting from %s: %d to %d' % (single_record[i]['section_header_offsets'], start_offset, end_offset))
                single_record[i]['section_content_vila'] = remove_page_numbers(vila_processed_text[start_offset:end_offset].strip())
            # Add the final section
            single_record[-1]['section_content_vila'] = remove_page_numbers(vila_processed_text[single_record[-1]['section_header_offsets'][1]:len(vila_processed_text)])
        # Add to all records
        all_matched_records = all_matched_records + single_record
            
        processed_paper_ids.add(paper_id)

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
    OUTPUT_CSV_FILE = "matched_sections_cleaned_final.csv.gz"
    OUTPUT_FREQ_FILE = "frequency_file.csv"
    OUTPUT_FREQ_FILE = "simple_freq_file.csv"
    MATCHING_MIN_SCORE = 0.6
    # Should be processed used in word_frequencies.py to yield output_freq_file
    #CORPUS_FOR_DEHYPHENATION = VILA_INPUT_DIR
    
    #OUTPUT_CSV = "matched_results.csv.gz" # change to desired csv name
    #OUTPUT_CSV = "test_output.csv.gz"
    
    print("\n--- Starting Text Processing Script ---")
    process_files_to_csv(
        vila_dir=VILA_INPUT_DIR,
        grobid_dir=GROBID_INPUT_DIR,
        output_csv=OUTPUT_CSV_FILE,
        freq_file=OUTPUT_FREQ_FILE,
        min_score=MATCHING_MIN_SCORE
    )

    print("\n--- Processing finished ---")
    
