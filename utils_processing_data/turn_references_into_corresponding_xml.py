#!/usr/bin/env python3
"""
Convert disfluency CSV to XML format for multiple languages
"""

import csv
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

def clean_disfluent_text(text):
    """
    Clean disfluent text by removing underscores and fixing spacing
    """
    # Remove underscores from disfluent text
    text_clean = text.replace('_', '')
    
    # Remove multiple whitespaces and replace with single space
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    # Remove whitespace before commas
    text_clean = re.sub(r'\s+,', ',', text_clean)
    
    # Remove whitespace before periods/fullstops
    text_clean = re.sub(r'\s+\.', '.', text_clean)
    
    # Remove whitespace before apostrophes
    text_clean = re.sub(r"\s+'", "'", text_clean)
    
    return text_clean

def create_sample_id(row_id):
    """
    Create sample id in format: sw0XXXX_SPEAKER_TURN.wav
    Example: sw2005_B_2 -> sw02005_B_2.wav
    """
    # Parse the ID (format: swXXXX_SPEAKER_TURN)
    parts = row_id.split('_')
    file_name = parts[0]  # e.g., "sw2005"
    speaker = parts[1]     # e.g., "B"
    turn = parts[2]        # e.g., "2"
    
    # Extract the number from file (e.g., "sw2005" -> "2005")
    file_number = file_name.replace('sw', '')
    sample_id = f"sw0{file_number}_{speaker}_{turn}.wav"
    
    return sample_id

def create_xml_for_language(csv_file, lang_code, lang_column, output_file):
    """
    Create XML file for a specific language
    
    Args:
        csv_file: Input CSV file path
        lang_code: Two-letter language code (e.g., 'en', 'ar', 'de')
        lang_column: Column name for disfluent text in that language
        output_file: Output XML file path
    """
    # Create root element
    root = ET.Element('testset')
    root.set('name', 'Disfluency Pilot')
    root.set('type', 'output')
    
    # Create task element
    task = ET.SubElement(root, 'task')
    task.set('src_lang', 'en')
    task.set('trg_lang', lang_code)
    
    # Read CSV and create samples
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get the disfluent text for this language
            text_disfluent = row[lang_column]
            
            # Skip if text is empty
            if not text_disfluent or text_disfluent.strip() == '':
                continue
            
            # Clean the text
            text_clean = clean_disfluent_text(text_disfluent)
            
            # Create sample element
            sample = ET.SubElement(task, 'sample')
            sample.set('id', create_sample_id(row['ID']))
            sample.text = text_clean
    
    # Convert to pretty XML string
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ', encoding='utf-8')
    
    # Write to file
    with open(output_file, 'wb') as f:
        f.write(pretty_xml)
    
    print(f"Created {output_file}")

def main():
    """
    Main function to process all languages
    """
    csv_file = 'translation_data/references.csv'
    
    # Language mappings: (lang_code, column_name, output_file)
    languages = [
        ('ar', 'AR_disfluent', 'generated_output/transcript/gold/audio_ar_ar_standard_gold_output.xml'),
        ('cs', 'CS_disfluent', 'generated_output/transcript/gold/audio_cs_cs_standard_gold_output.xml'),
        ('de', 'DE_disfluent', 'generated_output/transcript/gold/audio_de_de_standard_gold_output.xml'),
        ('es', 'ES_disfluent', 'generated_output/transcript/gold/audio_es_es_standard_gold_output.xml'),
        ('fr', 'FR_disfluent', 'generated_output/transcript/gold/audio_fr_fr_standard_gold_output.xml'),
        ('hi', 'HI_disfluent', 'generated_output/transcript/gold/audio_hi_hi_standard_gold_output.xml'),
        ('it', 'IT_disfluent', 'generated_output/transcript/gold/audio_it_it_standard_gold_output.xml'),
        ('zh', 'ZH_disfluent', 'generated_output/transcript/gold/audio_zh_zh_standard_gold_output.xml'),
    ]
    
    for lang_code, column_name, output_file in languages:
        create_xml_for_language(csv_file, lang_code, column_name, output_file)
    
    print("\nAll XML files created successfully!")

if __name__ == '__main__':
    main()