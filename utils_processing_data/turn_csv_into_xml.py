import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

def convert_csv_to_xml(csv_file, output_file='disfluency_dataset.xml'):
    """
    Convert CSV file to XML format for disfluency dataset.
    
    Args:
        csv_file: Path to input CSV file
        output_file: Path to output XML file (default: 'disfluency_dataset.xml')
    """
    # Create the root element
    root = ET.Element('testset', name="Disfluency Pilot", type="output")
    task = ET.SubElement(root, 'task', src_lang="en", trg_lang="en")
    
    # Read and process CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract fields
            file_name = row['file']
            speaker = row['speaker']
            turn = row['turn']
            text_disfluent = row['text_disfluent']
            
            # Create sample id: sw0XXXX_SPEAKER_TURN.wav
            # Extract the number from file (e.g., "sw2005" -> "2005")
            file_number = file_name.replace('sw', '')
            sample_id = f"sw0{file_number}_{speaker}_{turn}.wav"
            
            # Remove underscores from disfluent text
            text_clean = text_disfluent.replace('_', '')
            
            # Remove multiple whitespaces and replace with single space
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()
            
            # Remove whitespace before commas
            text_clean = re.sub(r'\s+,', ',', text_clean)
            
            # Remove whitespace before periods/fullstops
            text_clean = re.sub(r'\s+\.', '.', text_clean)
            
            # Remove whitespace before apostrophes
            text_clean = re.sub(r"\s+'", "'", text_clean)
            
            # Create sample element
            sample = ET.SubElement(task, 'sample', id=sample_id)
            sample.text = text_clean
    
    # Pretty print the XML
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", encoding='utf-8')
    
    # Write to file
    with open(output_file, 'wb') as f:
        f.write(pretty_xml)
    
    print(f"XML file created successfully at {output_file}")
    print(f"Processed {len(task)} samples")
    return output_file


# Example usage:
if __name__ == "__main__":
    # Assuming csv_file variable is provided
    csv_file = 'translation-dataset-with-timestamps.csv'
    convert_csv_to_xml(csv_file)