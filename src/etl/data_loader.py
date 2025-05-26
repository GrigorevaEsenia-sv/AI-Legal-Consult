import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from datasets import load_dataset
import os
import glob

def xml_to_json(xml_str):
    """Convert XML string to JSON format preserving all fields"""
    root = ET.fromstring(xml_str)
    
    result = {
        "meta": {},
        "body": {}
    }
    
    # Process meta section
    meta = root.find('meta')
    if meta is not None:
        # Identification
        identification = meta.find('identification')
        if identification is not None:
            result['meta']['identification'] = {
                'pravogovruNd': identification.find('pravogovruNd').attrib.get('val'),
                'issuedByIPS': identification.find('issuedByIPS').attrib.get('val'),
                'docdateIPS': identification.find('docdateIPS').attrib.get('val'),
                'docNumberIPS': identification.find('docNumberIPS').attrib.get('val'),
                'headingIPS': identification.find('headingIPS').text,
                'doc_typeIPS': identification.find('doc_typeIPS').attrib.get('val'),
                'doc_author_normal_formIPS': identification.find('doc_author_normal_formIPS').attrib.get('val'),
                'signedIPS': identification.find('signedIPS').attrib.get('val'),
                'statusIPS': identification.find('statusIPS').attrib.get('val'),
                'actual_datetimeIPS': identification.find('actual_datetimeIPS').attrib.get('val'),
                'actual_datetime_humanIPS': identification.find('actual_datetime_humanIPS').attrib.get('val'),
                'is_widely_used': identification.find('is_widely_used').attrib.get('val')
            }
        
        # References
        references = meta.find('references')
        if references is not None:
            result['meta']['references'] = {
                'classifierByIPS': references.find('classifierByIPS').attrib.get('val')
            }
        
        # Keywords
        keywords = meta.find('keywords')
        if keywords is not None:
            result['meta']['keywords'] = [k.attrib.get('val') for k in keywords.findall('keywordByIPS')]
    
    # Process body section
    body = root.find('body')
    if body is not None:
        result['body'] = {
            'textIPS': body.find('textIPS').text if body.find('textIPS') is not None else None,
            'taggedTextIPS': body.find('taggedTextIPS').text if body.find('taggedTextIPS') is not None else None
        }
    
    return result

def process_parquet_to_json(parquet_path, output_dir):
    """Convert parquet file to JSON files"""
    df = pd.read_parquet(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for idx, row in df.iterrows():
        xml_content = row['content']  # assuming the XML is in 'content' column
        json_data = xml_to_json(xml_content)
        
        # Save each document as separate JSON file
        output_path = output_dir / f"document_{idx}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print(glob.glob("data/raw/RusLawOD/*.parquet"))  # Check parquet files
    print(os.path.exists("data/raw/RusLawOD"))  # Check directory exists

    for parquet_path in glob.glob("data/raw/RusLawOD/*.parquet"):
        output_dir = f"data/raw/RusLawOD/law_corpus_{os.path.splitext(os.path.basename(parquet_path))[0]}.jsonl"
        print(output_dir)
        process_parquet_to_json(parquet_path, output_dir)
