import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm  # for progress tracking

def clean_text(text):
    """Clean text by removing excessive whitespace and normalizing"""
    if pd.isna(text):
        return None
    text = str(text).strip()
    text = ' '.join(text.split())  # collapse multiple whitespaces
    return text if text else None

def row_to_json(row):
    """Convert a single row of Parquet data to structured JSON"""
    doc = {
        "metadata": {
            "identification": {
                "pravogovruNd": row.get('pravogovruNd'),
                "issuedByIPS": row.get('issuedByIPS'),
                "docdateIPS": row.get('docdateIPS'),
                "docNumberIPS": row.get('docNumberIPS'),
                "heading": clean_text(row.get('headingIPS')),
                "doc_type": row.get('doc_typeIPS'),
                "author": row.get('doc_author_normal_formIPS'),
                "signedBy": row.get('signedIPS'),
                "status": row.get('statusIPS'),
                "scrape_timestamp": row.get('actual_datetimeIPS'),
                "scrape_timestamp_human": row.get('actual_datetime_humanIPS'),
                "is_widely_used": bool(row.get('is_widely_used', 0))
            },
            "references": {
                "classifier": row.get('classifierByIPS'),
                "keywords": [kw.strip() for kw in str(row.get('keywordsByIPS', '')).split(',') 
                             if kw.strip()] if pd.notna(row.get('keywordsByIPS')) else None
            }
        },
        "content": {
            "text": clean_text(row.get('textIPS')),
            "tagged_text": clean_text(row.get('taggedtextIPS'))
        }
    }
    
    # Remove None values to reduce JSON size
    def remove_none(d):
        if isinstance(d, dict):
            return {k: remove_none(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [remove_none(v) for v in d if v is not None]
        else:
            return d
    
    return remove_none(doc)

def parquet_to_json(parquet_path, output_dir, batch_size=1000):
    """
    Convert Parquet files to JSON format for LLM fine-tuning.
    Processes in batches to handle large files efficiently.
    
    Args:
        parquet_path: Path to input Parquet file
        output_dir: Directory to save JSON files
        batch_size: Number of documents to process at once
    """
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in batches for memory efficiency
        total_docs = len(df)
        num_batches = (total_docs + batch_size - 1) // batch_size
        
        print(f"Converting {total_docs} documents to JSON format...")
        
        for batch_num in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_docs)
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_docs = []
            for idx, row in batch_df.iterrows():
                doc = row_to_json(row)
                doc["id"] = f"doc_{start_idx + idx}"  # Add unique ID
                batch_docs.append(doc)
            
            # Save batch to JSON file
            output_file = output_dir / f"documents_batch_{batch_num}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_docs, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully converted {total_docs} documents to JSON format")
        
        # Create a manifest file listing all batches
        manifest = {
            "dataset_name": "RussianLegalDocuments",
            "num_documents": total_docs,
            "num_batches": num_batches,
            "batch_files": [f"documents_batch_{i}.json" for i in range(num_batches)],
            "schema": {
                "metadata": {
                    "identification": "Document identification information",
                    "references": "Classification and keywords"
                },
                "content": {
                    "text": "Original document text",
                    "tagged_text": "Morphosyntactic tagged text (CONLL_U format)"
                }
            }
        }
        
        with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error converting Parquet to JSON: {e}")
        raise

if __name__ == "__main__":
    # Example usage - process all parquet files in a directory
    input_dir = Path('/content/RusLawOD/')
    output_base = './RusLawOD/json_output/'
    
    for parquet_file in input_dir.glob('*.parquet'):
        print(f"Processing {parquet_file.name}...")
        output_dir = Path(output_base) / parquet_file.stem
        parquet_to_json(parquet_file, output_dir)