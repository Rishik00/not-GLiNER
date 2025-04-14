import json
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import logging
from datasets import load_dataset, Dataset
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NERDataConfig:
    languages: List[str]
    limit: int = 0  # 0 means all data
    output_file: str = "gliner_train_multi.json"
    intermediate_dir: str = "./"
    use_local_files: bool = False
    local_files: Dict[str, str] = None  # Map of language -> file path

def get_id_to_label(data):
    label_column_name = "ner_tags"
    features = data['train'].features

    label_list = features[label_column_name].feature.names
    label_to_id = {label_list[i]: features[label_column_name].feature.str2int(label_list[i]) for i in range(len(label_list))}
    id_to_label = {features[label_column_name].feature.str2int(label_list[i]): label_list[i] for i in range(len(label_list))}
    
    return id_to_label, label_to_id

def load_json_dataset(file_path: str) -> Tuple[List[Dict], Dict[int, str]]:
    logger.info(f"Loading dataset from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def download_data(lang: str = 'en', split: str = 'train') -> Tuple[Dataset, Dict[int, str], Dict[str, int]]:
    """Download dataset for the specified language and split."""
    try:
        dataset_configs = {
            'en': ('unimelb-nlp/wikiann', 'en'),
            'hi': ('ai4bharat/naamapadam', 'as')
        }
        
        if lang not in dataset_configs:
            raise ValueError(f"Unsupported language: {lang}. Supported languages: {list(dataset_configs.keys())}")
            
        dataset_name, config = dataset_configs[lang]
        logger.info(f"Downloading {lang} dataset from {dataset_name}")
        
        data = load_dataset(dataset_name, config)
        idtl, ltoid = get_id_to_label(data)
        
        return data[split], idtl, ltoid
    except Exception as e:
        logger.error(f"Error downloading data for {lang}: {e}")
        raise

def extract_entity_spans(entry, id_to_label, is_local_data=False):
    """Extract entity spans from tokens and NER tags."""
    try:
        # If this is already in the proper format from a local JSON file
        if is_local_data:
            return entry
            
        if len(entry['tokens']) != len(entry['ner_tags']):
            logger.warning("Tokens and NER tags are not aligned")
            return {'ner': [], 'tokenized_text': entry['tokens']}

        entity_spans = []
        current_span = []
        entity_label = None
        start_idx = None

        for idx, (token, tag) in enumerate(zip(entry['tokens'], entry['ner_tags'])):
            if id_to_label:
                label = id_to_label[tag]
            
            else:
                label = tag

            if label.startswith('B-'):  # Start a new entity
                if current_span:  
                    entity_spans.append([start_idx, idx - 1, entity_label])
                    
                current_span = [token]
                entity_label = label.split('-')[1]
                start_idx = idx

            elif label.startswith('I-'):  # Continuation of entity
                if current_span and entity_label == label.split('-')[1]:
                    current_span.append(token)
                else:
                    if current_span:
                        entity_spans.append([start_idx, idx - 1, entity_label])

                    current_span = [token]
                    entity_label = label.split('-')[1]
                    start_idx = idx
            elif label == 'O' and current_span:  # End of entity
                entity_spans.append([start_idx, idx - 1, entity_label])
                current_span = []
                entity_label = None
                start_idx = None
        
        # Store any remaining entity
        if current_span:
            entity_spans.append([start_idx, len(entry['tokens']) - 1, entity_label])
            
        return {'ner': entity_spans, 'tokenized_text': entry['tokens']}
    except Exception as e:
        logger.error(f"Error in extract_entity_spans: {e}")
        return {'ner': [], 'tokenized_text': entry.get('tokens', [])}

def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    try:
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")

def mix_data(config: NERDataConfig):
    """Mix data from multiple language files."""
    try:
        all_data = []
        for lang in config.languages:
            file_path = Path(config.intermediate_dir) / f"{lang}_file.json"
            logger.info(f"Loading data from {file_path}")
            
            if not file_path.exists():
                logger.warning(f"File {file_path} does not exist, skipping")
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                lang_data = json.load(f)
                all_data.append(lang_data)

        if not all_data:
            logger.error("No data to mix")
            return

        # Determine minimum length and interleave
        min_length = min(len(data) for data in all_data)
        
        mixed_data = []
        for i in range(min_length):
            for lang_data in all_data:
                mixed_data.append(lang_data[i])
        
        # Add remaining data
        for lang_data in all_data:
            mixed_data.extend(lang_data[min_length:])

        output_file = config.output_file
        save_data_to_file(mixed_data, output_file)
        logger.info(f'Mixed NER data saved to {output_file}')

    except Exception as e:
        logger.error(f"Error mixing data: {e}")

def process_language(config: NERDataConfig, lang: str):
    """Process data for a single language."""
    try:
        file_name = Path(config.intermediate_dir) / f'{lang}_file.json'
        
        # Check if we should use local files
        if config.use_local_files and config.local_files and lang in config.local_files:
            local_file = config.local_files[lang]
            logger.info(f"Using local file for {lang}: {local_file}")
            data = load_json_dataset(local_file)
            
            loop_end = min(config.limit, len(data)) if config.limit != 0 else len(data)
            data = data[:loop_end]
            
            # If using local files, the data may already be in the correct format
            all_data = []
            for entry in tqdm(data, desc=f"Processing {lang}"):
                processed_entry = extract_entity_spans(entry, id_to_label, is_local_data=True)
                all_data.append(processed_entry)
        else:
            # Use Hugging Face datasets
            data, id_to_label, _ = download_data(lang)
            
            loop_end = min(config.limit, len(data)) if config.limit != 0 else len(data)
            data = data[:loop_end]
            
            all_data = []
            for entry in tqdm(data, desc=f"Processing {lang}"):
                processed_entry = extract_entity_spans(entry, id_to_label)
                all_data.append(processed_entry)
        
        logger.info(f"Finished processing {len(all_data)} entries for language: {lang}")
        save_data_to_file(all_data, filepath=file_name)
        return file_name
    
    except Exception as e:
        logger.error(f"Error processing language {lang}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process NER data from multiple languages")
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "hi"],
                        help="Languages to process (space-separated)")
    parser.add_argument("--limit", type=int, default=0, 
                        help="Limit number of examples per language (0 for all)")
    parser.add_argument("--output", type=str, default="gliner_train_multi.json",
                        help="Output file for mixed data")
    parser.add_argument("--intermediate-dir", type=str, default="./",
                        help="Directory to store intermediate files")
    parser.add_argument("--use-local-files", action="store_true",
                        help="Use local files instead of downloading from HF")
    parser.add_argument("--local-file-en", type=str, help="Path to local English NER dataset")
    parser.add_argument("--local-file-hi", type=str, help="Path to local Hindi NER dataset")
    
    args = parser.parse_args()
    
    # Build the local_files dictionary from args
    local_files = {}
    if args.local_file_en:
        local_files['en'] = args.local_file_en
    if args.local_file_hi:
        local_files['hi'] = args.local_file_hi
    
    config = NERDataConfig(
        languages=args.languages,
        limit=args.limit,
        output_file=args.output,
        intermediate_dir=args.intermediate_dir,
        use_local_files=args.use_local_files,
        local_files=local_files
    )
    
    logger.info(f"Processing languages: {config.languages}")
    
    for lang in config.languages:
        process_language(config, lang)
    
    mix_data(config)
    
if __name__ == "__main__":
    main()