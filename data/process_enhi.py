import json
import re
from typing import List
from tqdm import tqdm
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_id_to_label(data):
    label_column_name = "ner_tags"
    features = data['train'].features
    label_list = features[label_column_name].feature.names
    label_to_id = {label_list[i]: features[label_column_name].feature.str2int( label_list[i] ) for i in range(len(label_list))}
    id_to_label = {features[label_column_name].feature.str2int( label_list[i] ): label_list[i] for i in range(len(label_list))}

    return id_to_label, label_to_id

def load_json_dataset(file_name):
    with open(file_name, 'r') as json_file:
        contents = json.load(file_name)
    
    return contents

def download_data(lang: str = 'en', split: str = 'train'):
    en_dataset_name = 'unimelb-nlp/wikiann'
    hi_dataset_name = 'ai4bharat/naamapadam'

    if lang == 'en':
        data = load_dataset(en_dataset_name, 'en')
        idtl, ltoid = get_id_to_label(data)
        return data[split], idtl, ltoid

    elif lang == 'hi':
        data = load_dataset(hi_dataset_name, 'as')
        idtl, ltoid = get_id_to_label(data)
        return data[split], idtl, ltoid

def extract_entity_spans(entry, id_to_label):
    assert len(entry['tokens']) == len(entry['ner_tags'])  # Ensure alignment

    entity_spans = []
    current_span = []

    entity_label = None
    start_idx = None  # Track the starting index of an entity

    for idx, (token, tag) in enumerate(zip(entry['tokens'], entry['ner_tags'])):
        label = id_to_label[tag]

        if label.startswith('B-'):  # Start a new entity
            if current_span:  
                entity_spans.append([start_idx, idx - 1, entity_label])  # Store previous entity
            current_span = [token]
            entity_label = label.split('-')[1]
            start_idx = idx  # Mark start index of new    entity

        elif label.startswith('I-'):  # Continuation of an entity
            if entity_label == label.split('-')[1]:  # Ensure it's the same entity type
                current_span.append(token)
            else:  
                if current_span:
                    entity_spans.append([start_idx, idx - 1, entity_label])
                    
                current_span = [token]
                entity_label = label.split('-')[1]
                start_idx = idx  # Reset start index

        if idx % 10000 == 0:
            logger.info("Done procerssing 10000")
        
        if idx == 10000:
            break
    
    # Store any remaining entity
    if current_span:
        entity_spans.append([start_idx, len(entry['tokens']) - 1, entity_label])
    return {'ner': entity_spans, 'tokenized_text': entry['tokens']}

def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)

def mix_data(enfile: str, hifile: str):
    try:
        with open(enfile, "r", encoding="utf-8") as file1:
            data1 = json.load(file1)
        # Load second file
        with open(hifile, "r", encoding="utf-8") as file2:
            data2 = json.load(file2)

    except Exception as e:
        logger.error(f"Something is wrong with file extraction: {e}")

    # Determine the minimum length to ensure parallel structure
    min_length = min(len(data1), len(data2))

    # Interleave data from both sources
    mixed_data = []
    for i in range(min_length):
        mixed_data.append(data1[i])
        mixed_data.append(data2[i])

    # Add any remaining data (if one file is longer)
    mixed_data.extend(data1[min_length:])
    mixed_data.extend(data2[min_length:])

    # Save to a new file
    try:
        output_file = "gliner_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mixed_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f'Something is wrong with file outputting: {e}')

    logger.info(f'Parallel mixed NER data saved to {output_file}')

def main(langs: List[str], limit: int=1000):
    for lang in langs:
        
        file_name = f'{lang}_file.json'
        data, idtol, _ = download_data(lang)

        logger.info(f"Processing data for language: {lang}")

        loop_end = min(limit, len(data)) if limit != 0 else len(data)
        data = data.select(range(loop_end))
        
        all_data = [extract_entity_spans(entry, idtol) for entry in tqdm(data)]
        logger.info(f"Finished processing data for language: {lang}")
        
        save_data_to_file(all_data, filepath=file_name)

    mix_data('en_file.json', 'hi_file.json')
    
if __name__ == "__main__":
    # download the pile-ner data: "wget https://huggingface.co/datasets/Universal-NER/Pile-NER-type/blob/main/train.json"
    output_file = 'gliner_train_multi.json'
    
    languages = ['en', 'hi']
    main(langs=languages)