import json
import argparse
from typing import List
from tqdm import tqdm
import logging
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_json: bool = True

def get_id_to_label(data):
    logger.info("Extracting label to ID mappings.")
    label_column_name = "ner_tags"
    features = data['train'].features
    label_list = features[label_column_name].feature.names
    label_to_id = {label: features[label_column_name].feature.str2int(label) for label in label_list}
    id_to_label = {features[label_column_name].feature.str2int(label): label for label in label_list}
    logger.info("Label mappings extracted.")
    return id_to_label, label_to_id

def load_json_dataset(file_name):
    logger.info(f"Loading JSON dataset from: {file_name}")
    with open(file_name, 'r') as json_file:
        contents = json.load(json_file)
    logger.info("JSON data loaded successfully.")
    return contents

def download_data(lang: str = 'en', split: str = 'train'):
    en_dataset_name = 'unimelb-nlp/wikiann'
    hi_dataset_name = 'ai4bharat/naamapadam'

    logger.info(f"Downloading data for language: {lang}")
    if lang == 'en':
        if load_json:
            logger.info("Using JSON file instead of Huggingface dataset for English.")
            data = load_json_dataset(file_name='samantar_ner_data.json')
            return data, None, None
        data = load_dataset(en_dataset_name, 'en')
        idtl, ltoid = get_id_to_label(data)
        return data[split], idtl, ltoid

    elif lang == 'hi':
        logger.info("Using Huggingface dataset for Hindi.")
        data = load_dataset(hi_dataset_name, 'as')
        idtl, ltoid = get_id_to_label(data)
        return data[split], idtl, ltoid

def extract_entity_spans(entry, id_to_label):
    assert len(entry['tokens']) == len(entry['ner_tags']), "Mismatch between tokens and NER tags"
    entity_spans = []
    current_span = []
    entity_label = None
    start_idx = None

    for idx, (token, tag) in enumerate(zip(entry['tokens'], entry['ner_tags'])):
        label = id_to_label[tag] if id_to_label else tag

        if label.startswith('B-'):
            if current_span:
                entity_spans.append([start_idx, idx - 1, entity_label])
            current_span = [token]
            entity_label = label.split('-')[1]
            start_idx = idx

        elif label.startswith('I-'):
            if entity_label == label.split('-')[1]:
                current_span.append(token)
            else:
                if current_span:
                    entity_spans.append([start_idx, idx - 1, entity_label])
                current_span = [token]
                entity_label = label.split('-')[1]
                start_idx = idx

    if current_span:
        entity_spans.append([start_idx, len(entry['tokens']) - 1, entity_label])

    return {'ner': entity_spans, 'tokenized_text': entry['tokens']}

def save_data_to_file(data, filepath):
    logger.info(f"Saving data to file: {filepath}")
    with open(filepath, 'w') as f:
        json.dump(data, f)
    logger.info(f"Data successfully saved to {filepath}")

def mix_data(enfile: str, hifile: str):
    try:
        logger.info(f"Loading English data from {enfile}")
        with open(enfile, "r", encoding="utf-8") as file1:
            data1 = json.load(file1)

        logger.info(f"Loading Hindi data from {hifile}")
        with open(hifile, "r", encoding="utf-8") as file2:
            data2 = json.load(file2)

    except Exception as e:
        logger.error(f"File loading failed: {e}")
        return

    logger.info("Mixing English and Hindi data.")
    min_length = min(len(data1), len(data2))
    mixed_data = []

    for i in range(min_length):
        mixed_data.append(data1[i])
        mixed_data.append(data2[i])

    mixed_data.extend(data1[min_length:])
    mixed_data.extend(data2[min_length:])

    output_file = "gliner_data.json"
    try:
        logger.info(f"Saving mixed data to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mixed_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save mixed data: {e}")
        return

    logger.info(f"Mixed NER data saved to {output_file}")

def main(langs: List[str], limit: int = 1000):
    for lang in langs:
        logger.info(f"Starting processing for language: {lang}")
        file_name = f'{lang}_file.json'
        data, idtol, _ = download_data(lang)

        loop_end = min(limit, len(data)) if limit != 0 else len(data)
        if hasattr(data, 'select'):
            logger.info(f"Selecting first {loop_end} samples using 'select'")
            data = data.select(range(loop_end))
        else:
            logger.info(f"Selecting first {loop_end} samples using slicing")
            data = data[:loop_end]

        all_data = [extract_entity_spans(entry, idtol) for entry in tqdm(data, desc=f"Processing {lang}")]
        logger.info(f"Finished extracting spans for {lang}.")

        save_data_to_file(all_data, filepath=file_name)

    mix_data('en_file.json', 'hi_file.json')

if __name__ == "__main__":
    logger.info("Starting the multilingual NER processing pipeline.")
    languages = ['en', 'hi']
    main(langs=languages)
    logger.info("All tasks completed.")
