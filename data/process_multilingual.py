import json
import re
from tqdm import tqdm
import logging
from datasets import load_dataset

def load_datasets(langs, name, split, do_eval):
    endata = load_dataset(name, split=split, languages=langs[0])
    hi_data = load_dataset(name, split=split, languages=langs[1])   

    en_data = endata['train'] if split == 'train' else endata['test']
    hi_data = hi_data['train'] if split == 'train' else hi_data['test']

    if do_eval:
        en_data_val = en_data['validation'] if split == 'train' else en_data['train']
        hi_data_val = hi_data['validation'] if split == 'train' else hi_data['train']   
        return en_data, hi_data, en_data_val, hi_data_val
        
    return en_data, hi_data

def random_sampling(seed):
    pass

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

    # Store any remaining entity
    if current_span:
        entity_spans.append([start_idx, len(entry['tokens']) - 1, entity_label])

    return {'ner': entity_spans, 'tokenized_text': entry['tokens']}

def process_data_en(data):
    """Processes a list of data entries to extract entity spans."""
    all_data = [extract_entity_spans(entry) for entry in tqdm(data)]
    return all_data

def process_data_hi(data):
    """Processes a list of data entries to extract entity spans."""
    all_data = [extract_entity_spans(entry) for entry in tqdm(data)]
    return all_data

def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    # download the pile-ner data: "wget https://huggingface.co/datasets/Universal-NER/Pile-NER-type/blob/main/train.json"
    input_dataset = 'ai4bharat/naamapadam'
    output_file = 'gliner_train_multi.json'
    languages = ['en', 'hi']

    data = load_datasets(languages, input_dataset)
    print(data)
