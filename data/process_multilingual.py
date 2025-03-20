import json
import re
import ast
from tqdm import tqdm
from datasets import load_dataset

def load_datasets(langs, name, split):
    dataset_dict = {f'dataset_{da}': None for da in langs}
    for da in langs:
        data = load_dataset(name, da, split)
        dataset_dict[f'dataset_{da}'] = data
    return dataset_dict

def tokenize_text(text, lang):
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

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

def extract_entity_spans(en, hi, entry, split='train'):
    for en_entity, hi_entity in zip(en, hi):
        pass

    processed_data=[]
    save_data_to_file(processed_data, output_file)

def process_data(data):
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
