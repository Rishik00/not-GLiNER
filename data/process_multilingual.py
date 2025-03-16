import json
import re
import ast
from tqdm import tqdm
from datasets import load_dataset

def tokenize_text(text):
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def extract_entity_spans(entry):
    pass

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
    input_file = ''
    output_file = 'gliner_train_multi.json'
    data = None
    processed_data = process_data(data)
    save_data_to_file(processed_data, output_file)

    print("dataset size:", len(processed_data))