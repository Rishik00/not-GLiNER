import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class HFModel:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)    
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)

    def predict(self, x):
        tokenized_text = self.tokenizer.tokenize(x)
        ner_results = self.nlp(x)
        res = ["O"] * len(tokenized_text)  # Default to "O" (non-entity)

        # Map token indices to NER predictions
        token_index = 0
        for result in ner_results:
            word = result['word']
            entity = result['entity']
            
            # Ignore "MISC" entities
            if "MISC" in entity:
                continue

            # Find the matching token in tokenized_text
            while token_index < len(tokenized_text):
                tokenized_word = tokenized_text[token_index].replace("##", "")  
                if tokenized_word == word:
                    res[token_index] = entity  # Assign NER label
                    token_index += 1  # Move to the next token
                    break
                token_index += 1

        return tokenized_text, res

def create_ner_data_from_corpus(output_file="samantar_data.json"):
    results = []
    ds = load_dataset("ai4bharat/samanantar", "as")
    model = HFModel("dslim/bert-large-NER")
    
    for i in range(10000):
        example = ds['train'][i]
        text = example['src']
        tok_text, res = model.predict(text)
        results.append({
            'tokens': tok_text, 
            'ner_tags': res
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"NER data saved to {output_file}")

if __name__ == "__main__":
    create_ner_data_from_corpus()