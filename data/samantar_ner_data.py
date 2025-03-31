import torch
import json
import argparse
import time
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ner_data_creation.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device set to {DEVICE}")

class HFModel:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)    
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)

    def predict(self, x):
        ner_results = self.nlp(x)
        res = ["O"] * len(ner_results)  # Default to "O" (non-entity)
        words = []

        # Map token indices to NER predictions
        for idx, result in enumerate(ner_results):
            word = result['word']
            entity = result['entity']

            if "MISC" in entity:
                result['entity'] = 'O'
            
            if entity != 'O':
                res[idx] = entity

            words.append(word)
        return {'text': word, 'ner_tags': res}

def create_ner_data_from_corpus(split: str, limit: int, output_file: str="samantar_data.json"):
    results = []
    ## Change as or remove it to get the full dataset
    ds = load_dataset("ai4bharat/samanantar", "as")
    model = HFModel("dslim/bert-base-NER")
    
    for i in range(limit):
        example = ds[split][i]
        text = example['src']

        res = model.predict(text)
        results.append(res)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"NER data saved to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="English NER Data Creation Script")
    parser.add_argument("--split", type=str, help="split - train/val/test")
    parser.add_argument("--limit", type=int, help="Limit for nmumber of rows")
    parser.add_argument("--output", type=str, help="Output file name (JSON)")
    
    args = parser.parse_args()
    start_time = time.time()

    create_ner_data_from_corpus(split=args.split,limit=args.output, output_file=args.output)

    end_time = time.time()

