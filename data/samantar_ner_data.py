import torch
import json
import argparse
import time
import logging
from dataclasses import dataclass
import gc
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ner_data_creation.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CustoMDatasetConfig:
    dataset_name: str
    dataset_split: str
    dataset_dir: str

class BERTNERPipeline:

    def __init__(self, model_name: str, tok_name: str, batch_size: int):
        
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")

        self.batch_size = batch_size
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",  # Merge spans with same entity
            device=self.device
        )

        logger.info(f"NER pipeline with model {model_name} loaded successfully")
        logger.info(f"Batch size set to {batch_size}")

    def predict(self, texts):
        results = []
        ner_results = self.ner_pipeline(texts)

        if isinstance(texts, str) or (isinstance(texts, list) and len(texts) == 1):
            ner_results = [ner_results]

        for text, entities in zip(texts, ner_results):
            words = text.split()
            tags = ["O"] * len(words)  # Initialize all tags as "O" (Outside)

            for entity in entities:
                entity_text = entity['word']
                entity_tag = entity['entity_group']

                # Filter out MISC tags if needed
                if "MISC" in entity_tag:
                    continue

                # Find the word position for this entity
                start_char = entity["start"]
                end_char = entity["end"]

                word_count, char_count = 0,0
                
                for i, word in enumerate(words):
                    prev_char_count = char_count
                    char_count += len(word) + 1  # +1 for space
                    
                    # If this word contains the entity start position
                    if prev_char_count <= start_char < char_count:
                        tags[i] = entity_tag
                        
                        # If entity spans multiple words
                        if end_char > char_count:
                            j = i + 1
                            while j < len(words) and char_count < end_char:
                                tags[j] = entity_tag
                                char_count += len(words[j]) + 1
                                j += 1
                        
                        break
                
                results.append({"tokens": words, "ner_tags": tags})
            return results if len(texts) > 1 else results[0]

    def __del__(self):
        """Clean up resources when object is destroyed."""
        if hasattr(self, 'ner_pipeline'):
            del self.ner_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

## function for any other dataset
def annotate_dataset(split, limit, output_file:str, batch_size, model_name, custom_dataset_config: bool):
    if custom_dataset_config == True:
        dc = CustoMDatasetConfig(
            dataset_dir='', ## Enter dataset dir, it usually is any additional param. Leave it be if none
            dataset_name='', ## Dataset name
            dataset_split='', ## train, val, test etc
        )
        ds = load_dataset(dc.dataset_name, dc.dataset_dir) 
        
        total = len(ds[split])
        limit = min(limit, total) if limit else total
        logger.info(f"Initialised a custom dataset: {dc.datraset_name}")

    logger.info(f"Initializing NER pipeline with model {model_name}")
    ner_pipeline = BERTNERPipeline(model_name, batch_size=batch_size)

    results = []
    start_time = time.time()

    for i in tqdm(range(0, limit, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, limit)
        batch_examples = ds[split].select(range(i, batch_end))
        batch_texts = [example['src'] for example in batch_examples]

        batch_results = ner_pipeline.predict(batch_texts)
        results.append(batch_results)

    # Save results
    logger.info(f"Saving {len(results)} processed examples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds "
                f"({limit/elapsed_time:.2f} examples/second)")
    
    return results

## function for samantar
def annotate_samanantar(split, limit, output_file:str, batch_size, model_name):
    logger.info(f"Initializing NER pipeline with model {model_name}")
    ner_pipeline = BERTNERPipeline(model_name, batch_size=batch_size)

    ds = load_dataset('ai4bharat/samanantar', 'as')
    total = len(ds[split])
    limit = min(limit, total) if limit else total

    results = []
    start_time = time.time()
    for i in tqdm(range(0, limit, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, limit)
        batch_examples = ds[split].select(range(i, batch_end))
        batch_texts = [example['src'] for example in batch_examples]

        batch_results = ner_pipeline.predict(batch_texts)
        results.append(batch_results)

    # Save results
    logger.info(f"Saving {len(results)} processed examples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds "
                f"({limit/elapsed_time:.2f} examples/second)")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Data Creation Script with HF Pipeline")
    parser.add_argument("--split", type=str, required=True, help="Dataset split (train/val/test)")
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to process")
    parser.add_argument("--output", type=str, default="samantar_ner_data.json", help="Output filename")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--model_name", type=str, default="dslim/bert-base-NER", help="NER model to use")
    parser.add_argument("--dataset", type=str, default="None", help="Normal corpus we want to use for annotating data")
    
    args = parser.parse_args()

    if args.dataset is None:
        annotate_samanantar(
            split=args.split,
            limit=args.limit,
            output_file=args.output,
            batch_size=args.batch_size,
            lang=args.lang,
            model_name=args.model_name
        )
    else:
        annotate_samanantar(
            split=args.split,
            limit=args.limit,
            output_file=args.output,
            batch_size=args.batch_size,
            lang=args.lang,
            model_name=args.model
        )       
