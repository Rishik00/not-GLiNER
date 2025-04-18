import torch
import json
import argparse
import time
import logging
import os
import gc
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
import psutil

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

class NERPipeline:
    def __init__(self, model_name, batch_size=8):
        """Initialize the NER pipeline with the given model name and batch size."""
        try:
            self.device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")
            
            # self.ner_pipeline = pipeline(
            #     "ner",
            #     model=model_name,
            #     aggregation_strategy="simple",  # Merge spans with same entity
            #     device=self.device
            # )

            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name, 
                device=self.device                
            )

            self.batch_size = batch_size
            
            # Log model information
            logger.info(f"NER pipeline with model {model_name} loaded successfully")
            logger.info(f"Batch size set to {batch_size}")
            
            if self.device == 0:
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    def predict(self, texts):
        """
        Process a list of texts for NER prediction.
        Returns a list of dictionaries with text and ner_tags.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            results = []
            
            # Process texts with the pipeline
            ner_results = self.ner_pipeline(texts)
            
            # If only one text was provided, wrap result in a list
            if isinstance(texts, str) or (isinstance(texts, list) and len(texts) == 1):
                ner_results = [ner_results]
                
            for text, entities in zip(texts, ner_results):
                words = text.split()
                tags = ["O"] * len(words)  # Initialize all tags as "O" (Outside)
                
                # Map entity predictions to words
                for entity in entities:
                    entity_text = entity["word"]
                    entity_tag = entity["entity"]
                    
                    # Filter out MISC tags if needed
                    if "MISC" in entity_tag:
                        continue
                    
                    # Find the word position for this entity
                    start_char = entity["start"]
                    end_char = entity["end"]
                    
                    # Count words up to the start position
                    word_count = 0
                    char_count = 0
                    
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
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return [{"text": [], "ner_tags": []}] * len(texts)

    def __del__(self):
        """Clean up resources when object is destroyed."""
        if hasattr(self, 'ner_pipeline'):
            del self.ner_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def create_ner_data_from_corpus(split, limit=None, batch_size=8, output_file="samantar_data.json", 
                               dataset_name="ai4bharat/samanantar", lang="as", model_name="dslim/bert-base-NER"):

    # Initialize pipeline
    logger.info(f"Initializing NER pipeline with model {model_name}")
    ner_pipeline = NERPipeline(model_name, batch_size=batch_size)
    
    # Load dataset
    logger.info(f"Loading dataset {dataset_name} (language: {lang})")
    try:
        ds = load_dataset(dataset_name, lang)
        logger.info(f"Dataset loaded successfully. Split sizes: {ds}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
    
    if split not in ds:
        logger.error(f"Split '{split}' not found in dataset. Available splits: {list(ds.keys())}")
        raise ValueError(f"Invalid split: {split}")
    
    # Determine how many examples to process
    total = len(ds[split])
    limit = min(limit, total) if limit is not None else total
    logger.info(f"Processing {limit} examples from {split} split (total available: {total})")
    
    # Process dataset
    results = []
    
    # Process in batches with progress bar
    logger.info(f"Starting NER processing")
    start_time = time.time()
    
    for i in tqdm(range(0, limit, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, limit)
        batch_examples = ds[split].select(range(i, batch_end))
        batch_texts = [example['src'] for example in batch_examples]
        
        try:
            # Process the batch
            batch_results = ner_pipeline.predict(batch_texts)
            results.extend(batch_results)
            
            # Log performance metrics periodically
            if i % 100 == 0 or batch_end == limit:
                elapsed = time.time() - start_time
                examples_processed = batch_end
                examples_per_second = examples_processed / elapsed if elapsed > 0 else 0
                
                # Save intermediate results
                if i > 0:
                    temp_file = f"{os.path.splitext(output_file)[0]}_temp.json"
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False)
                    logger.info(f"Saved intermediate results to {temp_file}")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
            continue
            
    # Save results
    logger.info(f"Saving {len(results)} processed examples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds "
                f"({limit/elapsed_time:.2f} examples/second)")
    
    return results

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return {
        "RAM": f"{process.memory_info().rss / (1024 * 1024):.2f} MB",
        "GPU": f"{torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB" if torch.cuda.is_available() else "N/A"
    }

def clean_data():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Data Creation Script with HF Pipeline")
    parser.add_argument("--split", type=str, required=True, help="Dataset split (train/val/test)")
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to process")
    parser.add_argument("--output", type=str, default="samantar_ner_data.json", help="Output filename")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--dataset", type=str, default="ai4bharat/samanantar", help="HuggingFace dataset name")
    parser.add_argument("--lang", type=str, default="as", help="Language code for the dataset")
    parser.add_argument("--model", type=str, default="dslim/bert-base-NER", help="NER model to use")
    
    args = parser.parse_args()
    
    logger.info(f"Starting script with arguments: {args}")
    logger.info(f"Initial memory usage: {get_memory_usage()}")
    
    try:
        start_time = time.time()
        
        create_ner_data_from_corpus(
            split=args.split,
            limit=args.limit,
            output_file=args.output,
            batch_size=args.batch_size,
            dataset_name=args.dataset,
            lang=args.lang,
            model_name=args.model
        )
        
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        logger.info(f"Final memory usage: {get_memory_usage()}")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
