import json
from huggingface_hub import HfApi, HfFolder

# Replace with your Hugging Face token and repository details
HF_TOKEN = "your_huggingface_token"
REPO_ID = "username/repo_name"
FILE_PATH = "path_to_your_file.json"

def validate_ner_dataset(file_path):
    """Validate that the JSON file contains 'ner_tags' and 'tokens' fields."""
    with open(file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The dataset should be a list of examples.")

    for example in data:
        if not isinstance(example, dict):
            raise ValueError("Each example should be a dictionary.")
        if "ner_tags" not in example or "tokens" not in example:
            raise ValueError("Each example must contain 'ner_tags' and 'tokens' fields.")
        if not isinstance(example["ner_tags"], list) or not isinstance(example["tokens"], list):
            raise ValueError("'ner_tags' and 'tokens' must be lists.")
        if len(example["ner_tags"]) != len(example["tokens"]):
            raise ValueError("'ner_tags' and 'tokens' must have the same length.")

    print("Validation successful: The dataset is properly formatted.")

def push_to_huggingface(file_path, repo_id, token):
    api = HfApi()
    HfFolder.save_token(token)

    validate_ner_dataset(file_path)

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print(f"File {file_path} successfully uploaded to {repo_id}.")

if __name__ == "__main__":
    push_to_huggingface(FILE_PATH, REPO_ID, HF_TOKEN)