import os
import json
from datasets import DatasetDict
import pandas as pd
from datasets import load_dataset, Value, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator
from data_collator import bart_data_collator

def exists(x):
    return x is not None

def get_dataset(dataset_name, metadata=False, synthetic_train_path=None):
    if dataset_name == 'umls':
        umls_data_path = "datasets/umls"
        train_file = os.path.join(umls_data_path, "train.jsonl")
        valid_file = os.path.join(umls_data_path, "valid.jsonl")
        test_file = os.path.join(umls_data_path, "test.jsonl")

        def prepare_umls_dataset(json_file):
            with open(json_file, "r") as f:
                data = [
                    {"text": entry["trg"], "context": entry["src"]}
                    for entry in (json.loads(line.strip()) for line in f if line.strip())
                ]
            return data

        # Load and process the train, valid, and test files
        train_data = prepare_umls_dataset(train_file)
        valid_data = prepare_umls_dataset(valid_file)
        test_data = prepare_umls_dataset(test_file)

        # Create Hugging Face datasets
        train_dataset = Dataset.from_list(train_data)
        valid_dataset = Dataset.from_list(valid_data)
        test_dataset = Dataset.from_list(test_data)

        # Wrap into DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset
        })
        print('train sample:', dataset['train'][0])
        print('valid sample:', dataset['valid'][0])

    elif dataset_name == 'mesh':
        # Paths to your JSONL files
        mesh_data_path = "datasets/mesh"
        train_file = os.path.join(mesh_data_path, "train.jsonl")
        valid_file = os.path.join(mesh_data_path, "valid.jsonl")
        test_file = os.path.join(mesh_data_path, "test.jsonl")

        def prepare_mesh_dataset(json_file):
            with open(json_file, "r") as f:
                data = [
                    {"text": entry["trg"], "context": entry["src"]}
                    for entry in (json.loads(line.strip()) for line in f if line.strip())
                ]
            return data

        # Load and process the train, valid, and test files
        train_data = prepare_mesh_dataset(train_file)
        valid_data = prepare_mesh_dataset(valid_file)
        test_data = prepare_mesh_dataset(test_file)

        # Create Hugging Face datasets
        train_dataset = Dataset.from_list(train_data)
        valid_dataset = Dataset.from_list(valid_data)
        test_dataset = Dataset.from_list(test_data)

        # Wrap into DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset
        })
    else:
        raise NotImplementedError
    return dataset

def get_dataloader(args, dataset, model_config, tokenizer, max_seq_len, mode='diffusion', shuffle=True, context_tokenizer=None):
    def tokenization(example):
        if mode == 'diffusion' and args.dataset_name in {'umls', 'mesh'}:
            assert context_tokenizer is not None
            source = example['context']
            target = example['text']

            cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)
            model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)
            
            # Add model target to model inputs
            for k in cond_inputs.keys():
                model_inputs[f'cond_{k}'] = cond_inputs[k]
            
            return model_inputs
        else:
            text = example["text"]
            print('inside dataloader:',text)
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)

    collate_fn=bart_data_collator(tokenizer, model_config.decoder_start_token_id)
    
    dataset = dataset.map(tokenization, remove_columns=['text', 'context'], batched=True, num_proc=None)
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 4
        )
    return dl

if __name__ == "__main__":

    dataset = get_dataset('umls')
    print(dataset['train'][0])