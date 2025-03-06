#!/usr/bin/env python3
"""
This script is used to create a dataset for the Phi-2 model using the OpenAssistant dataset.
It implements a PyTorch Lightning DataModule with streaming capabilities.
"""
# Standard Library Imports
from typing import Optional, Dict, List

# Third Party Imports
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local Imports
from config import DataConfig


class OpenAssistantDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the OpenAssistant dataset.
    Supports streaming mode for efficient processing of large datasets.
    """
    def __init__(self, config: DataConfig):
        """
        Initialize the OpenAssistantDataModule.

        :param config: DataConfig object containing configuration parameters
        """
        super().__init__()
        self.tokenizer_name = config.tokenizer_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.shuffle_buffer_size = config.shuffle_buffer_size
        self.max_length = config.max_length
        self.streaming = config.streaming
        self.validation_split = config.validation_split
        self.pin_memory = config.pin_memory
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation
        """
        # Load dataset in streaming mode
        self.dataset = load_dataset(
            "OpenAssistant/oasst1",
            split="train",
            streaming=self.streaming
        )
        
        # Shuffle the streaming dataset
        self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        
        # Group messages into conversations and convert to instruction-response pairs
        self.dataset = self.dataset.map(
            self._buffer_and_process_conversations,
            batched=True,
            batch_size=1000,  # Process 1000 messages at a time
            remove_columns=self.dataset.column_names
        )
        
        # Create train/val split
        val_size = int(self.validation_split * self.shuffle_buffer_size)
        self.train_dataset = self.dataset.skip(val_size)
        self.val_dataset = self.dataset.take(val_size)

    def _buffer_and_process_conversations(self, batch: Dict) -> Dict[str, List]:
        """
        Buffer messages and reconstruct conversations to create instruction-response pairs.
        
        This function takes a batch of messages, groups them by conversation ID,
        and converts them into instruction-response pairs.
        
        Returns a dictionary with lists for each column in the dataset.
        """
        # Group messages by conversation ID
        conversations = {}
        for i in range(len(batch["message_id"])):
            message = {key: batch[key][i] for key in batch}
            tree_id = message["message_tree_id"]
            if tree_id not in conversations:
                conversations[tree_id] = []
            conversations[tree_id].append(message)
        
        # Process each conversation into examples
        texts = []
        for tree_id, messages in conversations.items():
            # Skip incomplete conversations
            if len(messages) < 2:
                continue
                
            # Sort messages to reconstruct conversation flow
            sorted_msgs = self._sort_messages_by_parent(messages)
            
            # Convert to instruction-response pairs
            for i in range(0, len(sorted_msgs)-1, 2):
                if i+1 >= len(sorted_msgs):
                    break
                    
                if sorted_msgs[i]["role"] == "prompter" and sorted_msgs[i+1]["role"] == "assistant":
                    instruction = sorted_msgs[i]["text"]
                    response = sorted_msgs[i+1]["text"]
                    
                    # Format as per Phi-2's preferred format
                    formatted_text = f"Instruct: {instruction}\nOutput: {response}"
                    texts.append(formatted_text)
        
        # Return dictionary of lists instead of list of dictionaries
        return {"text": texts}
    
    def _sort_messages_by_parent(self, messages: List[Dict]) -> List[Dict]:
        """
        Sort the messages by parent to reconstruct conversation flow.
        """
        # Create dictionary mapping message_id to message
        msg_dict = {msg["message_id"]: msg for msg in messages}
        # Find root message (with no parent)
        root = next((msg for msg in messages if msg["parent_id"] is None), None)
        if not root:
            return []
        
        # Reconstruct conversation tree (simplified as linear)
        sorted_msgs = [root]
        current = root
        while True:
            # Find children of current message
            children = [msg for msg in messages if msg["parent_id"] == current["message_id"]]
            if not children:
                break
            # Sort by rank if available, otherwise take first
            children.sort(key=lambda x: x.get("rank", 0) if x.get("rank") is not None else float('inf'))
            next_msg = children[0]
            sorted_msgs.append(next_msg)
            current = next_msg
            
        return sorted_msgs

    def collate_fn(self, batch):
        """
        Tokenize and prepare the batch for training.
        """
        texts = [item["text"] for item in batch]
        
        # Tokenize all texts in the batch
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = input_ids.clone()
        
        # Find the position of "Output:" to mask the instruction part for loss calculation
        output_text = "Output:"
        output_tokens = self.tokenizer(output_text, add_special_tokens=False).input_ids
        
        # Find where the output starts for each example
        for i in range(len(texts)):
            # Find where the output starts
            output_start = None
            for j in range(len(input_ids[i]) - len(output_tokens) + 1):
                if input_ids[i, j:j+len(output_tokens)].tolist() == output_tokens:
                    output_start = j + len(output_tokens)
                    break
            
            # Mask tokens before the response for loss calculation
            if output_start is not None:
                labels[i, :output_start] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        """
        Return validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )


# if __name__ == "__main__":
#     # Example usage
#     # Force CPU usage
#     import torch
#     torch.cuda.is_available = lambda: False
    
#     datamodule = OpenAssistantDataModule(pin_memory=False)
#     datamodule.setup()
    
#     # Get a batch from the train dataloader
#     train_dataloader = datamodule.train_dataloader()
#     batch = next(iter(train_dataloader))
    
#     print(f"Batch keys: {batch.keys()}")
#     print(f"Input shape: {batch['input_ids'].shape}")
#     print(f"Attention mask shape: {batch['attention_mask'].shape}")
#     print(f"Labels shape: {batch['labels'].shape}")

#     # Print 8 raw samples from the batch with input, attention mask, and labels
#     for i in range(8):
#         print(f"Sample {i+1}:")
#         print(f"Raw Input: {datamodule.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)}")
#         print(f"Input Tokens: {batch['input_ids'][i]}")
#         print(f"Attention Mask: {batch['attention_mask'][i]}")
        
#         # Filter out -100 values before decoding
#         valid_label_ids = batch['labels'][i].clone()
#         valid_label_ids[valid_label_ids == -100] = datamodule.tokenizer.pad_token_id
#         print(f"Raw Labels: {datamodule.tokenizer.decode(valid_label_ids, skip_special_tokens=True)}")
        
#         print(f"Label Tokens: {batch['labels'][i]}")
#         print("\n")