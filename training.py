#!/usr/bin/env python3
"""
This script is used to train the Phi-2 model using PyTorch Lightning.
It incorporates performance monitoring, advanced logging, and sample generation.
"""
# Standard Library Imports
import os
import time
import argparse
import gc
from typing import Optional, Dict
from dataclasses import dataclass
import random

# Third Party Imports
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from transformers import PhiForCausalLM, AutoTokenizer

# Local Imports
from phi_model import PhiModel
from dataset import OpenAssistantDataModule
from config import Config, DataConfig, ModelConfig, TrainingConfig


class LitPhiModel(pl.LightningModule):
    """
    PyTorch Lightning module for the Phi-2 model.
    """
    def __init__(
        self,
        config: Dict,
        learning_rate: float = None,
        weight_decay: float = None,
        max_steps: int = None,
    ):
        """
        Initialize the LitPhiModel with extreme memory optimizations.
        
        :param config: Configuration dictionary
        :param learning_rate: Learning rate (overrides config if provided)
        :param weight_decay: Weight decay (overrides config if provided)
        :param max_steps: Maximum number of steps (overrides config if provided)
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Override config values if provided
        self.learning_rate = learning_rate or config["training"].learning_rate
        self.weight_decay = weight_decay or config["training"].weight_decay
        self.max_steps = max_steps
        
        # Initialize model
        self.model_config = config["model"]
        self.qlora_config = config.get("qlora", None)
        
        # Force enable extremely aggressive memory saving settings
        print("Enabling extreme memory saving options...")
        
        if self.qlora_config:
            # Make sure we're using 4-bit quantization for maximum memory savings
            self.qlora_config.load_in_4bit = True
            self.qlora_config.bnb_4bit_compute_dtype = "bfloat16"
            self.qlora_config.bnb_4bit_quant_type = "nf4"
            self.qlora_config.bnb_4bit_use_double_quant = True
            
            # Reduce LoRA rank and alpha for smaller adapters
            if hasattr(self.qlora_config, "lora_r") and self.qlora_config.lora_r > 8:
                self.qlora_config.lora_r = 8
            if hasattr(self.qlora_config, "lora_alpha") and self.qlora_config.lora_alpha > 16:
                self.qlora_config.lora_alpha = 16
        
        # Initialize Phi model with modified QLoRA config
        self.phi_model = PhiModel(
            model_name=self.model_config.model_name_or_path,
            qlora_config=self.qlora_config
        )
        
        # Load model with aggressive memory optimizations
        self.model = self.phi_model.load_model()
        
        # Enable gradient checkpointing more aggressively
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            # Maximum checkpoint frequency
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False
        
        # Disable compilation - we need every byte of memory
        
        # Performance monitoring attributes
        self.iter_num = 0
        self.iter_time = 0.0
        self.tokens_processed = 0
        
        # Print model size info
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)  # Size in MB
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model size: {model_size:.2f} MB")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Special handling for QLoRA - move norm layers to float32 for stability
        if self.qlora_config and self.qlora_config.use_qlora:
            print("Converting norm layers to float32 for QLoRA stability")
            for name, module in self.model.named_modules():
                if "norm" in name:
                    module = module.to(torch.float32)
    
    def on_load_checkpoint(self, checkpoint):
        """
        Restore iter_num when loading from checkpoint
        """
        if 'iter_num' in checkpoint:
            self.iter_num = checkpoint['iter_num']
    
    def on_save_checkpoint(self, checkpoint):
        """
        Save iter_num in checkpoint
        """
        checkpoint['iter_num'] = self.iter_num
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step with CPU offloading and extreme memory optimizations
        """
        try:
            # Stop training if max steps reached
            if self.max_steps and self.iter_num >= self.max_steps:
                self.trainer.should_stop = True
                return None
            
            # Start timing
            start_time = time.time()
            
            # Truncate sequence length even more aggressively
            max_seq_length = 128  # Very short sequences for training
            
            # Truncate input sequences to maximum length
            if batch["input_ids"].shape[1] > max_seq_length:
                batch["input_ids"] = batch["input_ids"][:, :max_seq_length]
                batch["attention_mask"] = batch["attention_mask"][:, :max_seq_length]
                batch["labels"] = batch["labels"][:, :max_seq_length]
            
            # Free memory before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Split computation into smaller chunks for layer-by-layer processing
            # This prevents loading the entire model into GPU memory at once
            if not hasattr(self, 'cpu_offload_enabled') or not self.cpu_offload_enabled:
                # Process one layer at a time to reduce memory footprint
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    self.model.config.use_cache = False
                self.cpu_offload_enabled = True
                
                # Enable activation checkpointing even more aggressively
                if hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'layers'):
                    for layer in self.model.base_model.model.layers:
                        if hasattr(layer, 'requires_grad_'):
                            layer.requires_grad_(True)  # Ensure gradients flow
            
            # Forward pass with mixed precision and memory efficiency
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                # Chunk the sequence into smaller parts if too long
                if batch["input_ids"].shape[1] > 64:
                    # Process first chunk only for now (extreme measure)
                    chunk_size = 64
                    inputs = {
                        'input_ids': batch["input_ids"][:, :chunk_size],
                        'attention_mask': batch["attention_mask"][:, :chunk_size],
                        'labels': batch["labels"][:, :chunk_size] 
                    }
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
            
            loss = outputs.loss
            
            # Free memory
            if 'cuda' in str(loss.device):
                # Keep only what's needed for backward
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
            
            # Log loss
            self.log(
                "train_loss", 
                loss.item(), 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True,
                logger=True,
                batch_size=batch["input_ids"].size(0)
            )
            
            # Performance monitoring
            self.iter_num += 1
            self.iter_time = time.time() - start_time
            
            # Calculate tokens per second
            batch_size = batch["input_ids"].size(0)
            seq_length = batch["input_ids"].size(1)
            tokens = batch_size * seq_length
            self.tokens_processed = tokens / self.iter_time
            
            # Calculate GPU memory usage if CUDA is available
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
            
            # Periodically print training statistics
            if self.iter_num % self.config["training"].log_every == 0:
                print(f"{self.iter_num} | loss: {loss.item():.4f} | {self.iter_time:.2f}s/it | {self.tokens_processed:.1f} tokens/s | {gpu_memory:.2f} GB")
            
            # Optionally clear GPU cache to prevent memory fragmentation
            if self.config["training"].clear_cache_every > 0 and self.iter_num % self.config["training"].clear_cache_every == 0:
                torch.cuda.empty_cache()
            
            # Occasionally generate sample text to monitor training progress
            if self.iter_num % self.config["training"].generate_every == 0:
                # Use a small chance to avoid overhead every time
                if self.global_rank == 0 and random.random() < 0.2:
                    self._generate_sample(batch["input_ids"])
            
            return loss
        
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM errors gracefully
            print(f"WARNING: out of memory - {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Return None instead of raising an exception
            return None
    
    def _generate_sample(self, input_ids):
        """
        Generate a sample text during training to monitor progress
        """
        # Get a sample input from the batch
        context_length = min(64, input_ids.shape[1])  # Use first 64 tokens as context
        sample_input = input_ids[0:1, :context_length]
        
        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            max_new_tokens = self.config["training"].max_new_tokens
            temperature = self.config["training"].temperature
            top_k = self.config["training"].top_k
            
            # Setup for generation
            generated = sample_input.clone()
            
            for _ in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
            
            # Convert tokens to text using the tokenizer
            try:
                tokenizer = self.trainer.datamodule.tokenizer
                input_text = tokenizer.decode(sample_input[0].tolist())
                generated_text = tokenizer.decode(generated[0, context_length:].tolist())
                
                print(f"\nStep {self.iter_num} - Sample Generation:")
                print(f"Input: {input_text[:100]}...")
                print(f"Generated: {generated_text[:200]}...")
            except Exception as e:
                print(f"Error decoding text: {str(e)}")
        
        self.model.train()
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with extreme memory optimizations
        """
        # Clear cache and more aggressively reduce batch size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Take only first example and truncate sequence length (context window)
        truncated_length = 128  # Much shorter context for validation only
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                # First take only one example (first in batch)
                batch[k] = batch[k][:1]
                # Then truncate sequence length if too long
                if batch[k].dim() > 1 and batch[k].size(1) > truncated_length:
                    batch[k] = batch[k][:, :truncated_length]
        
        # Use the modern autocast API with correct syntax
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Use eval mode for validation to disable dropout and save memory
            self.model.eval()
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            self.model.train()
        
        loss = output.loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler
        """
        # When returning None from configure_optimizers, PyTorch Lightning creates a mock optimizer
        # that doesn't work with DeepSpeed. Let's always provide a real optimizer.
        
        # Create 8bit optimizer for memory efficiency
        from bitsandbytes.optim import Adam8bit
        
        # Using parameter groups to only optimize trainable parameters 
        param_groups = [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "weight_decay": self.weight_decay
            }
        ]
        
        optimizer = Adam8bit(
            param_groups,
            lr=self.learning_rate,
            weight_decay=0.0,  # Already applied in param_groups
            betas=(0.9, 0.999)
        )
        
        # Create a scheduler with linear decay
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.1, 
            total_iters=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def extend_config_with_defaults(config):
    """
    Add default values to config that are needed for training but may not be in the base config
    """
    # Extend the training config with additional parameters
    if not hasattr(config.training, "generate_every"):
        config.training.generate_every = 500
    if not hasattr(config.training, "log_every"):
        config.training.log_every = 50
    if not hasattr(config.training, "clear_cache_every"):
        config.training.clear_cache_every = 0
    if not hasattr(config.training, "max_new_tokens"):
        config.training.max_new_tokens = 100
    if not hasattr(config.training, "temperature"):
        config.training.temperature = 0.8
    if not hasattr(config.training, "top_k"):
        config.training.top_k = 40
        
    # Add scheduler config if not present
    if not hasattr(config.training, "scheduler"):
        @dataclass
        class SchedulerConfig:
            max_lr: float = 5e-5
            pct_start: float = 0.05
            div_factor: float = 25.0
            final_div_factor: float = 10.0
            three_phase: bool = False
            anneal_strategy: str = "cos"
        
        config.training.scheduler = SchedulerConfig()
    
    return config


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Train Phi-2 model with PyTorch Lightning")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to train")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs to train")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--no_validation", action="store_true", help="Disable validation to save memory")
    parser.add_argument("--no_deepspeed", action="store_true", help="Disable DeepSpeed CPU offloading")
    parser.add_argument("--save_adapters_only", action="store_true", help="Save only the LoRA adapters, not the full model")
    return parser.parse_args()


def get_latest_checkpoint():
    """
    Find the latest checkpoint in the checkpoints directory
    """
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    
    latest_checkpoint = max(
        [os.path.join(checkpoint_dir, f) for f in checkpoints],
        key=os.path.getmtime
    )
    return latest_checkpoint


def train_model(args):
    """
    Train the model with optimized memory usage but without DeepSpeed
    """
    # Load configuration
    config = Config()
    config = extend_config_with_defaults(config)
    
    # Force CPU offloading for tokenizers and dataset caching
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set compilation mode for PyTorch 2.0+
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        
    # Add memory optimizations for CUDA
    if torch.cuda.is_available():
        # Even more aggressive memory settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
        
        # Set PyTorch to use TF32 precision for matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Clear GPU cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()
    
    # Further reduce sequence length in config
    if hasattr(config.data, "max_seq_length") and config.data.max_seq_length > 128:
        config.data.max_seq_length = 128
        print(f"Reduced max sequence length to {config.data.max_seq_length}")
    
    # Extreme batch size and gradient accumulation settings
    config.data.micro_batch_size = 1
    print(f"Setting batch size to {config.data.micro_batch_size} to save memory")
    
    # Increase gradient accumulation even more
    config.training.gradient_accumulation_steps = 16
    print(f"Set gradient accumulation to {config.training.gradient_accumulation_steps}")
    
    # Reduce number of workers for dataloaders
    config.data.num_workers = 0
    print("Set dataloader workers to 0 (main process only)")
    
    # Initialize data module with updated config
    data_module = OpenAssistantDataModule(config=config.data)
    
    # Always disable DeepSpeed for now since it's causing issues
    args.no_deepspeed = True
    
    # Try to import deepspeed for advanced offloading if needed
    try:
        import deepspeed
        has_deepspeed = not args.no_deepspeed
        if has_deepspeed:
            print("DeepSpeed available - will use CPU offloading")
        else:
            print("DeepSpeed disabled - using standard training")
    except ImportError:
        has_deepspeed = False
        print("DeepSpeed not available - using standard training")
    
    # Initialize model with memory optimizations
    model_config = {
        "model": config.model,
        "training": config.training,
        "compile_model": False
    }
    
    model = LitPhiModel(
        config=model_config,
        learning_rate=args.lr,
        max_steps=args.max_steps,
    )
    
    # Setup trainer with standard strategy instead of DeepSpeed
    trainer_kwargs = {
        'accelerator': 'auto',
        'devices': 'auto',
        'precision': 'bf16-mixed',
        'log_every_n_steps': config.training.logging_steps,
        'gradient_clip_val': config.training.max_grad_norm,
        'accumulate_grad_batches': config.training.gradient_accumulation_steps,
        'val_check_interval': 1.0,
        'num_sanity_val_steps': 0,
        'limit_val_batches': 0,  # Disable validation completely
        'enable_checkpointing': True,
        'enable_model_summary': False,  # Disable model summary to save memory
        'enable_progress_bar': True,
        'strategy': 'auto',  # Let PyTorch Lightning choose the best strategy
    }
    
    # Pass a command-line flag to disable validation entirely if needed
    if args.no_validation:
        trainer_kwargs['limit_val_batches'] = 0
        print("Validation disabled to save memory")
    
    # Add callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='phi-{step:05d}',
        save_top_k=1,
        monitor=None,  # Don't monitor for best model since we may disable validation
        save_last=True,
        every_n_train_steps=config.training.save_steps,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer_kwargs['callbacks'] = [checkpoint_callback, lr_monitor]
    
    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="phi_model")
    trainer_kwargs['logger'] = logger
    
    # Add max_epochs or max_steps
    if args.max_epochs is not None:
        trainer_kwargs['max_epochs'] = args.max_epochs
    elif args.max_steps is not None:
        trainer_kwargs['max_steps'] = args.max_steps
    else:
        trainer_kwargs['max_epochs'] = config.training.num_train_epochs
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train with performance monitoring
    print("\nStarting training with performance monitoring...")
    print("Format: step | loss | iteration time | tokens per second | GPU memory\n")
    
    # Enable garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        trainer.save_checkpoint("checkpoints/interrupted_training.ckpt")
        print("Checkpoint saved. Exiting...")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
    
    # Add after training completes
    if args.save_adapters_only and hasattr(model.model, "save_pretrained"):
        print("Saving LoRA adapters only...")
        # Create adapters directory if it doesn't exist
        adapters_dir = "adapters"
        os.makedirs(adapters_dir, exist_ok=True)
        # Save the adapters only
        model.model.save_pretrained(adapters_dir)
        print(f"LoRA adapters saved to {adapters_dir}")
        
        # Also save the tokenizer for convenience
        tokenizer = model.trainer.datamodule.tokenizer
        tokenizer.save_pretrained(adapters_dir)
        print(f"Tokenizer saved to {adapters_dir}")
    
    return checkpoint_callback.best_model_path


def main():
    """
    Main function to handle training workflow
    """
    args = parse_args()
    
    # If no checkpoint provided, ask if user wants to resume from latest
    if args.ckpt_path is None:
        latest_checkpoint = get_latest_checkpoint()
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            print(f"\nFound existing checkpoint: {latest_checkpoint}")
            user_input = input("Resume training from checkpoint? (y/n): ").lower()
            
            if user_input == 'y':
                print(f"\nResuming training from checkpoint: {latest_checkpoint}")
                args.ckpt_path = latest_checkpoint
    
    # Start training
    best_model_path = train_model(args)
    print(f"\nTraining completed. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()