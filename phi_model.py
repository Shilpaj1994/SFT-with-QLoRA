#!/usr/bin/env python3
"""
This script is used to create a Phi-2 model.
"""
# Third Party Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType


class PhiModel:
    """
    Class to create a Phi-2 model.
    """
    def __init__(self, model_name="microsoft/phi-2", qlora_config=None):
        self.model_name = model_name
        self.qlora_config = qlora_config
        
        # Configure LoRA
        if qlora_config:
            self.lora_config = LoraConfig(
                r=qlora_config.lora_r,
                lora_alpha=qlora_config.lora_alpha,
                lora_dropout=qlora_config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=qlora_config.target_modules
            )
        else:
            # Default LoRA config (for backward compatibility)
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "dense"]
            )
    
    def load_tokenizer(self):
        """
        Load the tokenizer for the Phi-2 model.
        """
        return AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    
    def load_model(self, use_lora=True):
        """
        Load the Phi-2 base model(foundation model).
        """
        model_kwargs = {
            "trust_remote_code": True
        }
        
        # Set up quantization for QLoRA if enabled
        if self.qlora_config and self.qlora_config.use_qlora:
            print(f"Using QLoRA with {self.qlora_config.bnb_4bit_quant_type} quantization")
            compute_dtype = torch.float16
            if self.qlora_config.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.qlora_config.load_in_4bit,
                bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            # Use standard bf16 if QLoRA is not enabled
            model_kwargs["torch_dtype"] = torch.bfloat16
            
        # Load the model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if use_lora:
            # Apply LoRA adapters
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def save_model(self, model, save_path):
        """
        Save the Phi-2 model.
        """
        model.save_pretrained(save_path)

