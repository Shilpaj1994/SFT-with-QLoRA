#!/usr/bin/env python3
"""
Configuration classes for the Phi model training.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """
    Configuration for data processing.
    """
    # Dataset parameters
    tokenizer_name: str = "microsoft/phi-2"
    batch_size: int = 8
    num_workers: int = 4
    shuffle_buffer_size: int = 10000
    max_length: int = 2048
    streaming: bool = True
    validation_split: float = 0.1
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """
    Configuration for the Phi model architecture.
    """
    # Base model to load
    model_name_or_path: str = "microsoft/phi-2"
    
    # Model architecture parameters (from model output)
    vocab_size: int = 51200
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 32  # Common ratio is hidden_size/80, but this is an assumption
    intermediate_size: int = 10240
    hidden_act: str = "gelu_new"
    layer_norm_eps: float = 1e-5
    
    # Dropout settings
    resid_dropout: float = 0.1
    embed_dropout: float = 0.0
    attention_dropout: float = 0.0

    # Positional embeddings
    rotary_emb: bool = True
    
    # Whether to tie weights between embedding and output layer
    tie_word_embeddings: bool = True


@dataclass
class QLoRAConfig:
    """
    Configuration for QLoRA (Quantized Low Rank Adaptation).
    """
    use_qlora: bool = False
    bits: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "dense"])


@dataclass
class SchedulerConfig:
    """
    Configuration for learning rate scheduler.
    """
    max_lr: float = 5e-5
    pct_start: float = 0.05
    div_factor: float = 25.0
    final_div_factor: float = 10.0
    three_phase: bool = False
    anneal_strategy: str = "cos"


@dataclass
class TrainingConfig:
    """
    Configuration for training the model.
    """
    # Basic training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Training length
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 4
    
    # Mixed precision training
    fp16: bool = True
    
    # Checkpoints
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 100
    report_to: str = "tensorboard"
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    
    # Generation during training
    generate_every: int = 500
    log_every: int = 50
    clear_cache_every: int = 0
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40


@dataclass
class Config:
    """
    Main configuration class that includes all sub-configurations.
    """
    # Use default_factory to avoid mutable defaults issue
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    
    # System settings
    seed: int = 42
    output_dir: str = "./outputs"
    compile_model: bool = True 