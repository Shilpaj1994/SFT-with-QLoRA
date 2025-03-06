#!/usr/bin/env python3
"""
Inference script for the Phi-2 model.
Supports both regular HuggingFace models and PyTorch Lightning models.
"""
# Standard Library Imports
import os
import argparse
import json
import time

# Third Party Imports
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Try to import Lightning (but don't fail if not available)
try:
    import pytorch_lightning as pl
    from training import LitPhiModel
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with Phi-2 model")
    parser.add_argument("--model_path", type=str, default="./outputs", help="Path to model or checkpoint")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="Base model name")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA adapters")
    parser.add_argument("--use_qlora", action="store_true", help="Whether to use QLoRA (4-bit quantization)")
    parser.add_argument("--use_lightning", action="store_true", help="Whether to load a Lightning checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--example_prompts", action="store_true", help="Use example prompts instead of dataset")
    return parser.parse_args()


def load_huggingface_model(args):
    """
    Load a HuggingFace model with improved error handling.
    """
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        print("Trying to continue with model loading anyway...")
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist!")
    
    # Check if the path is a .ckpt file and the user specified use_lora
    if args.model_path.endswith('.ckpt') and (args.use_lora or args.use_qlora):
        raise ValueError(
            f"It looks like you're trying to load a PyTorch Lightning checkpoint file ({args.model_path}) with the --use_lora flag.\n"
            f"Lightning checkpoint files should be loaded with --use_lightning instead.\n"
            f"Please try: python inference.py --model_path {args.model_path} --base_model {args.base_model} --use_lightning"
        )
    
    # Load model
    if args.use_lora or args.use_qlora:
        print(f"Loading base model {args.base_model}...")
        
        # Configure model loading parameters
        model_kwargs = {"trust_remote_code": True}
        
        # Set up quantization for QLoRA if enabled
        if args.use_qlora:
            print("Using 4-bit quantization (QLoRA)")
            compute_dtype = torch.float16
            if torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        try:
            # Load the base model with appropriate configuration
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model, 
                **model_kwargs
            )
            
            # Check if LoRA adapter path exists and has the expected files
            if not os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
                print(f"Warning: Could not find adapter_config.json in {args.model_path}")
                print("Available files in the directory:")
                for file in os.listdir(args.model_path):
                    print(f"  - {file}")
                raise ValueError(f"The specified path '{args.model_path}' does not appear to contain a valid LoRA adapter")
            
            print(f"Loading {'QLoRA' if args.use_qlora else 'LoRA'} adapters from {args.model_path}...")
            model = PeftModel.from_pretrained(base_model, args.model_path)
            
            # Special handling for QLoRA - move norm layers to float32 for stability
            if args.use_qlora:
                print("Converting normalization layers to float32 for stability")
                for name, module in model.named_modules():
                    if "norm" in name:
                        module = module.to(torch.float32)
        except Exception as e:
            raise ValueError(f"Failed to load LoRA adapters: {str(e)}")
    else:
        print(f"Loading model from {args.model_path}...")
        try:
            # Check if the model path contains the expected files
            expected_files = ["config.json", "pytorch_model.bin"] 
            alternative_files = ["config.json", "model.safetensors"]
            
            has_expected_files = all(os.path.exists(os.path.join(args.model_path, f)) for f in expected_files)
            has_alternative_files = all(os.path.exists(os.path.join(args.model_path, f)) for f in alternative_files)
            
            if not (has_expected_files or has_alternative_files):
                print(f"Warning: The model path '{args.model_path}' may not contain a valid HuggingFace model.")
                print("Expected files not found. Available files in the directory:")
                for file in os.listdir(args.model_path):
                    print(f"  - {file}")
                print("\nTrying to load anyway...")
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nFallback: trying to load from base model path and merge with adapters...")
            
            try:
                # Try loading the base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                
                # Check if there are any adapter files in the model path
                if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
                    print(f"Found adapter_config.json, attempting to load as a PEFT/LoRA model...")
                    model = PeftModel.from_pretrained(base_model, args.model_path)
                else:
                    # Just use the base model
                    print("Using base model as fallback")
                    model = base_model
            except Exception as nested_e:
                raise ValueError(f"Failed to load model from both paths. Original error: {str(e)}, Fallback error: {str(nested_e)}")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully and moved to {device}!")
    
    return model, tokenizer


def load_lightning_model(args):
    """
    Load a PyTorch Lightning model from checkpoint.
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError("PyTorch Lightning is not available. Please install it first.")
    
    print(f"Loading Lightning model from checkpoint {args.model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model from checkpoint
    model = LitPhiModel.load_from_checkpoint(
        args.model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        strict=False
    )
    model.eval()
    
    # Move model to right device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Lightning model loaded successfully and moved to {device}!")
    
    return model, tokenizer


def get_example_prompts():
    """
    Return a list of example prompts.
    """
    return [
        "Write a short story about a robot learning to feel emotions.",
        "Explain how nuclear fusion works in simple terms.",
        "What are the ethical implications of artificial intelligence?",
        "Create a recipe for a delicious vegetarian pasta dish.",
        "Write a poem about the changing seasons."
    ]


def generate_text_huggingface(model, tokenizer, prompt, args):
    """
    Generate text using the HuggingFace model with token-by-token generation.
    """
    device = next(model.parameters()).device
    
    # Get the model's default dtype for proper casting
    model_dtype = next(iter(model.parameters())).dtype
    
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Track generation time
    start_time = time.time()
    
    # Generate tokens one by one for more control
    try:
        for _ in tqdm(range(args.max_new_tokens), desc="Generating"):
            with torch.no_grad():
                # Ensure inputs have the right dtype to match model params
                if hasattr(model, 'dtype'):
                    input_dtype = model.dtype
                else:
                    input_dtype = model_dtype
                
                # Process the outputs
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / args.temperature
                
                # Apply top-k filtering
                if args.top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, args.top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                # Apply top-p (nucleus) filtering
                if args.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > args.top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back the indices to the original logits tensor
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we generate an EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                # Append to input_ids and attention_mask
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    
    except RuntimeError as e:
        if "expected scalar type" in str(e):
            # Alternative generation method using built-in generate method
            print("Falling back to built-in generate method...")
            with torch.no_grad():
                attention_mask = torch.ones_like(input_ids)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=True,
                )
                input_ids = generated_ids
        else:
            raise e
    
    # Calculate generation time and speed
    end_time = time.time()
    gen_time = end_time - start_time
    tokens_generated = input_ids.size(1) - tokenizer.encode(prompt, return_tensors='pt').size(1)
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    print(f"Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def generate_text_lightning(model, tokenizer, prompt, args):
    """
    Generate text using the Lightning model.
    """
    device = next(model.parameters()).device
    
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Track generation time
    start_time = time.time()
    
    # Generate tokens one by one
    for _ in tqdm(range(args.max_new_tokens), desc="Generating"):
        with torch.no_grad():
            # Forward pass through the model
            outputs = model(input_ids)
            
            # Get the logits for the last token
            logits = outputs.logits[:, -1, :] / args.temperature
            
            # Apply top-k sampling
            if args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > args.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Convert back to logits
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    # Calculate generation time and speed
    end_time = time.time()
    gen_time = end_time - start_time
    tokens_per_sec = args.max_new_tokens / gen_time if gen_time > 0 else 0
    print(f"Generated {input_ids.size(1) - tokenizer.encode(prompt, return_tensors='pt').size(1)} tokens in {gen_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def generate_text_simple(model, tokenizer, prompt, args):
    """
    Generate text using the model's built-in generate method with better dtype handling.
    """
    device = next(model.parameters()).device
    
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Track generation time
    start_time = time.time()
    
    # Use the built-in generate method with robust error handling
    with torch.no_grad():
        try:
            # First attempt with all parameters
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except RuntimeError as e:
            if "expected scalar type" in str(e):
                # Fix model dtypes before retrying
                print("Fixing model dtype issues...")
                for name, module in model.named_modules():
                    if any(x in name for x in ["lm_head", "embed_tokens", "norm"]):
                        # Critical components should be in same dtype
                        module.to(torch.float16 if torch.cuda.is_available() else torch.float32)
                
                # Force inputs to match primary dtype
                # But keep them as Long for token IDs
                attention_mask = attention_mask.to(torch.float16 if torch.cuda.is_available() else torch.float32)
                
                # Try a completely different approach - greedy decoding
                try:
                    print("Trying greedy decoding instead...")
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=min(100, args.max_new_tokens),  # Limit tokens for simplicity
                        do_sample=False,  # Use greedy decoding
                        use_cache=True,
                    )
                except Exception as e2:
                    print(f"Greedy decoding failed: {str(e2)}")
                    
                    # Super simple manual generation
                    print("Using manual token generation...")
                    current_ids = input_ids.clone()
                    generated_tokens = 0
                    max_tokens = min(50, args.max_new_tokens)  # Limit to 50 tokens
                    
                    for _ in range(max_tokens):
                        try:
                            # Convert the input_ids to long type explicitly
                            input_to_model = current_ids.to(torch.long)
                            # Forward pass with no gradient
                            with torch.no_grad():
                                # Get logits
                                outputs = model(input_ids=input_to_model)
                                # Get the next token - simple argmax
                                next_token_id = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                                # Convert to long to ensure right type
                                next_token_id = next_token_id.to(torch.long)
                                # Add the new token
                                current_ids = torch.cat([current_ids, next_token_id], dim=1)
                                generated_tokens += 1
                                
                                # Stop if we get an EOS token
                                if next_token_id.item() == tokenizer.eos_token_id:
                                    break
                        except Exception as e3:
                            print(f"Manual generation error: {str(e3)}")
                            break
                    
                    generated_ids = current_ids
                    print(f"Manually generated {generated_tokens} tokens")
            else:
                raise e
    
    # Calculate generation time and speed
    end_time = time.time()
    gen_time = end_time - start_time
    tokens_generated = generated_ids.size(1) - input_ids.size(1)
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    print(f"Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def interactive_mode(model, tokenizer, args):
    """
    Run the model in interactive mode.
    """
    print("\n" + "="*50)
    print("Interactive Mode - Enter your prompts (type 'exit' to quit)")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        user_input = input("\nEnter your prompt: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Format prompt for Phi-2
        prompt = f"Instruct: {user_input}\nOutput:"
        print("\nGenerating response...")
        
        # Generate response
        if args.use_lightning:
            response = generate_text_lightning(model, tokenizer, prompt, args)
        elif args.use_qlora:
            # Use the more robust method for QLoRA models
            response = generate_text_simple(model, tokenizer, prompt, args)
        else:
            response = generate_text_huggingface(model, tokenizer, prompt, args)
        
        # Extract just the output part
        output = response.split("Output:")[1].strip() if "Output:" in response else response
        
        # Print the response
        print("\n" + "="*50 + "\nResponse:")
        print(output)
        print("="*50)


def evaluate_on_dataset(model, tokenizer, args):
    """
    Evaluate the model on a dataset and save results.
    """
    if args.example_prompts:
        prompts = get_example_prompts()
        print(f"Using {len(prompts)} example prompts")
    else:
        # Load validation dataset
        dataset = load_dataset("OpenAssistant/oasst1", split="validation")
        
        # Filter for prompter messages to use as inputs
        prompter_msgs = [msg for msg in dataset if msg["role"] == "prompter"]
        
        # Limit to num_samples
        if args.num_samples < len(prompter_msgs):
            np.random.seed(42)
            indices = np.random.choice(len(prompter_msgs), args.num_samples, replace=False)
            prompter_msgs = [prompter_msgs[i] for i in indices]
        
        prompts = [msg["text"] for msg in prompter_msgs]
        message_ids = [msg["message_id"] for msg in prompter_msgs]
        print(f"Evaluating on {len(prompts)} samples from OpenAssistant dataset")
    
    results = []
    
    for i, prompt_text in enumerate(prompts):
        print(f"\nGenerating response for prompt {i+1}/{len(prompts)}")
        print(f"Prompt: {prompt_text[:100]}...")
        
        # Format prompt for Phi-2
        formatted_prompt = f"Instruct: {prompt_text}\nOutput:"
        
        # Generate response
        try:
            if args.use_lightning:
                response = generate_text_lightning(model, tokenizer, formatted_prompt, args)
            elif args.use_qlora:
                # Use the more robust method for QLoRA models
                response = generate_text_simple(model, tokenizer, formatted_prompt, args)
            else:
                response = generate_text_huggingface(model, tokenizer, formatted_prompt, args)
            
            # Extract just the output part
            output = response.split("Output:")[1].strip() if "Output:" in response else response
            
            # Create result entry
            result = {
                "prompt": prompt_text,
                "generated_response": output,
            }
            
            # Add message_id if using dataset
            if not args.example_prompts:
                result["message_id"] = message_ids[i]
                
            results.append(result)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            # Still add an entry with the error message
            results.append({
                "prompt": prompt_text,
                "error": str(e),
                "generated_response": ""
            })
    
    # Save results
    output_file = "example_outputs.json" if args.example_prompts else "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed. Results saved to {output_file}")
    
    # Print a few examples
    print("\nExample outputs:")
    for i in range(min(3, len(results))):
        if "error" in results[i]:
            print(f"\nPrompt: {results[i]['prompt'][:100]}...")
            print(f"Error: {results[i]['error']}")
        else:
            print(f"\nPrompt: {results[i]['prompt'][:100]}...")
            print(f"Response: {results[i]['generated_response'][:200]}...")


def examine_model_path(model_path):
    """
    Examine the model path to help diagnose loading issues.
    """
    print(f"\nExamining model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Path '{model_path}' does not exist!")
        return
    
    if os.path.isfile(model_path):
        print(f"Note: Path is a file, not a directory. If this is a checkpoint, make sure to use --use_lightning")
        return
    
    # List all files in the directory
    print("Files in the directory:")
    files = os.listdir(model_path)
    for file in files:
        file_path = os.path.join(model_path, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  - {file} ({file_size:.2f} MB)")
    
    # Check for key configuration files
    config_files = ["config.json", "adapter_config.json", "pytorch_model.bin", "model.safetensors"]
    found_files = [f for f in config_files if f in files]
    
    if "config.json" in files:
        try:
            with open(os.path.join(model_path, "config.json"), "r") as f:
                config = json.load(f)
            if "model_type" in config:
                print(f"\nModel type from config.json: {config['model_type']}")
            else:
                print("\nWarning: config.json does not contain 'model_type' key")
        except Exception as e:
            print(f"\nError reading config.json: {str(e)}")
    
    if "adapter_config.json" in files:
        try:
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            print(f"\nAdapter type: {adapter_config.get('peft_type', 'unknown')}")
            print(f"Base model path: {adapter_config.get('base_model_name_or_path', 'unknown')}")
        except Exception as e:
            print(f"\nError reading adapter_config.json: {str(e)}")
    
    print("\nDiagnosis:")
    if "config.json" in files and ("pytorch_model.bin" in files or "model.safetensors" in files):
        print("✓ Directory appears to contain a complete HuggingFace model")
    elif "adapter_config.json" in files:
        print("✓ Directory appears to contain a PEFT/LoRA adapter")
        print("  - Use with --use_lora flag")
        if "base_model_name_or_path" not in adapter_config:
            print("  - Make sure to specify the correct --base_model")
    elif any(f.endswith(".ckpt") for f in files):
        print("✓ Directory appears to contain PyTorch Lightning checkpoints")
        print("  - Use with --use_lightning flag")
    else:
        print("✗ Directory does not appear to contain a valid model in a recognized format")
        
    print("\nSuggested commands:")
    if "adapter_config.json" in files:
        print(f"  python inference.py --model_path {model_path} --base_model microsoft/phi-2 --use_lora")
    elif any(f.endswith(".ckpt") for f in files):
        ckpt_file = next(f for f in files if f.endswith(".ckpt"))
        print(f"  python inference.py --model_path {os.path.join(model_path, ckpt_file)} --base_model microsoft/phi-2 --use_lightning")
    else:
        print(f"  python inference.py --model_path {model_path} --base_model microsoft/phi-2")


def main():
    """
    Main function to run inference with the Phi-2 model.
    """
    args = parse_args()
    
    try:
        # Load model based on args
        if args.use_lightning:
            model, tokenizer = load_lightning_model(args)
        else:
            model, tokenizer = load_huggingface_model(args)
        
        # Run in interactive mode or evaluate on dataset
        if args.interactive:
            interactive_mode(model, tokenizer, args)
        else:
            evaluate_on_dataset(model, tokenizer, args)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        # Provide additional diagnostics
        print("\nModel loading failed. Examining model path to help diagnose the issue:")
        examine_model_path(args.model_path)
        
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        
        print("\nSuggested troubleshooting steps:")
        print("1. Verify that the model path contains a valid model")
        print("2. If using a LoRA adapter, make sure to use the --use_lora flag")
        print("3. If using a Lightning checkpoint, make sure to use the --use_lightning flag")
        print("4. Check that the base model name is correct")
        print("5. For LoRA adapters, make sure the adapter is compatible with the base model")
        
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()