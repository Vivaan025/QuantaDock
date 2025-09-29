#!/usr/bin/env python3
"""
LLM Optimization Script
Optimizes Large Language Models using various techniques:
- Quantization (INT8, INT4, FP16)
- LoRA fine-tuning
- Model pruning
- Memory optimization
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import time
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMOptimizer:
    def __init__(self, model_name: str, output_dir: str = "/workspace/results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, quantization: str = "none"):
        """Load model with specified quantization"""
        logger.info(f"Loading model: {self.model_name} with {quantization} quantization")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif quantization == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
            )
    
    def setup_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        """Setup LoRA configuration"""
        logger.info(f"Setting up LoRA with r={r}, alpha={alpha}, dropout={dropout}")
        
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # Prepare model for k-bit training if quantized
        if hasattr(self.model, 'is_loaded_in_8bit') or hasattr(self.model, 'is_loaded_in_4bit'):
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def benchmark_inference(self, prompt: str = "The future of AI is", max_length: int = 100, num_runs: int = 10):
        """Benchmark inference speed and memory usage"""
        logger.info("Benchmarking inference performance...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Benchmark
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_run = total_time / num_runs
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Tokens per second (approximate)
        total_tokens = (max_length - inputs.input_ids.shape[1]) * num_runs
        tokens_per_second = total_tokens / total_time
        
        benchmark_results = {
            "model_name": self.model_name,
            "total_time_seconds": total_time,
            "avg_time_per_run": avg_time_per_run,
            "tokens_per_second": tokens_per_second,
            "peak_memory_gb": peak_memory,
            "num_runs": num_runs,
            "max_length": max_length,
        }
        
        logger.info(f"Benchmark Results: {json.dumps(benchmark_results, indent=2)}")
        return benchmark_results
    
    def optimize_memory(self):
        """Apply memory optimization techniques"""
        logger.info("Applying memory optimizations...")
        
        # Gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Enable attention slicing for diffusion models (if applicable)
        if hasattr(self.model, 'enable_attention_slicing'):
            self.model.enable_attention_slicing(1)
        
        # Enable CPU offloading for large models
        if hasattr(self.model, 'enable_cpu_offload'):
            self.model.enable_cpu_offload()
    
    def export_optimized_model(self, output_path: str, format: str = "pytorch"):
        """Export optimized model in specified format"""
        logger.info(f"Exporting optimized model to {output_path} in {format} format")
        
        if format == "pytorch":
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
        elif format == "onnx":
            # ONNX export (simplified)
            try:
                import torch.onnx
                dummy_input = torch.randint(0, 1000, (1, 10)).to(self.device)
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    f"{output_path}/model.onnx",
                    input_names=['input_ids'],
                    output_names=['logits'],
                    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                'logits': {0: 'batch_size', 1: 'sequence'}}
                )
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")
        
    def run_optimization_pipeline(self, quantization: str = "int8", use_lora: bool = True):
        """Run complete optimization pipeline"""
        logger.info("Starting LLM optimization pipeline...")
        
        results = {}
        
        # Load model with quantization
        self.load_model(quantization)
        results['quantization'] = quantization
        
        # Setup LoRA if requested
        if use_lora:
            self.setup_lora()
        
        # Apply memory optimizations
        self.optimize_memory()
        
        # Benchmark performance
        benchmark_results = self.benchmark_inference()
        results['benchmark'] = benchmark_results
        
        # Export model
        export_path = self.output_dir / f"optimized_{self.model_name.replace('/', '_')}_{quantization}"
        self.export_optimized_model(str(export_path))
        results['export_path'] = str(export_path)
        
        # Save results
        results_file = self.output_dir / f"llm_optimization_results_{quantization}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization complete! Results saved to {results_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Optimize LLM models")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Model name or path")
    parser.add_argument("--quantization", choices=["none", "fp16", "int8", "int4"], default="int8")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--output-dir", default="/workspace/results", help="Output directory")
    
    args = parser.parse_args()
    
    optimizer = LLMOptimizer(args.model, args.output_dir)
    results = optimizer.run_optimization_pipeline(args.quantization, args.use_lora)
    
    print("LLM Optimization Complete!")
    print(f"Results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()
