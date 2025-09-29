#!/usr/bin/env python3
"""
Stable Diffusion Optimization Script
Optimizes Stable Diffusion models using various techniques:
- Mixed precision training
- Memory efficient attention
- Model quantization
- Pipeline optimization
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
import json
import time
import argparse
from pathlib import Path
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableDiffusionOptimizer:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", output_dir: str = "/workspace/results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_pipeline(self, precision: str = "fp16", enable_optimizations: bool = True):
        """Load Stable Diffusion pipeline with optimizations"""
        logger.info(f"Loading Stable Diffusion pipeline: {self.model_name}")
        
        torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        
        if enable_optimizations:
            self.apply_optimizations()
    
    def apply_optimizations(self):
        """Apply various optimizations to the pipeline"""
        logger.info("Applying optimizations...")
        
        # Memory efficient attention
        try:
            self.pipeline.enable_attention_slicing()
            logger.info("✓ Attention slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        # Memory efficient VAE
        try:
            self.pipeline.enable_vae_slicing()
            logger.info("✓ VAE slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable VAE slicing: {e}")
        
        # CPU offloading for memory constrained systems
        try:
            self.pipeline.enable_model_cpu_offload()
            logger.info("✓ CPU offloading enabled")
        except Exception as e:
            logger.warning(f"Could not enable CPU offloading: {e}")
        
        # Use efficient scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        logger.info("✓ DPM Solver scheduler enabled")
        
        # Enable memory efficient cross attention
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("✓ xFormers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"Could not enable xFormers: {e}")
    
    def benchmark_generation(self, prompts: list = None, num_inference_steps: int = 20, 
                           num_images: int = 4, height: int = 512, width: int = 512):
        """Benchmark image generation performance"""
        logger.info("Benchmarking Stable Diffusion generation...")
        
        if prompts is None:
            prompts = [
                "A beautiful landscape with mountains and lakes",
                "A futuristic cityscape at sunset",
                "Portrait of a robot in cyberpunk style",
                "Abstract art with vibrant colors"
            ]
        
        # Warm up
        logger.info("Warming up...")
        with torch.no_grad():
            self.pipeline(
                "warm up prompt",
                num_inference_steps=5,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
        
        # Benchmark
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        results = []
        total_start_time = time.time()
        
        for i, prompt in enumerate(prompts[:num_images]):
            logger.info(f"Generating image {i+1}/{len(prompts[:num_images])}: {prompt[:50]}...")
            
            start_time = time.time()
            
            with torch.no_grad():
                image = self.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=self.device).manual_seed(42 + i)
                ).images[0]
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Save image
            image_path = self.output_dir / f"generated_image_{i+1}.png"
            image.save(image_path)
            
            results.append({
                "prompt": prompt,
                "generation_time": generation_time,
                "image_path": str(image_path),
                "steps": num_inference_steps,
                "resolution": f"{width}x{height}"
            })
        
        total_time = time.time() - total_start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        benchmark_results = {
            "model_name": self.model_name,
            "total_time": total_time,
            "avg_time_per_image": total_time / len(results),
            "peak_memory_gb": peak_memory,
            "num_inference_steps": num_inference_steps,
            "resolution": f"{width}x{height}",
            "images_generated": len(results),
            "individual_results": results
        }
        
        logger.info(f"Benchmark Results: {json.dumps({k: v for k, v in benchmark_results.items() if k != 'individual_results'}, indent=2)}")
        return benchmark_results
    
    def optimize_for_batch_inference(self, batch_size: int = 4):
        """Optimize pipeline for batch inference"""
        logger.info(f"Optimizing for batch inference with batch_size={batch_size}")
        
        # Test batch generation
        prompts = ["Test prompt"] * batch_size
        
        start_time = time.time()
        with torch.no_grad():
            images = self.pipeline(
                prompts,
                num_inference_steps=20,
                height=512,
                width=512,
            ).images
        end_time = time.time()
        
        batch_time = end_time - start_time
        time_per_image = batch_time / batch_size
        
        logger.info(f"Batch generation: {batch_time:.2f}s total, {time_per_image:.2f}s per image")
        
        return {
            "batch_size": batch_size,
            "total_time": batch_time,
            "time_per_image": time_per_image,
            "throughput_images_per_second": batch_size / batch_time
        }
    
    def apply_quantization(self, quantization_type: str = "dynamic"):
        """Apply post-training quantization"""
        logger.info(f"Applying {quantization_type} quantization...")
        
        if quantization_type == "dynamic":
            # Dynamic quantization (CPU only)
            if self.device.type == "cpu":
                self.pipeline.unet = torch.quantization.quantize_dynamic(
                    self.pipeline.unet, {nn.Linear}, dtype=torch.qint8
                )
                logger.info("✓ Dynamic quantization applied to UNet")
            else:
                logger.warning("Dynamic quantization only supported on CPU")
        
        elif quantization_type == "static":
            # Static quantization would require calibration data
            logger.warning("Static quantization not implemented in this demo")
    
    def export_optimized_model(self, export_format: str = "pytorch"):
        """Export optimized model"""
        logger.info(f"Exporting model in {export_format} format...")
        
        export_path = self.output_dir / f"optimized_sd_{export_format}"
        export_path.mkdir(exist_ok=True)
        
        if export_format == "pytorch":
            self.pipeline.save_pretrained(str(export_path))
        elif export_format == "onnx":
            # ONNX export is complex for Stable Diffusion
            # This would require exporting each component separately
            logger.warning("ONNX export for Stable Diffusion is complex and not implemented in this demo")
        
        return str(export_path)
    
    def run_optimization_pipeline(self, precision: str = "fp16", quantization: str = None):
        """Run complete optimization pipeline"""
        logger.info("Starting Stable Diffusion optimization pipeline...")
        
        results = {}
        
        # Load pipeline with optimizations
        self.load_pipeline(precision, enable_optimizations=True)
        results['precision'] = precision
        
        # Apply quantization if requested
        if quantization:
            self.apply_quantization(quantization)
            results['quantization'] = quantization
        
        # Benchmark single image generation
        benchmark_results = self.benchmark_generation()
        results['single_image_benchmark'] = benchmark_results
        
        # Benchmark batch generation
        batch_results = self.optimize_for_batch_inference(batch_size=2)
        results['batch_benchmark'] = batch_results
        
        # Export optimized model
        export_path = self.export_optimized_model()
        results['export_path'] = export_path
        
        # Save results
        results_file = self.output_dir / f"sd_optimization_results_{precision}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization complete! Results saved to {results_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Optimize Stable Diffusion models")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Model name or path")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--quantization", choices=["dynamic", "static"], help="Quantization type")
    parser.add_argument("--output-dir", default="/workspace/results", help="Output directory")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for testing")
    
    args = parser.parse_args()
    
    optimizer = StableDiffusionOptimizer(args.model, args.output_dir)
    results = optimizer.run_optimization_pipeline(args.precision, args.quantization)
    
    print("Stable Diffusion Optimization Complete!")
    print(f"Results: {json.dumps({k: v for k, v in results.items() if k not in ['single_image_benchmark', 'batch_benchmark']}, indent=2)}")

if __name__ == "__main__":
    main()
