#!/usr/bin/env python3
"""
Master Optimization Runner
Orchestrates optimization of all ML model types with comprehensive benchmarking
"""

import argparse
import json
import subprocess
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

# Import configurations
sys.path.append('/workspace/configs')
from optimization_config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterOptimizer:
    def __init__(self, output_dir: str = "/workspace/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run_system_info(self):
        """Get system specifications"""
        logger.info("🔍 Gathering system information...")
        
        try:
            result = subprocess.run([
                sys.executable, "/workspace/scripts/system_info.py"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ System information gathered successfully")
                
                # Load the saved specs
                specs_file = Path("/workspace/system_specs.json")
                if specs_file.exists():
                    with open(specs_file, 'r') as f:
                        self.results['system_specs'] = json.load(f)
                        
                return True
            else:
                logger.error(f"❌ System info failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ System info error: {e}")
            return False
    
    def optimize_llm(self, model: str = "microsoft/DialoGPT-small", quantization: str = "int8"):
        """Optimize LLM model"""
        logger.info(f"🤖 Optimizing LLM: {model} with {quantization} quantization...")
        
        try:
            cmd = [
                sys.executable, "/workspace/scripts/optimize_llm.py",
                "--model", model,
                "--quantization", quantization,
                "--use-lora",
                "--output-dir", str(self.output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info("✅ LLM optimization completed successfully")
                
                # Load results
                results_file = self.output_dir / f"llm_optimization_results_{quantization}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.results['llm_optimization'] = json.load(f)
                        
                return True
            else:
                logger.error(f"❌ LLM optimization failed: {result.stderr}")
                self.results['llm_optimization'] = {"error": result.stderr}
                return False
                
        except Exception as e:
            logger.error(f"❌ LLM optimization error: {e}")
            self.results['llm_optimization'] = {"error": str(e)}
            return False
    
    def optimize_stable_diffusion(self, model: str = "runwayml/stable-diffusion-v1-5", precision: str = "fp16"):
        """Optimize Stable Diffusion model"""
        logger.info(f"🎨 Optimizing Stable Diffusion: {model} with {precision} precision...")
        
        try:
            cmd = [
                sys.executable, "/workspace/scripts/optimize_stable_diffusion.py",
                "--model", model,
                "--precision", precision,
                "--output-dir", str(self.output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
            
            if result.returncode == 0:
                logger.info("✅ Stable Diffusion optimization completed successfully")
                
                # Load results
                results_file = self.output_dir / f"sd_optimization_results_{precision}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.results['stable_diffusion_optimization'] = json.load(f)
                        
                return True
            else:
                logger.error(f"❌ Stable Diffusion optimization failed: {result.stderr}")
                self.results['stable_diffusion_optimization'] = {"error": result.stderr}
                return False
                
        except Exception as e:
            logger.error(f"❌ Stable Diffusion optimization error: {e}")
            self.results['stable_diffusion_optimization'] = {"error": str(e)}
            return False
    
    def optimize_text_encoder(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                            model_type: str = "sentence_transformer"):
        """Optimize text encoder model"""
        logger.info(f"📝 Optimizing Text Encoder: {model} ({model_type})...")
        
        try:
            cmd = [
                sys.executable, "/workspace/scripts/optimize_text_encoder.py",
                "--model", model,
                "--model-type", model_type,
                "--precision", "fp32",
                "--output-dir", str(self.output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            
            if result.returncode == 0:
                logger.info("✅ Text encoder optimization completed successfully")
                
                # Load results
                results_file = self.output_dir / f"text_encoder_optimization_results_{model_type}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.results['text_encoder_optimization'] = json.load(f)
                        
                return True
            else:
                logger.error(f"❌ Text encoder optimization failed: {result.stderr}")
                self.results['text_encoder_optimization'] = {"error": result.stderr}
                return False
                
        except Exception as e:
            logger.error(f"❌ Text encoder optimization error: {e}")
            self.results['text_encoder_optimization'] = {"error": str(e)}
            return False
    
    def optimize_audio_model(self, model: str = "facebook/wav2vec2-base-960h", precision: str = "fp32"):
        """Optimize audio embedding model"""
        logger.info(f"🔊 Optimizing Audio Model: {model} with {precision} precision...")
        
        try:
            cmd = [
                sys.executable, "/workspace/scripts/optimize_sound_to_vec.py",
                "--model", model,
                "--precision", precision,
                "--enable-preprocessing",
                "--output-dir", str(self.output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info("✅ Audio model optimization completed successfully")
                
                # Load results
                results_file = self.output_dir / f"audio_optimization_results_{precision}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.results['audio_optimization'] = json.load(f)
                        
                return True
            else:
                logger.error(f"❌ Audio optimization failed: {result.stderr}")
                self.results['audio_optimization'] = {"error": result.stderr}
                return False
                
        except Exception as e:
            logger.error(f"❌ Audio optimization error: {e}")
            self.results['audio_optimization'] = {"error": str(e)}
            return False
    
    def run_parallel_optimizations(self, tasks: list):
        """Run optimizations in parallel where possible"""
        logger.info("🚀 Running optimizations in parallel...")
        
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 to avoid GPU memory conflicts
            future_to_task = {}
            
            for task_name, task_func, args in tasks:
                future = executor.submit(task_func, *args)
                future_to_task[future] = task_name
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = result
                    logger.info(f"✅ {task_name} completed: {'Success' if result else 'Failed'}")
                except Exception as e:
                    logger.error(f"❌ {task_name} failed with exception: {e}")
                    results[task_name] = False
        
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("📊 Generating performance report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimization_summary": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        # System specs summary
        if 'system_specs' in self.results:
            sys_specs = self.results['system_specs']
            report['system_summary'] = {
                "cpu_cores": sys_specs.get('cpu', {}).get('cores_logical', 'Unknown'),
                "memory_gb": sys_specs.get('memory', {}).get('total_gb', 'Unknown'),
                "gpu_count": len(sys_specs.get('gpu', [])),
                "gpu_memory_gb": sum(gpu.get('total_memory_gb', 0) for gpu in sys_specs.get('gpu', [])),
            }
        
        # Optimization results summary
        for opt_type, opt_results in self.results.items():
            if 'optimization' in opt_type and isinstance(opt_results, dict) and 'error' not in opt_results:
                if 'benchmark' in opt_results:
                    benchmark = opt_results['benchmark']
                    if isinstance(benchmark, dict):
                        report['performance_metrics'][opt_type] = {
                            "processing_speed": benchmark.get('tokens_per_second', benchmark.get('audio_clips_per_second', 'N/A')),
                            "memory_usage_gb": benchmark.get('peak_memory_gb', 'N/A'),
                            "avg_processing_time": benchmark.get('avg_time_per_run', 'N/A')
                        }
        
        # Generate recommendations
        recommendations = []
        
        if 'system_specs' in self.results:
            gpu_info = self.results['system_specs'].get('gpu', [])
            if gpu_info:
                total_gpu_memory = sum(gpu.get('total_memory_gb', 0) for gpu in gpu_info)
                if total_gpu_memory < 8:
                    recommendations.append("Consider using INT8 quantization for large models due to limited GPU memory")
                elif total_gpu_memory >= 24:
                    recommendations.append("Your system has sufficient GPU memory for FP16 precision on large models")
        
        # Performance-based recommendations
        for opt_type, metrics in report['performance_metrics'].items():
            if isinstance(metrics.get('processing_speed'), (int, float)):
                if metrics['processing_speed'] < 10:
                    recommendations.append(f"Consider optimizing {opt_type} further - processing speed is below optimal")
                
        report['recommendations'] = recommendations
        
        # Save comprehensive report
        report_file = self.output_dir / "comprehensive_optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save all results
        all_results_file = self.output_dir / "all_optimization_results.json"
        with open(all_results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📊 Performance report saved to: {report_file}")
        logger.info(f"📁 All results saved to: {all_results_file}")
        
        return report
    
    def run_complete_optimization_suite(self, models_config: dict = None):
        """Run complete optimization suite for all model types"""
        logger.info("🚀 Starting complete ML model optimization suite...")
        
        if models_config is None:
            models_config = {
                "llm": {"model": "microsoft/DialoGPT-small", "quantization": "int8"},
                "stable_diffusion": {"model": "runwayml/stable-diffusion-v1-5", "precision": "fp16"},
                "text_encoder": {"model": "sentence-transformers/all-MiniLM-L6-v2", "model_type": "sentence_transformer"},
                "audio": {"model": "facebook/wav2vec2-base-960h", "precision": "fp32"}
            }
        
        start_time = time.time()
        
        # Step 1: System information
        self.run_system_info()
        
        # Step 2: Sequential optimizations (to avoid GPU memory conflicts)
        optimizations = [
            ("text_encoder", self.optimize_text_encoder, 
             (models_config["text_encoder"]["model"], models_config["text_encoder"]["model_type"])),
            ("llm", self.optimize_llm, 
             (models_config["llm"]["model"], models_config["llm"]["quantization"])),
            ("audio", self.optimize_audio_model, 
             (models_config["audio"]["model"], models_config["audio"]["precision"])),
            ("stable_diffusion", self.optimize_stable_diffusion, 
             (models_config["stable_diffusion"]["model"], models_config["stable_diffusion"]["precision"])),
        ]
        
        # Run optimizations sequentially to avoid memory issues
        for opt_name, opt_func, opt_args in optimizations:
            logger.info(f"🔄 Running {opt_name} optimization...")
            success = opt_func(*opt_args)
            
            if success:
                logger.info(f"✅ {opt_name} optimization completed successfully")
            else:
                logger.warning(f"⚠️ {opt_name} optimization failed, continuing with others...")
            
            # Clear GPU cache between optimizations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Step 3: Generate comprehensive report
        report = self.generate_performance_report()
        
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info("🎉 OPTIMIZATION SUITE COMPLETED!")
        logger.info(f"⏱️ Total time: {total_time/60:.2f} minutes")
        logger.info(f"📊 Report saved to: {self.output_dir}/comprehensive_optimization_report.json")
        logger.info("=" * 80)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Master ML Model Optimizer")
    parser.add_argument("--output-dir", default="/workspace/results", help="Output directory")
    parser.add_argument("--llm-model", default="microsoft/DialoGPT-small", help="LLM model to optimize")
    parser.add_argument("--sd-model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model")
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Text encoder model")
    parser.add_argument("--audio-model", default="facebook/wav2vec2-base-960h", help="Audio model")
    parser.add_argument("--run-single", choices=["system", "llm", "sd", "text", "audio"], help="Run single optimization")
    
    args = parser.parse_args()
    
    optimizer = MasterOptimizer(args.output_dir)
    
    if args.run_single:
        if args.run_single == "system":
            optimizer.run_system_info()
        elif args.run_single == "llm":
            optimizer.optimize_llm(args.llm_model)
        elif args.run_single == "sd":
            optimizer.optimize_stable_diffusion(args.sd_model)
        elif args.run_single == "text":
            optimizer.optimize_text_encoder(args.text_model)
        elif args.run_single == "audio":
            optimizer.optimize_audio_model(args.audio_model)
    else:
        # Run complete suite
        models_config = {
            "llm": {"model": args.llm_model, "quantization": "int8"},
            "stable_diffusion": {"model": args.sd_model, "precision": "fp16"},
            "text_encoder": {"model": args.text_model, "model_type": "sentence_transformer"},
            "audio": {"model": args.audio_model, "precision": "fp32"}
        }
        
        optimizer.run_complete_optimization_suite(models_config)

if __name__ == "__main__":
    main()
