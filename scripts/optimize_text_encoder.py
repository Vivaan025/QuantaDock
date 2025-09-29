#!/usr/bin/env python3
"""
Text Encoder/Tokenizer Optimization Script
Optimizes text encoders and tokenizers for various NLP tasks:
- BERT, RoBERTa, and other transformer encoders
- Sentence transformers
- Custom tokenizers
- Embedding optimization
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, 
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    TrainingArguments, Trainer
)
from sentence_transformers import SentenceTransformer
import json
import time
import argparse
from pathlib import Path
import logging
import numpy as np
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEncoderOptimizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 output_dir: str = "/workspace/results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.sentence_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_type: str = "sentence_transformer", precision: str = "fp32"):
        """Load text encoder model"""
        logger.info(f"Loading {model_type} model: {self.model_name}")
        
        torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        
        if model_type == "sentence_transformer":
            self.sentence_model = SentenceTransformer(self.model_name, device=self.device)
            if precision == "fp16":
                self.sentence_model = self.sentence_model.half()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype
            ).to(self.device)
            
    def optimize_tokenizer(self, vocab_size_limit: int = None):
        """Optimize tokenizer for faster processing"""
        logger.info("Optimizing tokenizer...")
        
        if self.tokenizer is None:
            logger.warning("No tokenizer loaded")
            return
        
        # Get tokenizer stats
        vocab_size = len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else self.tokenizer.vocab_size
        
        tokenizer_stats = {
            "original_vocab_size": vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
        }
        
        # Optimize settings for faster processing
        self.tokenizer.padding_side = "left"  # Better for batch processing
        
        logger.info(f"Tokenizer stats: {tokenizer_stats}")
        return tokenizer_stats
    
    def benchmark_encoding(self, texts: List[str] = None, batch_sizes: List[int] = [1, 8, 16, 32]):
        """Benchmark text encoding performance"""
        logger.info("Benchmarking text encoding performance...")
        
        if texts is None:
            texts = [
                "This is a sample sentence for benchmarking.",
                "Natural language processing is fascinating.",
                "Machine learning models require optimization.",
                "Text encoding is a crucial step in NLP pipelines.",
                "Transformers have revolutionized the field of AI.",
                "Efficient processing is key for production systems.",
                "Benchmarking helps identify performance bottlenecks.", 
                "Optimization techniques can significantly improve speed."
            ] * 125  # Create 1000 sentences
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Prepare batches
            batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            
            # Warm up
            if self.sentence_model:
                self.sentence_model.encode(texts[:3])
            else:
                inputs = self.tokenizer(texts[:3], padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            if self.sentence_model:
                # Sentence transformer encoding
                for batch in batches[:10]:  # Test first 10 batches
                    embeddings = self.sentence_model.encode(batch, convert_to_tensor=True)
            else:
                # Standard transformer encoding
                for batch in batches[:10]:  # Test first 10 batches
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            
            end_time = time.time()
            
            total_time = end_time - start_time
            total_texts = min(len(texts), batch_size * 10)
            texts_per_second = total_texts / total_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            results[f"batch_size_{batch_size}"] = {
                "total_time": total_time,
                "texts_per_second": texts_per_second,
                "peak_memory_gb": peak_memory,
                "total_texts_processed": total_texts
            }
            
            logger.info(f"Batch size {batch_size}: {texts_per_second:.2f} texts/sec, {peak_memory:.2f} GB memory")
        
        return results
    
    def benchmark_similarity_search(self, queries: List[str] = None, corpus: List[str] = None):
        """Benchmark semantic similarity search"""
        if self.sentence_model is None:
            logger.warning("Sentence transformer model required for similarity search")
            return {}
        
        logger.info("Benchmarking similarity search...")
        
        if queries is None:
            queries = [
                "What is machine learning?",
                "How does AI work?",
                "Natural language processing applications"
            ]
        
        if corpus is None:
            corpus = [
                "Machine learning is a subset of artificial intelligence.",
                "Artificial intelligence systems can learn from data.",
                "Natural language processing helps computers understand text.",
                "Deep learning uses neural networks with multiple layers.",
                "Computer vision enables machines to interpret images.",
                "Robotics combines AI with physical systems.",
                "Data science involves extracting insights from data."
            ] * 100  # Create larger corpus
        
        # Encode corpus
        start_time = time.time()
        corpus_embeddings = self.sentence_model.encode(corpus, convert_to_tensor=True)
        corpus_encoding_time = time.time() - start_time
        
        # Encode queries and search
        start_time = time.time()
        query_embeddings = self.sentence_model.encode(queries, convert_to_tensor=True)
        
        # Compute similarities
        similarities = torch.mm(query_embeddings, corpus_embeddings.T)
        top_k = torch.topk(similarities, k=5, dim=1)
        
        search_time = time.time() - start_time
        
        results = {
            "corpus_size": len(corpus),
            "num_queries": len(queries),
            "corpus_encoding_time": corpus_encoding_time,
            "search_time": search_time,
            "total_time": corpus_encoding_time + search_time,
            "queries_per_second": len(queries) / search_time,
        }
        
        logger.info(f"Similarity search: {results['queries_per_second']:.2f} queries/sec")
        return results
    
    def apply_distillation(self, teacher_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Apply knowledge distillation for model compression"""
        logger.info(f"Applying knowledge distillation from {teacher_model_name}")
        
        # This is a simplified demonstration of the distillation concept
        teacher_model = SentenceTransformer(teacher_model_name, device=self.device)
        
        # Example texts for distillation
        texts = [
            "Knowledge distillation helps create smaller models.",
            "Teacher models provide guidance for student models.",
            "Model compression reduces computational requirements."
        ]
        
        # Get teacher embeddings
        teacher_embeddings = teacher_model.encode(texts, convert_to_tensor=True)
        student_embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)
        
        # Calculate distillation loss (MSE between embeddings)
        distillation_loss = nn.MSELoss()(student_embeddings, teacher_embeddings)
        
        results = {
            "teacher_model": teacher_model_name,
            "student_model": self.model_name,
            "distillation_loss": distillation_loss.item(),
            "teacher_embedding_dim": teacher_embeddings.shape[1],
            "student_embedding_dim": student_embeddings.shape[1],
        }
        
        logger.info(f"Distillation loss: {distillation_loss.item():.4f}")
        return results
    
    def optimize_embedding_storage(self, embeddings: torch.Tensor, compression_method: str = "pca"):
        """Optimize embedding storage and retrieval"""
        logger.info(f"Optimizing embeddings with {compression_method}")
        
        if compression_method == "pca":
            # Simplified PCA compression
            from sklearn.decomposition import PCA
            
            embeddings_np = embeddings.cpu().numpy()
            
            # Try different compression ratios
            results = {}
            for n_components in [64, 128, 256]:
                if n_components >= embeddings_np.shape[1]:
                    continue
                    
                pca = PCA(n_components=n_components)
                compressed = pca.fit_transform(embeddings_np)
                reconstruction = pca.inverse_transform(compressed)
                
                # Calculate compression ratio and reconstruction error
                compression_ratio = embeddings_np.shape[1] / n_components
                mse = np.mean((embeddings_np - reconstruction) ** 2)
                
                results[f"pca_{n_components}"] = {
                    "compression_ratio": compression_ratio,
                    "reconstruction_mse": mse,
                    "original_dim": embeddings_np.shape[1],
                    "compressed_dim": n_components
                }
        
        elif compression_method == "quantization":
            # Simple quantization to int8
            embeddings_np = embeddings.cpu().numpy()
            
            # Min-max quantization
            min_val, max_val = embeddings_np.min(), embeddings_np.max()
            scale = (max_val - min_val) / 255
            quantized = np.round((embeddings_np - min_val) / scale).astype(np.uint8)
            dequantized = quantized.astype(np.float32) * scale + min_val
            
            mse = np.mean((embeddings_np - dequantized) ** 2)
            compression_ratio = 4  # float32 to uint8
            
            results = {
                "quantization_int8": {
                    "compression_ratio": compression_ratio,
                    "reconstruction_mse": mse,
                    "storage_reduction": "75%"
                }
            }
        
        return results
    
    def export_optimized_model(self, export_format: str = "pytorch"):
        """Export optimized model"""
        logger.info(f"Exporting model in {export_format} format...")
        
        export_path = self.output_dir / f"optimized_text_encoder_{export_format}"
        export_path.mkdir(exist_ok=True)
        
        if export_format == "pytorch":
            if self.sentence_model:
                self.sentence_model.save(str(export_path))
            else:
                self.model.save_pretrained(str(export_path))
                self.tokenizer.save_pretrained(str(export_path))
        
        return str(export_path)
    
    def run_optimization_pipeline(self, model_type: str = "sentence_transformer", 
                                precision: str = "fp32", enable_distillation: bool = False):
        """Run complete optimization pipeline"""
        logger.info("Starting text encoder optimization pipeline...")
        
        results = {}
        
        # Load model
        self.load_model(model_type, precision)
        results['model_type'] = model_type
        results['precision'] = precision
        
        # Optimize tokenizer if available
        if self.tokenizer:
            tokenizer_stats = self.optimize_tokenizer()
            results['tokenizer_optimization'] = tokenizer_stats
        
        # Benchmark encoding performance
        encoding_results = self.benchmark_encoding()
        results['encoding_benchmark'] = encoding_results
        
        # Benchmark similarity search if sentence transformer
        if self.sentence_model:
            similarity_results = self.benchmark_similarity_search()
            results['similarity_benchmark'] = similarity_results
            
            # Apply distillation if requested
            if enable_distillation:
                distillation_results = self.apply_distillation()
                results['distillation'] = distillation_results
            
            # Test embedding optimization
            sample_texts = ["Sample text for embedding optimization"] * 100
            embeddings = self.sentence_model.encode(sample_texts, convert_to_tensor=True)
            
            pca_results = self.optimize_embedding_storage(embeddings, "pca")
            quant_results = self.optimize_embedding_storage(embeddings, "quantization")
            
            results['embedding_optimization'] = {
                **pca_results,
                **quant_results
            }
        
        # Export optimized model
        export_path = self.export_optimized_model()
        results['export_path'] = export_path
        
        # Save results
        results_file = self.output_dir / f"text_encoder_optimization_results_{model_type}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization complete! Results saved to {results_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Optimize text encoder models")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Model name or path")
    parser.add_argument("--model-type", choices=["sentence_transformer", "transformer"], default="sentence_transformer")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--enable-distillation", action="store_true", help="Enable knowledge distillation")
    parser.add_argument("--output-dir", default="/workspace/results", help="Output directory")
    
    args = parser.parse_args()
    
    optimizer = TextEncoderOptimizer(args.model, args.output_dir)
    results = optimizer.run_optimization_pipeline(
        args.model_type, 
        args.precision, 
        args.enable_distillation
    )
    
    print("Text Encoder Optimization Complete!")
    print(f"Results: {json.dumps({k: v for k, v in results.items() if 'benchmark' not in k}, indent=2)}")

if __name__ == "__main__":
    main()
