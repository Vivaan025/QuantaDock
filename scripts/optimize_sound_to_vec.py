#!/usr/bin/env python3
"""
Sound-to-Vector (Audio Embedding) Optimization Script
Optimizes audio processing models for converting sound to vector representations:
- Audio feature extraction
- Speech-to-embedding models
- Audio classification models
- Wav2Vec, Whisper, and other audio transformers
"""

import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model,
    WhisperProcessor, WhisperModel,
    AutoFeatureExtractor, AutoModel
)
import json
import time
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioEmbeddingOptimizer:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", 
                 output_dir: str = "/workspace/results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, precision: str = "fp32"):
        """Load audio embedding model"""
        logger.info(f"Loading audio model: {self.model_name}")
        
        torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        
        try:
            if "wav2vec2" in self.model_name.lower():
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype
                ).to(self.device)
            elif "whisper" in self.model_name.lower():
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype
                ).to(self.device)
            else:
                # Generic approach
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype
                ).to(self.device)
                
            logger.info(f"Model loaded successfully with {precision} precision")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_synthetic_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic audio for testing"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a complex waveform with multiple frequencies
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.1 * np.sin(2 * np.pi * 1320 * t) + # E6 note
            0.05 * np.random.randn(len(t))       # Add some noise
        )
        
        return audio.astype(np.float32)
    
    def create_audio_dataset(self, num_samples: int = 100, duration: float = 3.0) -> List[np.ndarray]:
        """Create a synthetic audio dataset for benchmarking"""
        logger.info(f"Creating synthetic audio dataset with {num_samples} samples")
        
        dataset = []
        sample_rate = 16000
        
        for i in range(num_samples):
            # Vary the audio characteristics
            base_freq = 220 + (i % 10) * 55  # Vary base frequency
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = (
                0.4 * np.sin(2 * np.pi * base_freq * t) +
                0.2 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 3 * t) +
                0.05 * np.random.randn(len(t))
            )
            
            dataset.append(audio.astype(np.float32))
        
        return dataset
    
    def benchmark_audio_encoding(self, audio_samples: List[np.ndarray] = None, batch_sizes: List[int] = [1, 4, 8, 16]):
        """Benchmark audio encoding performance"""
        logger.info("Benchmarking audio encoding performance...")
        
        if audio_samples is None:
            audio_samples = self.create_audio_dataset(num_samples=64, duration=2.0)
        
        results = {}
        sample_rate = 16000
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Prepare batches
            batches = [audio_samples[i:i+batch_size] for i in range(0, len(audio_samples), batch_size)]
            
            # Warm up
            warm_up_batch = audio_samples[:min(3, len(audio_samples))]
            try:
                if "wav2vec2" in self.model_name.lower():
                    inputs = self.processor(warm_up_batch, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                else:
                    # Generic processing
                    inputs = self.processor(warm_up_batch, sampling_rate=sample_rate, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            except Exception as e:
                logger.warning(f"Warm-up failed: {e}")
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            total_processed = 0
            
            for batch in batches[:10]:  # Test first 10 batches
                try:
                    if "wav2vec2" in self.model_name.lower():
                        inputs = self.processor(batch, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                    else:
                        inputs = self.processor(batch, sampling_rate=sample_rate, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                    
                    total_processed += len(batch)
                    
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
                    continue
            
            end_time = time.time()
            
            total_time = end_time - start_time
            if total_processed > 0:
                audio_per_second = total_processed / total_time
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                results[f"batch_size_{batch_size}"] = {
                    "total_time": total_time,
                    "audio_clips_per_second": audio_per_second,
                    "peak_memory_gb": peak_memory,
                    "total_audio_processed": total_processed
                }
                
                logger.info(f"Batch size {batch_size}: {audio_per_second:.2f} clips/sec, {peak_memory:.2f} GB memory")
            else:
                logger.warning(f"No audio processed for batch size {batch_size}")
        
        return results
    
    def analyze_audio_features(self, audio: np.ndarray, sample_rate: int = 16000):
        """Analyze audio features using librosa"""
        logger.info("Analyzing audio features...")
        
        features = {}
        
        try:
            # Spectral features
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
            features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)))
            features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            features['mfcc_mean'] = [float(x) for x in np.mean(mfccs, axis=1)]
            features['mfcc_std'] = [float(x) for x in np.std(mfccs, axis=1)]
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            features['chroma_mean'] = [float(x) for x in np.mean(chroma, axis=1)]
            
            # Zero crossing rate
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features['tempo'] = float(tempo)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features['error'] = str(e)
        
        return features
    
    def optimize_audio_preprocessing(self, audio_samples: List[np.ndarray]):
        """Optimize audio preprocessing pipeline"""
        logger.info("Optimizing audio preprocessing...")
        
        preprocessing_results = {}
        sample_rate = 16000
        
        # Test different preprocessing techniques
        techniques = {
            "original": lambda x: x,
            "normalized": lambda x: x / (np.max(np.abs(x)) + 1e-8),
            "high_pass_filtered": lambda x: librosa.effects.preemphasis(x),
            "noise_reduced": lambda x: x * (np.abs(x) > 0.01),  # Simple noise gate
        }
        
        for technique_name, technique_func in techniques.items():
            start_time = time.time()
            
            processed_samples = []
            for audio in audio_samples[:20]:  # Test on subset
                try:
                    processed = technique_func(audio)
                    processed_samples.append(processed)
                except Exception as e:
                    logger.warning(f"Preprocessing failed for {technique_name}: {e}")
                    processed_samples.append(audio)
            
            preprocessing_time = time.time() - start_time
            
            # Analyze feature consistency
            if processed_samples:
                features = self.analyze_audio_features(processed_samples[0], sample_rate)
                
                preprocessing_results[technique_name] = {
                    "preprocessing_time": preprocessing_time,
                    "samples_per_second": len(processed_samples) / preprocessing_time,
                    "sample_features": {k: v for k, v in features.items() if k != 'error'}
                }
        
        return preprocessing_results
    
    def benchmark_feature_extraction(self, audio_samples: List[np.ndarray]):
        """Benchmark different feature extraction methods"""
        logger.info("Benchmarking feature extraction methods...")
        
        extraction_results = {}
        sample_rate = 16000
        
        # Test different feature extraction methods
        methods = {
            "raw_waveform": lambda x: x,
            "mel_spectrogram": lambda x: librosa.feature.melspectrogram(y=x, sr=sample_rate),
            "mfcc": lambda x: librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13),
            "chroma": lambda x: librosa.feature.chroma_stft(y=x, sr=sample_rate),
            "spectral_contrast": lambda x: librosa.feature.spectral_contrast(y=x, sr=sample_rate),
        }
        
        for method_name, method_func in methods.items():
            start_time = time.time()
            
            features = []
            for audio in audio_samples[:10]:  # Test on subset
                try:
                    feature = method_func(audio)
                    features.append(feature)
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {method_name}: {e}")
            
            extraction_time = time.time() - start_time
            
            if features:
                feature_shape = features[0].shape if hasattr(features[0], 'shape') else len(features[0])
                
                extraction_results[method_name] = {
                    "extraction_time": extraction_time,
                    "features_per_second": len(features) / extraction_time,
                    "feature_shape": feature_shape,
                    "feature_dimensionality": np.prod(feature_shape) if hasattr(features[0], 'shape') else len(features[0])
                }
        
        return extraction_results
    
    def export_optimized_model(self, export_format: str = "pytorch"):
        """Export optimized audio model"""
        logger.info(f"Exporting model in {export_format} format...")
        
        export_path = self.output_dir / f"optimized_audio_model_{export_format}"
        export_path.mkdir(exist_ok=True)
        
        if export_format == "pytorch":
            self.model.save_pretrained(str(export_path))
            if hasattr(self.processor, 'save_pretrained'):
                self.processor.save_pretrained(str(export_path))
        
        return str(export_path)
    
    def run_optimization_pipeline(self, precision: str = "fp32", enable_preprocessing_optimization: bool = True):
        """Run complete audio optimization pipeline"""
        logger.info("Starting audio embedding optimization pipeline...")
        
        results = {}
        
        # Load model
        self.load_model(precision)
        results['precision'] = precision
        
        # Create test dataset
        audio_samples = self.create_audio_dataset(num_samples=50, duration=2.0)
        
        # Benchmark encoding performance
        encoding_results = self.benchmark_audio_encoding(audio_samples)
        results['encoding_benchmark'] = encoding_results
        
        # Optimize preprocessing if requested
        if enable_preprocessing_optimization:
            preprocessing_results = self.optimize_audio_preprocessing(audio_samples)
            results['preprocessing_optimization'] = preprocessing_results
            
            # Benchmark feature extraction
            feature_extraction_results = self.benchmark_feature_extraction(audio_samples)
            results['feature_extraction_benchmark'] = feature_extraction_results
        
        # Analyze sample audio features
        sample_features = self.analyze_audio_features(audio_samples[0])
        results['sample_audio_analysis'] = sample_features
        
        # Export optimized model
        export_path = self.export_optimized_model()
        results['export_path'] = export_path
        
        # Save results
        results_file = self.output_dir / f"audio_optimization_results_{precision}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization complete! Results saved to {results_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Optimize audio embedding models")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h", help="Model name or path")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--enable-preprocessing", action="store_true", help="Enable preprocessing optimization")
    parser.add_argument("--output-dir", default="/workspace/results", help="Output directory")
    
    args = parser.parse_args()
    
    optimizer = AudioEmbeddingOptimizer(args.model, args.output_dir)
    results = optimizer.run_optimization_pipeline(
        args.precision, 
        args.enable_preprocessing
    )
    
    print("Audio Embedding Optimization Complete!")
    print(f"Results: {json.dumps({k: v for k, v in results.items() if 'benchmark' not in k}, indent=2)}")

if __name__ == "__main__":
    main()
