# ML Model Optimization Configuration
# Configure optimization parameters for different model types

# LLM Optimization Settings
LLM_CONFIG = {
    "quantization_options": ["none", "fp16", "int8", "int4"],
    "default_quantization": "int8",
    "lora_config": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "benchmark_settings": {
        "max_length": 100,
        "num_runs": 10,
        "prompt": "The future of AI is"
    }
}

# Stable Diffusion Optimization Settings
STABLE_DIFFUSION_CONFIG = {
    "precision_options": ["fp32", "fp16"],
    "default_precision": "fp16",
    "optimization_features": [
        "attention_slicing",
        "vae_slicing",
        "cpu_offloading",
        "xformers_attention"
    ],
    "benchmark_settings": {
        "num_inference_steps": 20,
        "height": 512,
        "width": 512,
        "batch_size": 2,
        "prompts": [
            "A beautiful landscape with mountains and lakes",
            "A futuristic cityscape at sunset",
            "Portrait of a robot in cyberpunk style",
            "Abstract art with vibrant colors"
        ]
    }
}

# Text Encoder Optimization Settings
TEXT_ENCODER_CONFIG = {
    "model_types": ["sentence_transformer", "transformer"],
    "default_model_type": "sentence_transformer",
    "precision_options": ["fp32", "fp16"],
    "default_precision": "fp32",
    "batch_sizes": [1, 8, 16, 32],
    "compression_methods": ["pca", "quantization"],
    "pca_components": [64, 128, 256],
    "benchmark_texts": [
        "This is a sample sentence for benchmarking.",
        "Natural language processing is fascinating.",
        "Machine learning models require optimization.",
        "Text encoding is a crucial step in NLP pipelines.",
        "Transformers have revolutionized the field of AI.",
        "Efficient processing is key for production systems.",
        "Benchmarking helps identify performance bottlenecks.",
        "Optimization techniques can significantly improve speed."
    ]
}

# Audio Embedding Optimization Settings
AUDIO_CONFIG = {
    "precision_options": ["fp32", "fp16"],
    "default_precision": "fp32",
    "sample_rate": 16000,
    "batch_sizes": [1, 4, 8, 16],
    "audio_duration": 3.0,
    "num_test_samples": 50,
    "preprocessing_techniques": [
        "original",
        "normalized", 
        "high_pass_filtered",
        "noise_reduced"
    ],
    "feature_extraction_methods": [
        "raw_waveform",
        "mel_spectrogram",
        "mfcc",
        "chroma",
        "spectral_contrast"
    ]
}

# Hardware Optimization Settings
HARDWARE_CONFIG = {
    "gpu_memory_fraction": 0.9,
    "enable_mixed_precision": True,
    "enable_gradient_checkpointing": True,
    "max_batch_size_auto_detect": True,
    "memory_optimization_level": "aggressive",  # conservative, moderate, aggressive
    "cuda_benchmark": True,
    "dataloader_num_workers": 4
}

# Export Settings
EXPORT_CONFIG = {
    "formats": ["pytorch", "onnx", "tensorrt"],
    "default_format": "pytorch",
    "onnx_opset_version": 11,
    "tensorrt_precision": "fp16",
    "optimization_level": "O2"
}

# Monitoring and Logging
MONITORING_CONFIG = {
    "log_level": "INFO",
    "enable_wandb": False,
    "enable_tensorboard": True,
    "save_intermediate_results": True,
    "benchmark_iterations": 10,
    "memory_profiling": True,
    "timing_profiling": True
}
