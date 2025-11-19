"""
Video Transcription App with GPU Optimization
Version: 2.1
Author: Your Name
License: MIT
"""

# Environment variables must be set before other imports
import os
os.environ.update({
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_LAUNCH_BLOCKING": "1",
    "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION": "0"
})

# Streamlit import
import streamlit as st

# Standard library imports
import gc
import json
import logging
import re
import subprocess
import tempfile
import threading
import warnings
import concurrent.futures
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Union

# Third party imports
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*no running event loop.*')
warnings.filterwarnings('ignore', message='.*attention mask.*')
warnings.filterwarnings('ignore', message='.*forced_decoder_ids.*')
warnings.filterwarnings('ignore', message='.*meta device.*')

# Custom exceptions
class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass

class FFmpegNotFoundError(Exception):
    """Custom exception for missing ffmpeg"""
    pass

class GPUError(Exception):
    """Custom exception for GPU-related errors"""
    pass

# GPU initialization and management
class GPUManager:
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.device = "cuda:0" if self.has_gpu else "cpu"
        self.dtype = torch.float16 if self.has_gpu else torch.float32
        
        if self.has_gpu:
            self.init_gpu()
    
    def init_gpu(self) -> None:
        """Initialize GPU with optimal settings"""
        try:
            # Basic CUDA settings
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # Advanced optimizations for RTX 4090
            if self.get_vram_gb() >= 24:  # RTX 4090 specific
                torch.backends.cuda.enable_mem_efficient_sdp = True
                if hasattr(torch.backends.cuda, 'memory_efficient_fusion'):
                    torch.backends.cuda.memory_efficient_fusion = True
                # Additional RTX 4090 optimizations
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            raise GPUError(f"GPU initialization failed: {str(e)}")
    
    def get_vram_gb(self) -> float:
        """Get total VRAM in GB"""
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get current GPU status and metrics"""
        if not self.has_gpu:
            return None

        try:
            info = {
                "device_name": torch.cuda.get_device_name(0),
                "total_memory": self.get_vram_gb(),
                "allocated_memory": torch.cuda.memory_allocated() / (1024**3),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__
            }

            try:
                smi = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=temperature.gpu,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ], text=True).strip().split(',')
                
                info.update({
                    "temperature": float(smi[0]),
                    "utilization": float(smi[1])
                })
            except:
                pass

            return info
            
        except Exception as e:
            logger.warning(f"Error getting GPU info: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """Free GPU memory"""
        if self.has_gpu:
            torch.cuda.empty_cache()
            gc.collect()

# Initialize GPU manager
gpu = GPUManager()

class AudioProcessor:
    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
    
    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable"""
        common_locations = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            str(Path.home() / "AppData/Local/Programs/ffmpeg/bin/ffmpeg.exe")
        ]
        
        from shutil import which
        ffmpeg_path = which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
            
        for loc in common_locations:
            if Path(loc).exists():
                return str(Path(loc))
                
        raise FFmpegNotFoundError("ffmpeg not found. Please install ffmpeg and add it to PATH.")
    
    def extract_audio(self, video_path: Path) -> str:
        """Extract audio from video or return path if already MP3"""
        if str(video_path).lower().endswith('.mp3'):
            return str(video_path)

        try:
            output_path = Path(tempfile.mktemp(suffix='.mp3'))
            
            input_path = str(video_path).replace('\\', '/')
            output_path_str = str(output_path).replace('\\', '/')
            
            ffmpeg_cmd = f'"{self.ffmpeg_path}" -hide_banner -y -i "{input_path}" -q:a 0 -map 0:a "{output_path_str}"'
            
            process = subprocess.run(
                ffmpeg_cmd,
                shell=True,
                capture_output=True,
                check=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise ValueError("Audio extraction failed")
                
            return str(output_path)
            
        except Exception as e:
            if 'output_path' in locals() and output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Audio extraction failed: {str(e)}")

class TranscriptionManager:
    def __init__(self, model_name: str, language: str):
        self.model_name = model_name
        self.language = language
        self.audio_processor = AudioProcessor()
        self.pipe = self._init_model()
        
    def _init_model(self):
        """Initialize Whisper model with GPU optimization"""
        try:
            # Clear GPU memory first
            if gpu.has_gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            processor = WhisperProcessor.from_pretrained(self.model_name)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=gpu.dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"  # Use efficient attention
            )

            if gpu.has_gpu:
                model.eval()
                model = model.to(gpu.device)
                # Enable memory efficient attention
                model.config.use_cache = True

            return pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=gpu.dtype,
                device=gpu.device if gpu.has_gpu else "cpu"
            )
        except Exception as e:
            raise TranscriptionError(f"Model initialization failed: {str(e)}")
    
    def _get_params(self) -> dict:
        """Get optimized transcription parameters"""
        params = {
            "batch_size": 16,
            "chunk_length_s": 30,
            "stride_length_s": 2,
            "return_timestamps": True,  # Chunk-level timestamps (less memory intensive)
            "generate_kwargs": {
                "language": self.language,
                "task": "transcribe"
            }
        }

        if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Very conservative settings
            params.update({
                "batch_size": 8,  # Very small batch size to prevent memory issues
                "chunk_length_s": 20,  # Smaller chunks
                "stride_length_s": 2,  # Less overlap
                "num_workers": 1  # Single worker
            })

        return params
    
    def format_transcript_with_timestamps(self, result: dict) -> str:
        """Format transcript with timestamps"""
        if 'chunks' not in result:
            return result.get('text', '')

        formatted_text = []
        for chunk in result['chunks']:
            if 'timestamp' in chunk and chunk['timestamp']:
                start_time = chunk['timestamp'][0]
                end_time = chunk['timestamp'][1]

                # Format time as MM:SS or HH:MM:SS
                if start_time >= 3600:
                    start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
                    end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}"
                else:
                    start_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    end_str = f"{int(end_time//60):02d}:{int(end_time%60):02d}"

                formatted_text.append(f"[{start_str} - {end_str}] {chunk['text']}")
            else:
                formatted_text.append(chunk['text'])

        return '\n\n'.join(formatted_text)

    def transcribe(self, file_path: Path) -> dict:
        """Transcribe audio file"""
        try:
            gpu.cleanup()

            if not str(file_path).lower().endswith('.mp3'):
                audio_path = self.audio_processor.extract_audio(file_path)
            else:
                audio_path = str(file_path)

            try:
                params = self._get_params()
                result = self.pipe(audio_path, **params)

                # Format transcript with timestamps
                if 'chunks' in result:
                    result['formatted_text'] = self.format_transcript_with_timestamps(result)
                    result['text'] = result['formatted_text']

                if audio_path != str(file_path):
                    try:
                        os.unlink(audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {e}")

                return result

            except Exception as e:
                raise TranscriptionError(f"Transcription failed: {str(e)}")

        finally:
            gpu.cleanup()

# Initialize GPU
has_gpu = gpu.has_gpu

# Import ML components
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)

class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass

# Model initialization and parameter functions are handled by TranscriptionManager class

def transcribe_audio(file_path: Path, model_name: str, language: str) -> dict:
    """
    Transcribe audio with optimized error handling and GPU monitoring
    """
    try:
        # Initialize GPU if available
        if has_gpu:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log initial GPU state
            info = get_gpu_info()
            if info:
                logger.info(f"Initial GPU Memory: {info['allocated_memory']:.1f}GB / {info['total_memory']:.1f}GB")
        
        # Convert video to audio if needed
        if not str(file_path).lower().endswith('.mp3'):
            audio_path = extract_audio(file_path)
        else:
            audio_path = str(file_path)
        
        try:
            # Initialize model and get parameters
            pipe = init_model(model_name)
            params = get_transcription_params(language)
            
            # Process audio
            result = pipe(audio_path, **params)
            
            # Cleanup temporary file
            if audio_path != str(file_path):
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
            
            return result
            
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")
            
    finally:
        # Cleanup
        if has_gpu:
            torch.cuda.empty_cache()
            gc.collect()

class FFmpegNotFoundError(Exception):
    """Custom exception for missing ffmpeg"""
    pass

def find_ffmpeg() -> str:
    """Find ffmpeg executable in system PATH or common locations"""
    common_locations = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        str(Path.home() / "AppData/Local/Programs/ffmpeg/bin/ffmpeg.exe")
    ]
    
    # Check PATH first
    from shutil import which
    ffmpeg_path = which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
        
    # Check common locations
    for loc in common_locations:
        if Path(loc).exists():
            return str(Path(loc))
            
    raise FFmpegNotFoundError("ffmpeg not found. Please install ffmpeg and add it to PATH.")

def extract_audio(video_path: Path) -> str:
    """Extract audio from video file or return path if already MP3"""
    if str(video_path).lower().endswith('.mp3'):
        return str(video_path)

    try:
        # Find ffmpeg
        ffmpeg_exe = find_ffmpeg()
        
        # Create temporary output file
        output_path = Path(tempfile.mktemp(suffix='.mp3'))
        
        # Normalize paths
        input_path = str(video_path).replace('\\', '/')
        output_path_str = str(output_path).replace('\\', '/')
        
        # Create ffmpeg command
        ffmpeg_cmd = f'"{ffmpeg_exe}" -hide_banner -y -i "{input_path}" -q:a 0 -map 0:a "{output_path_str}"'
        
        # Run ffmpeg
        process = subprocess.run(
            ffmpeg_cmd,
            shell=True,
            capture_output=True,
            check=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if process.stderr:
            logger.info(f"FFmpeg output:\n{process.stderr}")
            
        # Verify output
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ValueError("Audio extraction failed")
            
        return str(output_path)
        
    except Exception as e:
        if 'output_path' in locals() and output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Audio extraction failed: {str(e)}")

def process_files(
    files: List[Path],
    model_name: str,
    language: str,
    skip_existing: bool = True
) -> dict:
    """Process multiple files with progress tracking"""

    results = {}
    progress = st.progress(0)
    status = st.empty()

    try:
        # Filter existing files if requested
        if skip_existing:
            files = [f for f in files if not f.with_suffix('.txt').exists()]
            
        if not files:
            st.info("No files to process")
            return results
            
        # Process files
        for i, file in enumerate(files):
            try:
                status.info(f"Processing: {file.name}")
                
                result = transcribe_audio(file, model_name, language)
                results[file.name] = result
                
                # Save transcript
                transcript_path = file.with_suffix('.txt')
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                    
                st.success(f"✓ Completed: {file.name}")
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                
            progress.progress((i + 1) / len(files))
            
        return results
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return results

# Initialize GPU - configuration is handled by GPUManager instance
if gpu.has_gpu:
    logger.info(f"GPU: {gpu.get_info()['device_name']}")

def transcribe_audio(file_path: Path, model_name: str, language: str) -> dict:
    """
    Transcribe audio file using TranscriptionManager
    """
    try:
        # Create transcription manager
        manager = TranscriptionManager(model_name=model_name, language=language)
        
        # Log GPU state before processing
        if gpu.has_gpu:
            info = gpu.get_info()
            if info:
                logger.info(f"GPU Memory: {info['allocated_memory']:.1f}GB / {info['total_memory']:.1f}GB")
        
        # Process audio using manager
        result = manager.transcribe(file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def init_model(model_name: str):
    """Initialize model by creating a temporary TranscriptionManager"""
    manager = TranscriptionManager(model_name=model_name, language="auto")
    return manager.pipe

def get_transcription_params(language: str = "en") -> dict:
    """Get optimized transcription parameters by creating a temporary TranscriptionManager"""
    manager = TranscriptionManager(model_name="openai/whisper-large-v3", language=language)
    return manager._get_params()

# Suppress non-critical warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*no running event loop.*')
warnings.filterwarnings('ignore', message='.*attention mask.*')
warnings.filterwarnings('ignore', message='.*forced_decoder_ids.*')
warnings.filterwarnings('ignore', message='.*meta device.*')

# Import transformers components
# Import remaining dependencies
from transformers import pipeline
from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import tempfile
import subprocess
import time
import re
from datetime import datetime, timedelta
import json
import requests
from functools import partial
from queue import Queue, Empty
from app0 import get_next_api_key, BASE_URL

BASE_URL = "https://www.googleapis.com/youtube/v3"

# Remove duplicate function - we'll use the one from GPUManager class

def get_model_config():
    """Get model configuration based on current GPU manager settings"""
    config = {
        "torch_dtype": gpu.dtype,
        "device_map": gpu.device,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "use_cache": True if torch.__version__ >= "2.0.0" else False
    }
    return config

def get_whisper_settings(language: str = "en"):
    """Get optimized Whisper model settings"""
    settings = {
        "model_config": get_model_config(),
        "pipeline_settings": {
            "device": gpu.device,
            "framework": "pt",
            "batch_size": 16,
            "chunk_length_s": 30,
            "stride_length_s": 2,
            "num_workers": min(4, os.cpu_count() or 1) if gpu.has_gpu else 1
        }
    }
    
    if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Conservative
        settings["pipeline_settings"].update({
            "batch_size": 32 if language == "en" else 24,  # Conservative batch sizes
            "chunk_length_s": 30,  # Standard chunk length
            "stride_length_s": 3,  # Moderate overlap
            "num_workers": min(2, os.cpu_count() or 1)  # Limit parallel workers
        })
    
    return settings

def transcribe_audio(audio_path, model_name, language, batch_size, chunk_length):
    """Transcribe audio with enhanced error recovery and performance monitoring"""
    retry_count = 0
    max_retries = 3
    last_error = None
    monitor_thread = None

    def start_gpu_monitoring():
        if not torch.cuda.is_available():
            return None, {}

        perf_metrics = {
            "initial_state": None,
            "model_load": None,
            "transcription": [],
            "stage_timings": {
                "model_load": 0,
                "processing": 0,
                "total": 0
            },
            "memory": {
                "initial": 0,
                "peak": 0,
                "current": 0
            },
            "gpu": {
                "max_util": 0,
                "avg_util": 0,
                "samples": [],
                "power_samples": [],
                "temp_samples": []
            }
        }

        def periodic_monitor():
            while not getattr(periodic_monitor, "stop", False):
                metrics = get_gpu_info()
                if metrics:
                    perf_metrics["transcription"].append(metrics)
                    
                    # Alert on issues
                    if metrics["temperature"] > 80:
                        logger.warning(f"🌡️ High temperature: {metrics['temperature']}°C")
                    if metrics["utilization"] > 95:
                        logger.warning(f"⚡ High utilization: {metrics['utilization']}%")
                time.sleep(0.1)

        monitor_thread = threading.Thread(target=periodic_monitor)
        monitor_thread.start()
        return monitor_thread, perf_metrics

    while retry_count < max_retries:
        try:
            # Initialize performance monitoring
            start_time = time.time()
            monitor_thread, perf_metrics = start_gpu_monitoring()

            # Convert video to audio if needed
            audio_file = extract_audio(audio_path)

            # Initialize GPU
            if torch.cuda.is_available():
                # Basic memory management
                torch.cuda.empty_cache()
                gc.collect()

                # Log GPU state
                info = get_gpu_info()
                if info:
                    logger.info(f"""
                    GPU Status:
                    Memory: {info['allocated_memory']/1024:.1f}GB / {info['total_memory']/1024:.1f}GB
                    """)

            # Load model
            processor = AutoProcessor.from_pretrained(model_name)
            
            st.info("Loading Whisper model...")
            
            # Create processor first
            processor = WhisperProcessor.from_pretrained(model_name)
            
            # Configure GPU settings
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            try:
                # Load model with basic settings first
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(device)
                
                if torch.cuda.is_available():
                    # Apply GPU optimizations
                    model.eval()
                    torch.cuda.empty_cache()
                    
                    # Enable efficient memory usage
                    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                        torch.backends.cuda.enable_mem_efficient_sdp = True
                
                # Create optimized pipeline
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=device
                )
                
                st.success("✨ Model loaded successfully")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                raise

            # Configure transcription parameters
            transcribe_params = {
                "batch_size": 16,  # Start with conservative batch size
                "chunk_length_s": 30,
                "stride_length_s": 2,
                "return_timestamps": True,
                "generate_kwargs": {
                    "language": language,
                    "task": "transcribe"
                }
            }
            
            if torch.cuda.is_available():
                # Adjust batch size based on VRAM - RTX 4090 conservative
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if vram_gb >= 24:  # RTX 4090
                    transcribe_params.update({
                        "batch_size": 24,  # Conservative batch size for stability
                        "chunk_length_s": 30,  # Standard chunk length
                        "stride_length_s": 3,  # Moderate overlap
                        "num_workers": min(2, os.cpu_count() or 1)
                    })

            result = pipe(audio_file, **transcribe_params)

            # Clean up
            if torch.cuda.is_available():
                if monitor_thread:
                    setattr(periodic_monitor, "stop", True)
                    monitor_thread.join(timeout=1.0)
                torch.cuda.empty_cache()
                gc.collect()

            # Remove temporary file
            if audio_file != str(audio_path):
                try:
                    os.unlink(audio_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")

            # Log performance
            elapsed_time = time.time() - start_time
            logger.info(f"""
            ✨ Transcription completed in {elapsed_time:.1f}s
            Memory peak: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB
            """)

            return result

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"🔄 GPU OOM error, attempt {retry_count}/{max_retries}")
                torch.cuda.empty_cache()
                gc.collect()
                batch_size = max(8, batch_size // 2)
                time.sleep(1)
                continue
            raise RuntimeError(f"GPU memory error after {max_retries} retries") from e

        except (OSError, IOError) as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"🔄 I/O error, attempt {retry_count}/{max_retries}: {str(e)}")
                time.sleep(2)
                continue
            raise RuntimeError(f"I/O error after {max_retries} retries: {str(e)}") from e

        except Exception as e:
            logger.error(f"❌ Transcription error: {str(e)}")
            raise

    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")

def normalize_path(path):
    if not path:
        return path

    try:
        # Handle paths correctly, including spaces and special characters
        resolved_path = Path(path.strip()).resolve(strict=False)
        return resolved_path
    except Exception as e:
        st.error(f"Error normalizing path: {str(e)}")
        return Path(path)

def init_gpu_settings():
    if torch.cuda.is_available():
        try:
            # Advanced cleanup and optimization
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            
            # RTX 4090 optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_mem_efficient_sdp = True
            
            # Get GPU info
            info = get_gpu_info()
            if info:
                st.success(f"GPU: {info['device_name']} | CUDA {info['cuda_version']}")
                st.info(f"""
                🚀 RTX 4090 Configuration:
                
                💾 Memory:
                - Total VRAM: {info['total_memory']/1024:.1f} GB
                - In Use: {info['allocated_memory']/1024:.1f} GB
                
                ⚡ Advanced Settings:
                - Scaled Dot-Product Attention
                - TF32 Enabled
                - Mixed Precision (FP16)
                - Memory Efficient Attention
                - Large Batch Processing
                """)
            
            return True
            
        except Exception as e:
            st.error(f"GPU setup error: {str(e)}")
            st.warning("Falling back to CPU processing")
            return False

def check_gpu_and_model(get_metrics_func):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        init_gpu_settings()
        metrics = get_metrics_func()
        
        if metrics:
            st.success(f"GPU: {metrics['device_name']} ({metrics['total_memory']/1024:.1f} GB)")
            st.info(f"CUDA: {metrics['cuda_version']} | PyTorch: {metrics['torch_version']}")
            
            # Display GPU status
            st.info(f"""
            GPU Status:
            💾 Memory: {metrics['allocated_memory']/1024:.1f} GB used / {metrics['total_memory']/1024:.1f} GB total
            {'⚡ GPU Load: ' + str(int(metrics['utilization'])) + '%' if metrics.get('utilization') is not None else ''}
            {'🌡️ Temp: ' + str(metrics.get('temperature')) + '°C' if metrics.get('temperature') is not None else ''}
            """)
            
            if torch.backends.cuda.matmul.allow_tf32:
                st.info("✓ TF32 enabled for better performance")
    else:
        st.warning("No GPU found - Transcription running on CPU")
    return device

def sanitize_filename(filepath):
    """Cleans the filename and renames if necessary"""
    try:
        path = Path(filepath)
        parent = path.parent
        stem = path.stem
        suffix = path.suffix
        
        # Remove problematic characters, but keep date and VideoID
        # Format: YYYY-MM-DD_Title_VideoID.ext
        parts = stem.split('_')
        
        if len(parts) >= 3:  # Expected format
            date = parts[0]  # YYYY-MM-DD
            video_id = parts[-1]  # VideoID at end
            title = '_'.join(parts[1:-1])  # Everything in between is title
            
            # Clean only the title
            clean_title = ''.join(c for c in title if c.isalnum() or c in ' -_.')
            clean_title = clean_title.strip()
            
            # Reassemble filename
            new_stem = f"{date}_{clean_title}_{video_id}"
        else:
            # Fallback: Clean entire name
            new_stem = ''.join(c for c in stem if c.isalnum() or c in ' -_.')
            
        new_name = new_stem + suffix
        new_path = parent / new_name
        
        # Rename if necessary
        if str(path) != str(new_path):
            st.info(f"Cleaning filename:\nOld: {path.name}\nNew: {new_path.name}")
            path.rename(new_path)
            return new_path
            
        return path
        
    except Exception as e:
        st.error(f"Error cleaning filename: {str(e)}")
        return path  # Return original on error

def find_ffmpeg():
    """Find ffmpeg executable in PATH or common locations"""
    # Common Windows locations
    common_locations = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.1-full_build/bin/ffmpeg.exe"),
        str(Path.home() / "AppData/Local/Programs/ffmpeg/bin/ffmpeg.exe")
    ]
    
    # First check PATH
    from shutil import which
    ffmpeg_path = which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Then check common locations
    for loc in common_locations:
        if Path(loc).exists():
            return str(Path(loc))
    
    raise FileNotFoundError("ffmpeg not found. Please install ffmpeg and add it to PATH.")

def extract_audio(video_path: Path) -> str:
    """Extract audio from video or return path if it's already an MP3"""
    if str(video_path).lower().endswith('.mp3'):
        return str(video_path)

    output_path = None
    try:
        # Find ffmpeg
        ffmpeg_exe = find_ffmpeg()
        # Prepare input and output paths
        clean_path = sanitize_filename(video_path)
        if not clean_path.exists():
            raise FileNotFoundError(f"File not found: {video_path}")
        
        output_path = Path(tempfile.mktemp(suffix='.mp3'))
        st.info(f"Processing file: {clean_path.name}")

        # Normalize paths for ffmpeg
        input_path = str(clean_path)
        if input_path.startswith('\\'):
            # Network path - convert to forward slashes but keep double backslash at start
            path_without_prefix = input_path.lstrip('\\')
            input_path = '//' + path_without_prefix.replace('\\', '/')
        else:
            # Local path - convert to forward slashes
            input_path = input_path.replace('\\', '/')

        # Convert output path to forward slashes
        output_path_str = str(output_path).replace('\\', '/')

        # Create ffmpeg command string with proper quoting and forward slashes
        ffmpeg_cmd = f'"{ffmpeg_exe}" -hide_banner -y -i "{input_path}" -q:a 0 -map 0:a -af "loudnorm=I=-16:LRA=11:TP=-1.5" "{output_path_str}"'
        
        try:
            # Run ffmpeg with proper path handling
            process = subprocess.run(
                ffmpeg_cmd,
                shell=True,
                capture_output=True,
                check=True,
                encoding='utf-8',
                errors='replace',
                env={
                    'PYTHONIOENCODING': 'utf-8',
                    'PATH': os.environ['PATH'],
                    'SYSTEMROOT': os.environ.get('SYSTEMROOT', '')
                }
            )
            
            if process.stderr:
                st.info(f"FFmpeg Output:\n{process.stderr}")
                
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg Error (Code {e.returncode}):\n{e.stderr}")
            if output_path and output_path.exists():
                output_path.unlink()
            raise

        # Verify output file
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not created: {output_path}")
        if output_path.stat().st_size == 0:
            raise ValueError("Output file is empty")

        st.success("Audio extraction successful")
        return str(output_path)

    except Exception as e:
        st.error(f"Error in audio extraction: {str(e)}")
        if output_path and output_path.exists():
            output_path.unlink()
        raise

def extract_video_id(filename):
    """Extract YouTube video ID from filename"""
    base_name = Path(filename).stem
    parts = base_name.split('_')
    
    if not parts:
        return None

    # Check last part first
    last_part = parts[-1]
    if len(last_part) == 11 and is_valid_video_id(last_part):
        return last_part
        
    # Try end of last part
    if len(last_part) > 11:
        end_part = last_part[-11:]
        if is_valid_video_id(end_part):
            return end_part
            
    return None

def is_valid_video_id(candidate):
    """Validate YouTube video ID format"""
    return len(candidate) == 11 and bool(re.match(r'^[\w-]{11}$', candidate))

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def format_chunk_with_timestamp(chunk):
    try:
        if chunk.get("timestamp") and None not in chunk["timestamp"]:
            start_time = format_time(chunk["timestamp"][0])
            end_time = format_time(chunk["timestamp"][1])
            return f"[{start_time} - {end_time}]\n{chunk['text']}\n\n"
        return f"[Timestamps unavailable]\n{chunk['text']}\n\n"
    except (TypeError, KeyError, AttributeError):
        return f"{chunk.get('text', '')}\n\n"

def process_files(files_to_process: List[Path], model_name: str, language: str, skip_existing: bool = True) -> dict:
    """Process multiple files in parallel with GPU acceleration"""
    # Initialize UI components
    status_area = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Remove duplicates and filter existing files
        files_to_process = list(dict.fromkeys(files_to_process))
        original_count = len(files_to_process)
        
        if skip_existing:
            files_to_process = [
                f for f in files_to_process
                if not f.with_suffix('.txt').exists()
            ]
            skipped = original_count - len(files_to_process)
            if skipped > 0:
                status_area.info(f"Skipping {skipped} existing files")

        if not files_to_process:
            status_area.info("No files to process")
            return {}

        status_area.info(f"Processing {len(files_to_process)} files...")
        results = {}
        completed = 0

        def safe_status_update(msg_type: str, msg: str) -> None:
            try:
                if hasattr(status_area, msg_type):
                    getattr(status_area, msg_type)(msg)
            except Exception as e:
                logger.warning(f"Status update error: {e}")

        # Configure sequential processing (to avoid Streamlit context issues)
        status_funcs = {
            "success": lambda msg: safe_status_update("success", msg),
            "error": lambda msg: safe_status_update("error", msg)
        }

        # Process files sequentially to avoid threading issues with Streamlit
        for file in files_to_process:
            try:
                success, filename, result = parallel_transcribe_worker(
                    Path(file),
                    model_name,
                    language,
                    status_funcs
                )
                completed += 1

                if success and result:
                    results[filename] = result

                    # Save transcript
                    transcript_path = Path(filename).with_suffix('.txt')
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(result['text'])

                    # Display transcript preview
                    with status_area.expander(f"📝 Preview: {filename}", expanded=False):
                        st.text_area("Transcription:", result['text'], height=200, disabled=True)

                progress_bar.progress(completed / len(files_to_process))
                status_text.text(f"Completed: {completed}/{len(files_to_process)}")
            except Exception as e:
                status_area.error(f"Error processing {file}: {str(e)}")

        if completed == len(files_to_process):
            status_area.success(f"✅ Successfully processed {completed} files")
        return results

    except Exception as e:
        status_area.error(f"Processing error: {str(e)}")
        return {}

def parallel_transcribe_worker(file_path: Path, model_name: str, language: str, status_funcs: dict) -> tuple:
    """Worker function for parallel processing using TranscriptionManager"""
    try:
        # Create manager instance for this worker
        manager = TranscriptionManager(model_name=model_name, language=language)
        
        # Process file
        result = manager.transcribe(file_path)
        if result:
            status_funcs["success"](f"✓ Processed: {file_path.name}")
            return True, file_path.name, result
            
        return False, file_path.name, None
    except Exception as e:
        status_funcs["error"](f"Error processing {file_path.name}: {str(e)}")
        return False, file_path.name, None

def show_transcription_tab():
    """Display the transcription interface with enhanced UI and monitoring"""
    st.title("Fast Video & Audio Transcription")

    with st.sidebar:
        if 'workflow' not in st.session_state:
            st.header("📝 Transcription Settings")
            
            # Language and model selection
            col1, col2 = st.columns(2)
            with col1:
                language = st.selectbox("Language", ["de", "en"], index=0)
            with col2:
                mixed_precision = st.checkbox("Mixed Precision", value=True,
                                         help="Reduces memory usage with minimal quality impact")
            
            # Model selection based on language
            model_options = {
                "de": [
                    "openai/whisper-large-v3",
                    "openai/whisper-large-v2",
                    "openai/whisper-large-v3-turbo",
                    "distil-whisper/distil-large-v2"
                ],
                "en": [
                    "distil-whisper/distil-large-v3",
                    "openai/whisper-large-v3",
                    "distil-whisper/distil-large-v2",
                    "distil-whisper/distil-medium.en"
                ]
            }
            
            model_name = st.selectbox(
                "Model",
                model_options[language],
                index=0,
                help="Select model based on your needs (larger models = better quality but slower)"
            )

            # Get optimized settings based on hardware
            settings = get_whisper_settings(language)
            pipeline_settings = settings["pipeline_settings"]

            # Performance settings
            st.subheader("⚡ Performance Settings")
            with st.expander("Advanced Settings", expanded=False):
                # RTX 4090 Preset Selection
                st.markdown("### 🎯 RTX 4090 Performance Presets")
                preset = st.selectbox(
                    "Performance Preset",
                    ["🛡️ STABIL", "⚡ OPTIMIERT (Empfohlen)", "🚀 SCHNELL (Experimentell)", "🔥 MAX. SCHNELL (Riskant)"],
                    index=1,  # Default: OPTIMIERT
                    help="Wählen Sie eine Performance-Einstellung für Ihre RTX 4090"
                )

                # Apply preset values
                if preset == "🛡️ STABIL":
                    batch_val, chunk_val, stride_val, workers_val = 8, 20, 2, 1
                    help_text = "✨ Maximale Stabilität - Keine Crashes"
                elif preset == "⚡ OPTIMIERT (Empfohlen)":
                    batch_val, chunk_val, stride_val, workers_val = 16, 30, 3, 2
                    help_text = "🎯 Beste Balance - Schnell und zuverlässig"
                elif preset == "🚀 SCHNELL (Experimentell)":
                    batch_val, chunk_val, stride_val, workers_val = 32, 45, 5, 4
                    help_text = "⚡ Höhere Geschwindigkeit - Kann crashen"
                else:  # MAX. SCHNELL
                    batch_val, chunk_val, stride_val, workers_val = 56, 60, 8, 6
                    help_text = "🔥 Maximale Performance - Hoher Crash-Risk"

                st.info(help_text)
                st.markdown("---")

                # Manual override
                st.markdown("### 🔧 Manuelles Anpassen")
                batch_size = st.slider(
                    "Batch Size", 1, 64, batch_val,
                    help=f"**RTX 4090 Empfehlung:** {batch_val}\n\nGrößere Batch Size = Schneller, aber mehr RAM/GPU-Speicher"
                )
                chunk_length = st.slider(
                    "Chunk Length (Seconds)", 10, 90, chunk_val,
                    help=f"**RTX 4090 Empfehlung:** {chunk_val}s\n\nLängere Chunks = Besserer Kontext, aber mehr Speicher"
                )
                stride_length = st.slider(
                    "Stride Length (Seconds)", 1, 10, stride_val,
                    help=f"**RTX 4090 Empfehlung:** {stride_val}s\n\nÜberlappung zwischen Chunks für bessere Übergänge"
                )

                st.checkbox("Enable dynamic optimization", value=True,
                          help="Automatically adjust settings based on GPU state")

            # File handling options
            st.subheader("📁 File Options")
            auto_save = st.checkbox("Auto-save transcripts", value=True)
            skip_existing = st.checkbox("Skip existing files", value=True)

            # GPU Status
            if gpu.has_gpu:
                st.subheader("💾 GPU Status")
                try:
                    metrics = gpu.get_info()
                    if metrics:
                        col1, col2 = st.columns(2)
                        
                        # Memory usage in first column
                        with col1:
                            if 'total_memory' in metrics and 'allocated_memory' in metrics:
                                mem_ratio = min(metrics['allocated_memory'] / metrics['total_memory'], 1.0)
                                st.progress(mem_ratio,
                                          text=f"{metrics['allocated_memory']:.1f}GB / {metrics['total_memory']:.1f}GB")
                        
                        # Temperature and utilization in second column
                        with col2:
                            status = []
                            if 'temperature' in metrics:
                                status.append(f"🌡️ {metrics['temperature']}°C")
                            if 'utilization' in metrics:
                                status.append(f"⚡ {metrics['utilization']}%")
                            if status:
                                st.caption(" | ".join(status))
                        
                        # GPU info footer
                        st.caption(f"CUDA {metrics['cuda_version']} | PyTorch {metrics['torch_version']}")
                    else:
                        st.info("GPU metrics not available")
                except Exception as e:
                    logger.warning(f"GPU status error: {e}")
                    st.warning("Unable to display GPU status")

    # File selection interface
    st.subheader("📂 Select Files")
    folder_paths = st.text_area(
        "Enter folder paths (one per line)",
        placeholder="e.g.:\nC:\\Videos\nD:\\Audio",
        help="Multiple paths can be separated by line breaks"
    )

    # Action buttons with status
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col2:
        start_transcription = st.button("🚀 Start", use_container_width=True)
    with col3:
        refresh = st.button("🔄 Refresh", use_container_width=True)
    with col4:
        clear = st.button("🗑️ Clear", use_container_width=True)

    # Processing status and results
    status_container = st.container()
    progress_container = st.container()
    results_container = st.container()

    if folder_paths and (start_transcription or refresh):
        with progress_container:
            paths = [Path(p.strip()) for p in folder_paths.split('\n') if p.strip()]
            valid_paths = []
            files_to_process = []

            for path in paths:
                if path.exists() and path.is_dir():
                    valid_paths.append(path)
                    if skip_existing:
                        existing = set(f.stem for f in path.glob("*.txt"))
                        new_files = [
                            f for f in path.glob("*")
                            if f.suffix.lower() in ('.mp3', '.mp4')
                            and f.stem not in existing
                        ]
                    else:
                        new_files = [
                            f for f in path.glob("*")
                            if f.suffix.lower() in ('.mp3', '.mp4')
                        ]
                    files_to_process.extend(new_files)

            if files_to_process:
                st.info(f"📁 Found {len(files_to_process)} files to process")
                
                if start_transcription:
                    process_files(
                        files_to_process,
                        model_name,
                        language,
                        skip_existing=skip_existing
                    )
            else:
                st.warning("No files found to process")

# GPU status is handled by GPUManager instance
def main():
    """Main application entry point"""
    # Sidebar for workflow
    if 'workflow' not in st.session_state:
        show_transcription_tab()
    else:
        st.error("Workflow functionality will be implemented in a future update")

if __name__ == "__main__":
    main()
