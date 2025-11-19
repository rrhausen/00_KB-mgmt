# Technical Context

## Core Technologies

### Python Libraries
1. **Streamlit**
   - Version: 1.29.0+
   - Purpose: Web interface and user interaction
   - Key features: Progress bars, file upload, sidebar controls, tabbed interface

2. **Transformers (Hugging Face)**
   - Models: Whisper variants (large-v3, large-v2, turbo, etc.)
   - Purpose: Speech-to-text transcription with GPU acceleration
   - Features: SDPA attention, batch processing, timestamp generation

3. **PyTorch**
   - Version: 2.8.0 with CUDA 12.9
   - Purpose: ML model backend with RTX 4090 optimization
   - Features: CUDA 12.9 support, TF32 precision, SDPA attention
   - Optimizations: Memory-efficient attention, automatic mixed precision

4. **FFmpeg**
   - Purpose: Audio extraction and processing
   - Features: Audio normalization, format conversion
   - Format support: MP3, MP4, WebM

5. **Additional Dependencies**
   - yt-dlp: YouTube video downloading
   - moviepy: Video processing
   - numpy: Array operations (2.1.3)
   - python-dotenv: Environment variable management

### External Services
1. **YouTube Data API v3**
   - Purpose: Video metadata and content access
   - Authentication: Multi-key rotation system
   - Quota management: Automatic key switching on quota limits
   - API keys stored in: YT_API_keys.txt

2. **ODIN AI**
   - Purpose: Knowledge base storage and retrieval
   - Integration: Direct API connection
   - Default Project ID: vc0lgCl8YHKTHQ9ByzYt
   - Authentication: API_KEY and API_SECRET from .env

## Development Setup

### Environment Requirements
1. **Python**: 3.11+ (knowledge_env_cuda128 conda environment)
2. **GPU**: NVIDIA RTX 4090 with CUDA 12.9
3. **System**: Windows 10/11 with 16GB+ RAM
4. **Storage**: 2GB+ free space for media files
5. **Software**: FFmpeg, CUDA drivers, conda

### GPU Configuration
- **Target Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **CUDA Version**: 12.9
- **PyTorch**: Built with CUDA 12.9 support
- **Memory Management**: Automatic VRAM monitoring and cleanup
- **Performance Features**: TF32 math, SDPA attention, memory-efficient processing

### Performance Presets (RTX 4090)
1. **🛡️ STABIL**: Batch size 8, chunks 20s, 1 worker
2. **⚡ OPTIMIERT**: Batch size 16, chunks 30s, 2 workers (default)
3. **🚀 SCHNELL**: Batch size 32, chunks 45s, 4 workers
4. **🔥 MAX. SCHNELL**: Batch size 56, chunks 60s, 6 workers

### File System Requirements
- **Base Directory**: C:\Users\rrhau\Documents\pycode\00_KB-mgmt\
- **Naming Convention**: YYYY-MM-DD_Title_videoId.ext
- **Metadata Storage**: JSON format with video details
- **Transcripts**: TXT format with chunk-level timestamps

## Technical Architecture

### GPU Management System
```python
# Custom GPUManager class handles:
- RTX 4090 specific optimizations
- CUDA 12.9 configuration
- VRAM monitoring and cleanup
- TF32 and SDPA attention setup
- Memory-efficient processing
- Automatic fallback to CPU
```

### Transcription Pipeline
```python
# Processing flow:
1. GPU initialization with RTX 4090 settings
2. Whisper model loading with GPU optimization
3. Audio chunking with configurable parameters
4. Batch processing with performance presets
5. Timestamp generation and formatting
6. Memory cleanup and error recovery
```

## Current Implementation Status

### ✅ **Completed Features**
- **RTX 4090 Optimization**: Full CUDA 12.9 integration with custom GPUManager
- **Performance Presets**: All 4 presets implemented with UI controls
- **Timestamp Processing**: Chunk-level timestamps with formatted output
- **Memory Management**: VRAM monitoring, automatic cleanup, error recovery
- **Error Handling**: Custom exceptions (GPUError, TranscriptionError, etc.)
- **Sequential Processing**: Fixed threading conflicts in Streamlit context
- **API Integration**: YouTube API with key rotation, ODIN AI upload
- **Web Interface**: Streamlit tabs, progress indicators, settings management

### 🔧 **Technical Specifications**
- **GPU Utilization**: 80-95% target with automatic throttling
- **Memory Efficiency**: >90% VRAM utilization with 20-22GB max usage
- **Processing Speed**: 2-5x faster than CPU transcription
- **Batch Sizes**: 8-56 depending on performance preset
- **Chunk Lengths**: 20-60 seconds with configurable overlap
- **Worker Threads**: 1-6 parallel workers

## Technical Constraints & Solutions

### Memory Management
- **Constraint**: VRAM limitations with large models
- **Solution**: Dynamic batch sizing, automatic cleanup, sequential processing
- **Monitoring**: Real-time VRAM tracking with peak usage stats

### Threading Limitations
- **Constraint**: Streamlit context conflicts with parallel processing
- **Solution**: Sequential processing with progress tracking
- **Benefit**: Eliminated "missing ScriptRunContext" errors

### API Rate Limiting
- **Constraint**: YouTube API quota limits
- **Solution**: Multi-key rotation system with automatic switching
- **Recovery**: Graceful handling of quota exceeded errors

### GPU Compatibility
- **Constraint**: CUDA version mismatches
- **Solution**: PyTorch installation with specific CUDA 12.9 build
- **Verification**: GPU availability checks and fallback mechanisms

## Environment Variables Configuration
```env
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
TOKENIZERS_PARALLELISM=false
CUDA_LAUNCH_BLOCKING=1
NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
```

## Error Handling Architecture
- **GPU Errors**: VRAM overflow detection, automatic fallback mechanisms
- **API Errors**: Retry logic, key rotation, graceful degradation
- **File Errors**: Permission checks, path validation, cleanup procedures
- **Memory Errors**: Prevention, detection, recovery with automatic cleanup