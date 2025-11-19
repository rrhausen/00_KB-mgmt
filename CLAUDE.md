# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based video transcription and knowledge base management system that downloads YouTube content, transcribes it using GPU-optimized Whisper models, and uploads to ODIN AI Knowledge Base. The application is primarily designed for German-language content management.

## Architecture

The application follows a modular tab-based architecture:

- **app.py** - Main entry point with tab navigation and session state management
- **app0.py** - YouTube download functionality with API integration and metadata fetching
- **app1.py** - GPU-optimized transcription using Whisper models with Flash Attention
- **app2.py** - ODIN AI Knowledge Base upload with multi-format support

Each module is designed to operate independently through `show_*_tab()` functions while sharing session state for transcribed files data.

## Development Environment Setup

### Conda Environment
The project uses a specific conda environment defined in `knowledge_env_backup.yml` with CUDA 12.4 support:

```bash
conda env create -f knowledge_env_backup.yml
conda activate knowledge_env
```

### Critical Dependencies
- **PyTorch with CUDA 12.4**: Must be installed via conda environment for GPU optimization
- **Flash Attention 2**: Required for memory-efficient GPU transcription
- **Whisper models**: Large v3/v2 models for transcription
- **Streamlit 1.40+**: Web interface framework

### Environment Configuration
Create `.env` file with:
```
API_KEY=your_odin_api_key
API_SECRET=your_odin_api_secret
```

YouTube API keys should be placed in `YT_API_keys.txt` (one per line).

## Running the Application

```bash
# Start main application
streamlit run app.py

# Verify GPU optimization before running
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Flash Attention enabled:', torch.backends.cuda.flash_sdp_enabled())"
```

## GPU Optimization Configuration

The transcription module (app1.py) is heavily optimized for RTX 3090 (24GB VRAM):

- **Environment variables** are pre-set for optimal GPU memory allocation
- **Recommended settings**: Batch size 16, chunk length 30 seconds
- **Memory management**: Automatic cache clearing and efficient batch processing
- **Flash Attention**: Must be installed for optimal performance

## Key Integration Points

### YouTube API Integration
- Multi-key rotation system in `YouTubeAPI` class
- Automatic quota management and error handling
- Metadata fetching for enhanced transcription context

### ODIN AI Integration
- Direct file upload via REST API in `app2.py`
- Multi-format support (TXT, PDF, MP4, DOCX, HTML, JSON, XML, CSV, MP3, MD)
- Configurable metadata and batch upload capabilities

### Session State Management
- `transcribed_files` list persists across tabs
- Automatic file detection between transcription and upload phases
- Progress tracking for long-running operations

## File Structure Patterns

- Source code files use numerical naming (app0.py, app1.py, app2.py) for tab order
- Configuration files are excluded from git (.env, API keys)
- Documentation in both German (README.md) and English
- Temporary files are managed through Python's tempfile module

## Common Development Commands

```bash
# Install dependencies after conda environment setup
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Test GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check Flash Attention installation
python -c "import flash_attn; print('Flash Attention 2 installed')"

# Run with specific port for development
streamlit run app.py --server.port 8501
```

## Error Handling Patterns

- GPU memory errors trigger automatic cache clearing
- Network failures include retry logic with exponential backoff
- API quota issues trigger key rotation in YouTube module
- File processing errors are logged but don't stop batch operations