# RTX 4090 Video Transcription & Knowledge Base System

🚀 **High-performance Streamlit application** for video/audio transcription using RTX 4090 GPU acceleration and ODIN AI Knowledge Base integration.

## 🎯 Features

### 🎥 **YouTube Download** (Tab 1)
- **Single video & channel downloads** with yt-dlp
- **Multiple format support**: MP3, MP4, WebM
- **Metadata extraction** with YouTube API integration
- **Batch processing** for entire channels

### 🎙️ **GPU-Accelerated Transcription** (Tab 2)
- **RTX 4090 optimized** with CUDA 12.9 support
- **Multiple Whisper models**: Large-v3, Large-v2, Turbo
- **Real-time progress monitoring** with ETA
- **Chunk-level timestamps**: `[MM:SS - MM:SS]` format
- **Performance presets**: Stabil, Optimized, Fast, Max Speed

### 📚 **Knowledge Base Upload** (Tab 3)
- **Auto-detection** of transcribed files
- **Multiple formats**: TXT, PDF, MP4, DOCX, HTML, JSON, XML, CSV, MP3, MD
- **Batch upload** with configurable metadata
- **Direct ODIN AI API integration**

## ⚡ Performance & Hardware

### **RTX 4090 Optimizations**
- **CUDA 12.9** with PyTorch 2.8.0
- **Memory-efficient attention** (SDPA)
- **Dynamic batch processing** with GPU monitoring
- **Automatic fallback** to CPU if needed

### **Performance Presets**
| Preset | Batch Size | Chunks | Workers | Speed | Stability |
|--------|------------|---------|---------|-------|------------|
| 🛡️ STABIL | 8 | 20s | 1 | Slow | ✅ Maximum |
| ⚡ **OPTIMIERT** | **16** | **30s** | **2** | **Fast** | ✅ High |
| 🚀 SCHNELL | 32 | 45s | 4 | Very Fast | ⚠️ Medium |
| 🔥 MAX. SCHNELL | 56 | 60s | 6 | Ultra Fast | ❌ Risky |

## 🛠️ Installation

### **Prerequisites**
- **NVIDIA RTX 4090** (24GB VRAM recommended)
- **CUDA 12.9** compatible drivers
- **Windows 10/11** with 16GB+ RAM
- **Python 3.11+**

### **Step 1: Setup Conda Environment**
```bash
# Create conda environment
conda create -n knowledge_env_cuda128 python=3.11
conda activate knowledge_env_cuda128

# Install system dependencies
conda install -c conda-forge ninja
```

### **Step 2: Install PyTorch with CUDA 12.9**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### **Step 3: Install Python Dependencies**
```bash
pip install -r requirements.txt
pip install numpy==2.1.3
pip install transformers
pip install streamlit
pip install python-dotenv
pip install packaging
pip install pyyaml
pip install requests
pip install tqdm
pip install regex
pip install einops
pip install filelock
pip install sympy
pip install protobuf
```

### **Step 4: Verify Installation**
```bash
# Check GPU availability
python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python -c "import torch; print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')"
```

### **Step 5: Configure Environment**
Create `.env` file in project root:
```env
API_KEY=your_odin_api_key
API_SECRET=your_odin_api_secret
```

Create `YT_API_keys.txt` for YouTube API:
```
your_youtube_api_key_1
your_youtube_api_key_2
```

### **Step 6: Launch Application**
```bash
conda activate knowledge_env_cuda128
streamlit run app.py
```

## 🎮 Usage

### **YouTube Download**
1. Navigate to **"YouTube Download"** tab
2. Enter video URL or channel ID
3. Select download format (MP3/MP4)
4. Click **Download**

### **Transcription**
1. Navigate to **"Transcription"** tab
2. Select folder containing media files
3. Choose language (DE/EN) and model
4. Select performance preset (recommended: ⚡ OPTIMIERT)
5. Click **🚀 Start**

### **Knowledge Base Upload**
1. Navigate to **"Knowledge Base Upload"** tab
2. Select folder with transcribed files
3. Configure upload metadata
4. Click **Upload to ODIN AI**

## 📊 System Requirements

### **Minimum Requirements**
- **GPU**: NVIDIA RTX 3060+ (6GB VRAM)
- **RAM**: 16GB system memory
- **Storage**: 2GB free space
- **OS**: Windows 10/11

### **Recommended Requirements**
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 32GB+ system memory
- **Storage**: 10GB+ free space
- **OS**: Windows 11

## 🔧 Configuration

### **Environment Variables**
```env
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
TOKENIZERS_PARALLELISM=false
CUDA_LAUNCH_BLOCKING=1
NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
```

### **GPU Optimizations**
- **TF32 enabled** for faster computation
- **SDPA attention** for memory efficiency
- **Dynamic batch sizing** based on VRAM
- **Automatic memory cleanup**

## 📁 Project Structure

```
00_KB-mgmt/
├── app.py                 # Main application with tab navigation
├── app0.py                # YouTube download module
├── app1.py                # Transcription module with RTX 4090 optimization
├── app2.py                # ODIN AI upload module
├── requirements.txt        # Python dependencies
├── .env                   # API credentials
├── YT_API_keys.txt        # YouTube API keys
├── RTX_4090_Settings.md   # Performance presets documentation
├── CLAUDE.md              # Development guide
└── README.md              # This file
```

## 🚨 Troubleshooting

### **Common Issues**

**"CUDA not available"**
```bash
# Check NVIDIA drivers
nvidia-smi
# Update PyTorch
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

**"Memory errors"**
- Use **🛡️ STABIL** preset
- Reduce batch size
- Close other GPU applications

**"Model initialization failed"**
- Check GPU memory availability
- Restart application
- Verify CUDA installation

### **Performance Tips**
- Use **⚡ OPTIMIERT** preset for best balance
- Close unnecessary browser tabs
- Monitor GPU temperature with `nvidia-smi`

## 📈 Performance Metrics

### **Transcription Speed**
- **RTX 4090**: ~2-5x faster than CPU
- **Batch processing**: Handles multiple files efficiently
- **Real-time progress**: Live status updates

### **Memory Usage**
- **GPU VRAM**: 8-16GB depending on settings
- **System RAM**: 4-8GB during processing
- **Temp files**: Automatically cleaned up

## 🤝 Support

For issues and feature requests:
1. Check this README for solutions
2. Verify system requirements
3. Review GPU driver compatibility
4. Check application logs for error details

---

**Version**: 2.1
**Last Updated**: 2025-10-27
**Compatible**: RTX 4090 + CUDA 12.9
**License**: MIT