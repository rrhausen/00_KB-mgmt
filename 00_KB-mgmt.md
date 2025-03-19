# app.py

```python
import streamlit as st
from app1 import show_transcription_tab
from app2 import show_upload_tab
from dotenv import load_dotenv

# Lade .env Datei
load_dotenv()

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Transkription und KB Upload",
    page_icon="🎯",
    layout="wide"
)

def main():
    st.title("Transkription und Knowledge Base Upload")
    
    # Initialisiere session_state für transkribierte Dateien wenn nicht vorhanden
    if 'transcribed_files' not in st.session_state:
        st.session_state.transcribed_files = []
    
    # Tab-Auswahl
    tab1, tab2 = st.tabs(["Transkription", "Knowledge Base Upload"])
    
    with tab1:
        show_transcription_tab()
    
    with tab2:
        show_upload_tab()

if __name__ == "__main__":
    main()
```

# app1.py

```python
import warnings
import streamlit as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from pathlib import Path
import tempfile
import os
import subprocess
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The attention mask is not set')
warnings.filterwarnings('ignore', message='.*forced_decoder_ids.*')

def normalize_path(path):
    if not path:
        return path
    
    try:
        # Konvertiere Windows-Pfade in raw string format
        path = str(path).strip()
        
        # Behandle UNC Pfade
        if path.startswith('\\\\'):
            return Path(path)
        
        # Behandle Netzwerklaufwerke
        if ':' in path:
            # Versuche den Pfad in einen absoluten Windows-Pfad umzuwandeln
            abs_path = os.path.abspath(path)
            # Versuche den Pfad in einen UNC-Pfad umzuwandeln
            if abs_path.startswith('\\\\'):
                return Path(abs_path)
            try:
                # Versuche den Pfad aufzulösen
                resolved_path = Path(path).resolve()
                if os.path.exists(resolved_path):
                    return resolved_path
            except:
                pass
            # Wenn alles andere fehlschlägt, verwende den ursprünglichen Pfad
            return Path(path)
            
        return Path(path)
    except Exception as e:
        st.error(f"Fehler bei der Pfadnormalisierung: {str(e)}")
        return Path(path)

def optimize_gpu_settings():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()

def check_gpu_and_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        optimize_gpu_settings()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.success(f"GPU gefunden: {gpu_name} ({gpu_memory:.2f} GB)")
        cuda_version = torch.version.cuda
        st.info(f"CUDA Version: {cuda_version}")
    else:
        st.error("Keine GPU gefunden - Transkription läuft auf CPU")
    return device

def extract_audio(video_path):
    if str(video_path).lower().endswith('.mp3'):
        return str(video_path)
    output_path = tempfile.mktemp(suffix='.mp3')
    command = [
        'ffmpeg', '-i', str(video_path),
        '-q:a', '0', '-map', 'a',
        '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5',
        output_path
    ]
    subprocess.run(command, capture_output=True)
    return output_path

def transcribe_audio(audio_path, model_name, language, batch_size, chunk_length):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Explizit fp16
            device_map="auto",
            use_safetensors=True,
            attn_implementation="flash_attention_2"  # Flash Attention 2
        )
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,  # Konsistent fp16
        )
        
        transcribe_params = {
            "batch_size": batch_size,
            "chunk_length_s": chunk_length,
            "return_timestamps": True,
            "stride_length_s": chunk_length // 6,
            "generate_kwargs": {
                "language": language,
                "task": "transcribe",
                "no_repeat_ngram_size": 3,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4
            }
        }
        return pipe(audio_path, **transcribe_params)
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {str(e)}")
        raise

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
            return f'[{start_time} - {end_time}]\n{chunk["text"]}\n\n'
        return f'[Zeitstempel nicht verfügbar]\n{chunk["text"]}\n\n'
    except (TypeError, KeyError, AttributeError):
        return f'{chunk.get("text", "")}\n\n'

def get_output_filename(original_filename):
    base_name = os.path.splitext(original_filename)[0]
    return f"{base_name}.txt"

def save_transcription(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def process_files(files_to_process, model_name, language, batch_size, chunk_length, auto_save=False, output_dir=None, skip_existing=True):
    results = {}
    total_files = len(files_to_process)
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    for idx, file in enumerate(files_to_process):
        try:
            if isinstance(file, str):
                file_path = file
                file_name = os.path.basename(file)
            else:
                file_path = save_uploaded_file(file)
                file_name = file.name

            output_filename = get_output_filename(file_name)
            output_path = os.path.join(output_dir, output_filename) if output_dir else None

            if skip_existing and output_path and os.path.exists(output_path):
                st.info(f"Überspringe vorhandene Datei: {file_name}")
                continue

            status_text.text(f"Verarbeite Datei {idx+1} von {total_files}: {file_name}")
            audio_path = extract_audio(file_path)
            
            result = transcribe_audio(audio_path, model_name, language, batch_size, chunk_length)
            results[file_name] = result

            if auto_save and output_dir:
                full_text = "".join(format_chunk_with_timestamp(chunk) for chunk in result["chunks"])
                save_transcription(full_text, output_path)
                if 'transcribed_files' not in st.session_state:
                    st.session_state.transcribed_files = []
                st.session_state.transcribed_files.append(output_path)
                st.success(f"Gespeichert: {output_path}")

            progress = (idx + 1) / total_files
            progress_bar.progress(progress)
            
            elapsed_time = time.time() - start_time
            files_per_second = (idx + 1) / elapsed_time if elapsed_time > 0 else 0
            remaining_files = total_files - (idx + 1)
            eta = remaining_files / files_per_second if files_per_second > 0 else 0
            
            status_text.text(f"Fortschritt: {idx+1}/{total_files} | "
                           f"Geschwindigkeit: {files_per_second:.2f} Dateien/s | "
                           f"ETA: {eta:.1f}s")

        except Exception as e:
            st.error(f"Fehler bei {file_name}: {str(e)}")

    status_text.text("Transkription abgeschlossen!")
    return results

def show_transcription_tab():
    st.title("Schnelle Video & Audio Transkription")
    device = check_gpu_and_model()

    with st.sidebar:
        st.header("Transkriptions-Einstellungen")
        model_name = st.selectbox(
            "Modell",
            [
                "openai/whisper-large-v3",
                "openai/whisper-large-v2",
                "openai/whisper-large",
                "openai/whisper-large-v3-turbo",
                "openai/whisper-turbo",
                "distil-whisper/distil-large-v2",
                "distil-whisper/distil-medium.en",
                "distil-whisper/distil-small.en",
                "distil-whisper/distil-large-v3"
            ],
            help="Größere Modelle sind genauer, aber langsamer"
        )

        language = st.selectbox(
            "Sprache",
            ["de", "en"],
            help="Sprache der Aufnahme"
        )

        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=24,
            step=1,
            help="Größere Batches = schneller, aber mehr VRAM-Nutzung"
        )
        st.markdown("↑ Empfohlener Wert: 24")

        chunk_length = st.slider(
            "Chunk Länge (Sekunden)",
            min_value=10,
            max_value=60,
            value=30,
            step=1,
            help="Länge der Audio-Chunks für Verarbeitung"
        )
        st.markdown("↑ Empfohlener Wert: 30")

        auto_save = st.checkbox("Automatisch als TXT speichern", value=True)
        skip_existing = st.checkbox("Vorhandene ausnehmen", value=True)
        
        upload_type = st.radio(
            "Auswahl",
            ["Ordner", "Einzelne Datei"],
            horizontal=True,
            index=0
        )

    files_to_process = []
    output_dir = None

    if upload_type == "Einzelne Datei":
        uploaded_file = st.file_uploader(
            "Video- oder Audiodatei hochladen",
            type=["mp4", "mpeg4", "mp3"],
            help="Maximale Dateigröße: 2GB"
        )

        if uploaded_file:
            files_to_process = [uploaded_file]
            folder_path = st.text_input(
                "Speicherort für TXT-Datei",
                placeholder="z.B. V:\\Optionen\\26_OptionsUniversum\\31_Ausbildung_OptHändler_2024",
                help="Vollständigen Pfad eingeben oder per Copy/Paste aus dem Explorer einfügen"
            )
            if folder_path:
                normalized_path = normalize_path(folder_path)
                try:
                    if os.path.exists(normalized_path):
                        output_dir = normalized_path
                        st.success(f"Speicherort: {output_dir}")
                    else:
                        st.error("Ordner nicht gefunden. Bitte überprüfen Sie den Pfad.")
                except Exception as e:
                    st.error(f"Fehler beim Zugriff auf den Pfad: {str(e)}")
    else:
        folder_path = st.text_input(
            "Ordnerpfad eingeben",
            placeholder="z.B. V:\\Optionen\\26_OptionsUniversum\\31_Ausbildung_OptHändler_2024",
            help="Vollständigen Pfad eingeben oder per Copy/Paste aus dem Explorer einfügen"
        )

        if folder_path:
            normalized_path = normalize_path(folder_path)
            try:
                if os.path.exists(normalized_path):
                    st.success(f"Ordner gefunden: {folder_path}")
                    available_files = [f for f in os.listdir(normalized_path) 
                                     if f.lower().endswith(('.mp4', '.mpeg4', '.mp3'))]
                    
                    if available_files:
                        file_status = {}
                        for file in available_files:
                            txt_file = os.path.splitext(file)[0] + ".txt"
                            txt_path = os.path.join(normalized_path, txt_file)
                            file_status[file] = os.path.exists(txt_path)

                        selected_files = st.multiselect(
                            "Dateien zum Transkribieren auswählen",
                            available_files,
                            default=[f for f in available_files if not (skip_existing and file_status[f])],
                            format_func=lambda x: f"{'🟢 ' if file_status[x] else ''}{x}"
                        )

                        files_to_process = [os.path.join(normalized_path, f) for f in selected_files]
                        output_dir = normalized_path
                        if selected_files:
                            st.info(f"Ausgewählte Dateien: {len(selected_files)}")
                    else:
                        st.warning("Keine unterstützten Mediendateien im ausgewählten Ordner gefunden.")
                else:
                    st.error("Ordner nicht gefunden. Bitte überprüfen Sie den Pfad.")
            except Exception as e:
                st.error(f"Fehler beim Zugriff auf den Ordner: {str(e)}")

    if files_to_process:
        if st.button("Transkription starten"):
            results = process_files(files_to_process, model_name, language, batch_size, chunk_length, auto_save, output_dir, skip_existing)
            if results:
                st.success(f"Transkription von {len(results)} Datei(en) abgeschlossen.")
```

# app2.py

```python
import streamlit as st
import os
from pathlib import Path
import requests
import datetime
from dotenv import load_dotenv

# Lade .env Datei und API-Credentials

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

def upload_file(file_path, metadata):
    url = "https://api.getodin.ai/v2/project/knowledge/add/file"
    files = {
        "file": (file_path.name, open(file_path, "rb"), 
                "text/plain" if file_path.suffix == '.txt' else "application/octet-stream")
    }
    headers = {
        "accept": "application/json",
        "X-API-KEY": api_key,
        "X-API-SECRET": api_secret
    }
    response = requests.post(url, data=metadata, files=files, headers=headers)
    return response

def show_upload_tab():
    st.header("Knowledge Base Upload")
    
    # Status der transkribierten Dateien anzeigen
    if 'transcribed_files' in st.session_state and st.session_state.transcribed_files:
        st.info(f"Verfügbare transkribierte Dateien: {len(st.session_state.transcribed_files)}")
        
        # Option zum Löschen der Session-Daten
        if st.button("Transkribierte Dateien zurücksetzen"):
            st.session_state.transcribed_files = []
            st.experimental_rerun()
        
        use_transcribed = st.checkbox('Transkribierte Dateien verwenden', value=True)
        if use_transcribed:
            files_to_process = st.session_state.transcribed_files
        else:
            files_to_process = []
    else:
        use_transcribed = False
        files_to_process = []
        st.info("Keine transkribierten Dateien verfügbar")
    
    if not use_transcribed:
        folder_path = st.text_input(
            "Ordnerpfad eingeben",
            placeholder="z.B. V:\\Dokumente\\Wissen",
            help="Vollständigen Pfad eingeben oder per Copy/Paste aus dem Explorer einfügen"
        )
        
        file_types = st.multiselect(
            'Dateitypen',
            ['txt', 'pdf', 'mp4', 'docx', 'html', 'json', 'xml', 'csv', 'mp3', 'md'],
            default=['txt']
        )
        
        if folder_path and os.path.exists(folder_path):
            files_to_process = []
            for ext in file_types:
                files_to_process.extend(list(Path(folder_path).rglob(f'*.{ext}')))
    
    # Upload-Metadaten
    col1, col2 = st.columns(2)
    with col1:
        project_id = st.text_input('Project ID', value='vc0lgCl8YHKTHQ9ByzYt')
        author_name = st.text_input('Author Name', value='')
        uploaded_by = st.text_input('Uploaded by', value='trendguru@email.de')
    with col2:
        added_by = st.text_input('Added by', value='trendguru@email.de')
        category = st.text_input('Category', value='null')
        last_upload = st.date_input('Last Upload', value=datetime.date.today())
    
    if st.button('Upload starten') and files_to_process:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(files_to_process)
        successful_uploads = 0
        failed_uploads = 0
        
        for idx, file_path in enumerate(files_to_process):
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            status_text.text(f"Verarbeite Datei {idx+1} von {total_files}: {file_path.name}")
            
            if file_path.stat().st_size <= 1 * 1024 * 1024 * 1024:  # 1 GB
                try:
                    metadata = {
                        "project_id": project_id,
                        "metadata": f'{{"author_name": "{author_name}", "uploaded_by": "{uploaded_by}", "added_by": "{added_by}", "category": "{category}", "last_upload": "{last_upload}"}}'
                    }
                    
                    response = upload_file(file_path, metadata)
                    
                    if response.status_code == 200:
                        successful_uploads += 1
                        st.success(f"Erfolgreich hochgeladen: {file_path.name}")
                    else:
                        failed_uploads += 1
                        st.error(f"Fehler beim Upload von {file_path.name}: {response.text}")
                
                except Exception as e:
                    failed_uploads += 1
                    st.error(f"Fehler bei {file_path.name}: {str(e)}")
            else:
                failed_uploads += 1
                st.warning(f"Datei zu groß (>1GB): {file_path.name}")
            
            progress_bar.progress((idx + 1) / total_files)
        
        status_text.text("Upload abgeschlossen!")
        st.info(f"Upload-Statistik: {successful_uploads} erfolgreich, {failed_uploads} fehlgeschlagen")
```

# README.md

```markdown
# Video Transkription und Knowledge Base Upload

Eine Streamlit-Anwendung zur Transkription von Video/Audio-Dateien und Upload in die ODIN AI Knowledge Base.

## Struktur

Die Anwendung ist in drei Hauptdateien aufgeteilt:

- `app.py` - Hauptanwendung mit Tab-Steuerung
- `app1.py` - Transkriptionsmodul mit GPU-Optimierung
- `app2.py` - ODIN AI Upload-Modul
- `requirements.txt` - Abhängigkeiten

## Hauptfunktionen

### 1. Transkription

- **Video/Audio Upload** (Einzeldatei oder Ordner)
- **Konfigurierbare Parameter**:
  - Modellauswahl (Whisper Large v3, v2, etc.)
  - Batch Size (optimiert für RTX 3090)
  - Chunk Length
  - Sprache (DE/EN)
- **GPU-Optimierte Verarbeitung**
- **Fortschrittsanzeige mit ETA**
- **Zeitstempel** im Format `[MM:SS]` oder `[HH:MM:SS]`

### 2. Knowledge Base Upload

- **Automatische Übernahme** transkribierter Dateien
- **Alternative manuelle Ordnerauswahl**
- **Unterstützte Dateiformate**: TXT, PDF, MP4, DOCX, HTML, JSON, XML, CSV, MP3, MD
- **Konfigurierbare Upload-Metadaten**

## Besondere Features

- **GPU-Optimierungen** für maximale Performance
- **Session State Management** für große Dateimengen
- **Intelligente Audioextraktion** mit Normalisierung
- **Überspringen existierender Transkripte**
- **Detaillierte Fortschrittsanzeige** mit Geschwindigkeitsmetrik
- **Umfassende Fehlerbehandlung**

## Technische Optimierungen

- **Native PyTorch SDPA** für optimale GPU-Nutzung
- **Effizientes Memory Management**
- **Audio-Normalisierung** für bessere Transkriptionsqualität
- **Batch-Optimierung** für RTX 3090 (24GB VRAM)
- **Konfiguration** über `.env` Datei

## Installation

1. Repository klonen
2. CUDA-Abhängigkeiten installieren:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

1. Repository klonen:

   ```bash
   git clone <repository-url>
   ```

2. Anforderungen installieren:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   conda install -c conda-forge ninja
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   ```

3. Prüfen, ob CUDA und Flash Attention aktiviert sind:

   ```bash
   python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available()); print('Flash Attention aktiviert:', torch.backends.cuda.flash_sdp_enabled())"
   ```

4. Prüfen, ob Flash Attention 2 installiert ist:

   ```bash
   python -c "import flash_attn; print('Flash Attention 2 ist installiert')"
   ```

5. `.env` Datei erstellen mit:

   ```env
   API_KEY=your_api_key
   API_SECRET=your_api_secret
   ```

6. Anwendung starten:

   ```bash
   streamlit run app.py
   ```

## Verwendung

### 1. Transkription
- Datei(en) auswählen
- Transkriptionsparameter einstellen
- Transkription starten

### 2. Knowledge Base Upload
- Transkribierte Dateien werden automatisch erkannt
- Alternativ: Ordner mit Dateien auswählen
- Metadaten konfigurieren
- Upload starten

## Hinweise

- **Empfohlene Batch Size**: 16
- **Empfohlene Chunk Length**: 30 Sekunden
- **Maximale Dateigröße**: 1GB

```

# requirements.txt

```text
# Core Dependencies
streamlit>=1.39.0
torch>=2.1.1+cu124
torchaudio>=2.0.0
transformers>=4.36.0
accelerate>=0.26.0
optimum>=1.16.1

# Audio/Video Processing
moviepy>=1.0.3
ffmpeg-python>=0.2.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# API and Environment
python-dotenv>=1.0.0
requests>=2.31.0

# Optional Performance Optimizations
flash-attn>=2.3.3
```

