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
