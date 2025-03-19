import warnings
import streamlit as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from pathlib import Path
import tempfile
import os
import subprocess
import time
from datetime import datetime, timedelta
import json
import requests
from app0 import get_next_api_key, BASE_URL

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The attention mask is not set')
warnings.filterwarnings('ignore', message='.*forced_decoder_ids.*')

BASE_URL = "https://www.googleapis.com/youtube/v3"

def normalize_path(path):
    if not path:
        return path

    try:
        # Behandle Pfade korrekt, auch mit Leerzeichen und Sonderzeichen
        resolved_path = Path(path.strip()).resolve(strict=False)
        return resolved_path
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

        # Check Flash Attention 2
        flash_attention_status = torch.backends.cuda.flash_sdp_enabled()
        flash_attention_message = "aktiviert" if flash_attention_status else "nicht aktiviert"
        st.info(f"Flash Attention 2 ist {flash_attention_message}")
    else:
        st.error("Keine GPU gefunden - Transkription läuft auf CPU")
    return device

def extract_audio(video_path):
    """Extrahiert Audio aus Video oder gibt den Pfad zurück, wenn es bereits eine MP3 ist"""
    if str(video_path).lower().endswith('.mp3'):
        return str(video_path)
        
    output_path = tempfile.mktemp(suffix='.mp3')
    try:
        command = [
            'ffmpeg', '-i', str(video_path),
            '-q:a', '0', '-map', 'a',
            '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5',
            output_path
        ]
        subprocess.run(command, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Fehler bei der Audio-Extraktion: {e.stderr.decode()}")
        raise
    except Exception as e:
        st.error(f"Unerwarteter Fehler bei der Audio-Extraktion: {str(e)}")
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
            return f"[{start_time} - {end_time}]\n{chunk['text']}\n\n"
        return f"[Zeitstempel nicht verfügbar]\n{chunk['text']}\n\n"
    except (TypeError, KeyError, AttributeError):
        return f"{chunk.get('text', '')}\n\n"

def transcribe_audio(audio_path, model_name, language, batch_size, chunk_length):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        # Konvertiere Video zu Audio wenn nötig
        audio_file = extract_audio(audio_path)
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=True
        )
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
        )
        
        transcribe_params = {
            "batch_size": batch_size,
            "chunk_length_s": chunk_length,
            "return_timestamps": True,
            "stride_length_s": chunk_length // 6,
            "generate_kwargs": {
                "language": language,
                "task": "transcribe"
            }
        }
        
        result = pipe(audio_file, **transcribe_params)
        
        # Lösche temporäre Audio-Datei wenn es eine war
        if audio_file != audio_path:
            try:
                os.unlink(audio_file)
            except Exception as e:
                st.warning(f"Konnte temporäre Audio-Datei nicht löschen: {str(e)}")
        
        return result
        
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {str(e)}")
        raise

def get_video_details(video_id):
    """Ruft Details eines Videos von der YouTube API ab"""
    api_key = get_next_api_key()  # Funktion aus app0.py
    url = f'{BASE_URL}/videos?part=snippet&id={video_id}&key={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and 'items' in data and len(data['items']) > 0:
            snippet = data['items'][0]['snippet']
            # Beschreibung auf 75 Zeichen begrenzen und "..." hinzufügen wenn gekürzt
            description = snippet.get('description', '')
            if len(description) > 75:
                description = description[:72] + "..."
                
            return {
                "channelId": snippet.get('channelId'),
                "description": description,
                "thumbnail": snippet.get('thumbnails', {}).get('high', {}).get('url')
            }
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
            return get_video_details(video_id)  # Rekursiver Aufruf mit nächstem Key
        else:
            st.error(f"Error fetching video details: {data.get('error', {}).get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error accessing YouTube API: {str(e)}")
        return {
            "channelId": None,
            "description": None,
            "thumbnail": None
        }

def save_transcription(file_path, result):
    """Speichert die Transkription mit Metadaten als einen validen JSON-String"""
    try:
        base_path = Path(file_path).with_suffix('')  # Pfad ohne Erweiterung
        id_file = base_path.with_suffix('.id')
        txt_file = base_path.with_suffix('.txt')
        
        metadata = {}
        if id_file.exists():
            # Wenn .id Datei existiert, verwende deren Inhalt
            try:
                with open(id_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                st.warning(f"Konnte Metadaten nicht laden: {str(e)}")
        else:
            # Extrahiere Informationen aus dem Dateinamen
            # Format: YYYY-MM-DD_Title_videoId.ext
            filename = base_path.name
            video_id = filename[-11:]  # Letzte 11 Zeichen sind die Video-ID
            date_str = filename[:10]  # Erste 10 Zeichen sind das Datum
            title = filename[11:-12]  # Zwischen Datum und Video-ID ist der Titel
            
            # Formatiere das Datum
            try:
                published_date = datetime.strptime(date_str, '%Y-%m-%d')
                published_at = published_date.strftime('%Y-%m-%dT00:00:00Z')
            except:
                published_at = datetime.now().strftime('%Y-%m-%dT00:00:00Z')
            
            # Hole zusätzliche Informationen von der YouTube API
            api_details = get_video_details(video_id)
            
            metadata = {
                "videoId": video_id,
                "publishedAt": published_at,
                "title": title.replace('_', ' '),
                "channelId": api_details.get('channelId') if api_details else None,
                "description": api_details.get('description') if api_details else None,
                "thumbnail": api_details.get('thumbnail') if api_details else None
            }
        
        # Erstelle das Transkript
        transcript_content = "".join(format_chunk_with_timestamp(chunk) for chunk in result["chunks"])
        
        # Kombiniere Metadaten und Transkript in einem JSON-Objekt
        combined_data = {
            "videoId": metadata.get("videoId"),
            "publishedAt": metadata.get("publishedAt"),
            "channelId": metadata.get("channelId"),
            "Title": metadata.get("title"),
            "Description": metadata.get("description"),
            "thumbnail": metadata.get("thumbnail"),
            "content": transcript_content
        }
        
        # Speichere als JSON
        with open(txt_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        # Lösche .id Datei nach erfolgreicher Verarbeitung
        if id_file.exists():
            id_file.unlink()
            
    except Exception as e:
        st.error(f"Fehler beim Speichern der Transkription: {str(e)}")

def process_files(files_to_process, model_name, language, batch_size, chunk_length, auto_save=False, output_dir=None, skip_existing=True):
    # Entferne mögliche Duplikate und erstelle eine Liste
    files_to_process = list(dict.fromkeys(files_to_process))
    total_files = len(files_to_process)
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Container für Statusmeldungen
    with st.expander("Verarbeitungsprotokoll", expanded=False):
        status_messages = st.container()
    
    for idx, file in enumerate(files_to_process):
        try:
            file_path = Path(file)
            status_text.text(f"Verarbeite Datei {idx+1} von {total_files}: {file_path.name}")
            
            # Prüfe ob bereits eine TXT-Datei existiert
            txt_file = file_path.with_suffix('.txt')
            if skip_existing and txt_file.exists():
                continue
                
            result = transcribe_audio(str(file_path), model_name, language, batch_size, chunk_length)
            
            # Speichere Transkription
            save_transcription(file_path, result)
            
            # Statusmeldung in der aufklappbaren Box
            with status_messages:
                st.success(f"Transkribiert und gespeichert ({idx+1}/{total_files}): {file_path}")
            
            results[file_path.name] = result
            progress_bar.progress((idx + 1) / total_files)
            
        except Exception as e:
            with status_messages:
                st.error(f"Fehler bei {file_path.name}: {str(e)}")
    
    status_text.text(f"Transkription abgeschlossen! {total_files}/{total_files} Dateien verarbeitet")
    return results

def show_sidebar_workflow():
    with st.sidebar:
        st.header("Automatisierter Workflow")
        
        # YouTube Einstellungen
        st.subheader("YouTube Einstellungen")
        youtube_url = st.text_input('YouTube Channel URL oder ID', key='workflow_yt_url')
        download_mode = st.radio("Download Modus", ['Date Range', 'Playlist', 'All Videos'], key='workflow_mode')
        
        # Date Range Einstellungen wenn ausgewählt
        if download_mode == 'Date Range':
            start_date = st.date_input('Start Datum', datetime.now(), key='workflow_start_date')
            end_date = st.date_input('End Datum', datetime.now() + timedelta(days=1), key='workflow_end_date')
        
        # Format und Verzeichnis
        format_type = st.radio('Download Format', ('mp3', 'mp4'), index=0, key='workflow_format')
        base_directory = st.text_input('Basis-Verzeichnis', '/mnt/youtube', key='workflow_dir')
        
        # Knowledge Base Einstellungen
        st.subheader("Knowledge Base Einstellungen")
        project_id = st.text_input('Project ID', value='vc0lgCl8YHKTHQ9ByzYt', key='workflow_project_id')
        author_name = st.text_input('Author Name', value='', key='workflow_author')
        
        # Start Button
        start_workflow = st.button('Workflow starten')
        
        if start_workflow and youtube_url and base_directory:
            st.session_state.workflow = {
                'youtube_url': youtube_url,
                'download_mode': download_mode,
                'format_type': format_type,
                'base_directory': base_directory,
                'project_id': project_id,
                'author_name': author_name
            }
            if download_mode == 'Date Range':
                st.session_state.workflow.update({
                    'start_date': start_date,
                    'end_date': end_date
                })

def process_workflow():
    if 'workflow' in st.session_state:
        workflow = st.session_state.workflow
        
        # Fortschrittscontainer in der Sidebar
        with st.sidebar:
            st.markdown("---")  # Trennlinie
            st.subheader("Workflow Fortschritt")
            download_progress = st.progress(0, "Download")
            transcribe_progress = st.progress(0, "Transkription")
            upload_progress = st.progress(0, "Upload")
            status_text = st.empty()
            download_counter = st.empty()
            
            # 1. Download Videos
            status_text.text("1/3 Downloading Videos...")
            from app0 import get_channel_id, get_videos_by_date, get_videos_by_playlist, get_all_videos
            
            downloaded_files = []  # Liste für erfolgreich heruntergeladene Dateien
            videos = []
            if workflow['youtube_url']:
                channel_list = get_channel_id(workflow['youtube_url'])
                if channel_list:
                    channel_id = channel_list[0]['channelId']
                    
                    if workflow['download_mode'] == 'Date Range':
                        videos = get_videos_by_date(
                            channel_id, 
                            workflow['start_date'].strftime('%Y-%m-%d'),
                            workflow['end_date'].strftime('%Y-%m-%d')
                        )
                    elif workflow['download_mode'] == 'Playlist':
                        videos = get_videos_by_playlist(channel_id)
                    else:  # All Videos
                        videos = get_all_videos(channel_id)
                    
                    if videos:
                        st.session_state.videos = videos
                        # Download der Videos
                        for i, video in enumerate(videos):
                            progress = (i + 1) / len(videos)
                            download_progress.progress(progress)
                            download_counter.text(f"Downloading: {i + 1} of {len(videos)} videos")
                            
                            # Download-Logik
                            if 'contentDetails' in video:
                                video_id = video['contentDetails']['videoId']
                            elif 'id' in video and 'videoId' in video['id']:
                                video_id = video['id']['videoId']
                            else:
                                continue

                            video_title = video['snippet']['title']
                            video_date = video['snippet']['publishedAt']
                            video_url = f'https://www.youtube.com/watch?v={video_id}'
                            
                            from app0 import download_video
                            success = download_video(
                                video_url, 
                                video_title, 
                                video_id, 
                                video_date, 
                                workflow['format_type'], 
                                workflow['base_directory']
                            )
                            if success:
                                downloaded_files.append(Path(workflow['base_directory']) / f"{video_title}_{video_id}.{workflow['format_type']}")
            
            # 2. Transkription
            status_text.text("2/3 Transkribiere Videos...")
            
            # Finde alle neu heruntergeladenen Dateien, die noch keine TXT haben
            files_to_transcribe = []
            for media_file in downloaded_files:
                txt_file = media_file.with_suffix('.txt')
                if not txt_file.exists():
                    files_to_transcribe.append(media_file)
            
            if files_to_transcribe:
                model_name = "openai/whisper-large-v3"
                language = "de"
                batch_size = 24
                chunk_length = 30
                
                for i, file in enumerate(files_to_transcribe):
                    progress = (i + 1) / len(files_to_transcribe)
                    transcribe_progress.progress(progress)
                    status_text.text(f"Transkribiere: {file.name}")
                    
                    try:
                        result = transcribe_audio(
                            str(file),
                            model_name,
                            language,
                            batch_size,
                            chunk_length
                        )
                        
                        # Verwende save_transcription statt direktem Schreiben
                        save_transcription(file, result)
                        
                    except Exception as e:
                        st.error(f"Fehler bei der Transkription von {file.name}: {str(e)}")
            else:
                transcribe_progress.progress(1.0)
                status_text.text("Keine neuen Dateien zu transkribieren")
            
            # 3. Knowledge Base Upload
            status_text.text("3/3 Upload zur Knowledge Base...")
            from app2 import upload_file
            
            # Finde alle TXT-Dateien der heruntergeladenen Videos
            files_to_upload = [f.with_suffix('.txt') for f in downloaded_files if f.with_suffix('.txt').exists()]
            
            if files_to_upload:
                for i, file_path in enumerate(files_to_upload):
                    progress = (i + 1) / len(files_to_upload)
                    upload_progress.progress(progress)
                    status_text.text(f"Upload: {file_path.name}")
                    
                    try:
                        metadata = {
                            "project_id": workflow['project_id'],
                            "metadata": f'{{"author_name": "{workflow["author_name"]}", "uploaded_by": "workflow", "added_by": "workflow", "category": "transcript", "last_upload": "{datetime.now().strftime("%Y-%m-%d")}"}}'
                        }
                        
                        response = upload_file(file_path, metadata)
                        if response.status_code != 200:
                            st.error(f"Fehler beim Upload von {file_path.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Fehler beim Upload von {file_path.name}: {str(e)}")
            else:
                upload_progress.progress(1.0)
                status_text.text("Keine Dateien zum Upload")
            
            # Workflow-Status zurücksetzen
            if 'workflow' in st.session_state:
                del st.session_state.workflow
            
            if not files_to_upload:
                status_text.text("Workflow abgeschlossen! Keine neuen Dateien verarbeitet.")
            else:
                status_text.text(f"Workflow abgeschlossen! {len(files_to_upload)} Dateien verarbeitet.")

def show_transcription_tab():
    st.title("Schnelle Video & Audio Transkription")
    device = check_gpu_and_model()

    with st.sidebar:
        if not 'workflow' in st.session_state:  # Nur anzeigen wenn kein Workflow läuft
            st.header("Transkriptions-Einstellungen")
            
            # Sprache zuerst auswählen
            language = st.selectbox("Sprache", ["de", "en"], index=0)
            
            # Modell basierend auf Sprache
            if language == "de":
                model_name = st.selectbox(
                    "Modell",
                    ["openai/whisper-large-v3", "openai/whisper-large-v2", "openai/whisper-large",
                     "openai/whisper-large-v3-turbo", "openai/whisper-turbo",
                     "distil-whisper/distil-large-v2", "distil-whisper/distil-medium.en",
                     "distil-whisper/distil-small.en", "distil-whisper/distil-large-v3"],
                    index=0  # Default ist openai/whisper-large-v3
                )
            else:  # language == "en"
                model_name = st.selectbox(
                    "Modell",
                    ["distil-whisper/distil-large-v3", "openai/whisper-large-v3", 
                     "openai/whisper-large-v2", "openai/whisper-large",
                     "openai/whisper-large-v3-turbo", "openai/whisper-turbo",
                     "distil-whisper/distil-large-v2", "distil-whisper/distil-medium.en",
                     "distil-whisper/distil-small.en"],
                    index=0  # Default ist distil-whisper/distil-large-v3
                )

            batch_size = st.slider("Batch Size", 1, 32, 24)
            chunk_length = st.slider("Chunk Länge (Sekunden)", 10, 60, 30)
            auto_save = st.checkbox("Automatisch als TXT speichern", value=True)
            skip_existing = st.checkbox("Vorhandene ausnehmen", value=True)

    folder_paths = st.text_area(
        "Ordnerpfad(e) eingeben (ein Pfad pro Zeile)", 
        placeholder="z.B.:\nV:\\AudioFiles\nV:\\VideoFiles",
        help="Mehrere Pfade können durch Zeilenumbruch getrennt eingegeben werden"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        start_transcription = st.button("Transkription starten")
    with col3:
        refresh = st.button("🔄 Aktualisieren")

    # Container für Status und Debug-Info
    status_container = st.container()
    debug_container = st.container()
    file_selection_container = st.container()

    all_files_to_process = []
    normalized_paths = []

    with debug_container:
        if folder_paths:
            paths = [path.strip() for path in folder_paths.split('\n') if path.strip()]
            # Liste für alle gefundenen Dateien
            all_files_to_process = []
            
            for path in paths:
                normalized_path = normalize_path(path)
                if normalized_path and os.path.exists(normalized_path):
                    normalized_paths.append(normalized_path)
                    # Dateien sofort sammeln für korrekte Statusanzeige
                    if os.path.exists(normalized_path):
                        available_files = [os.path.join(normalized_path, f) for f in os.listdir(normalized_path) 
                                         if f.lower().endswith(('.mp4', '.mp3'))]
                        if skip_existing:
                            existing_transcriptions = {os.path.splitext(f)[0] 
                                                    for f in os.listdir(normalized_path) 
                                                    if f.lower().endswith('.txt')}
                            new_files = [f for f in available_files 
                                       if os.path.splitext(os.path.basename(f))[0] not in existing_transcriptions]
                            all_files_to_process.extend(new_files)
                        else:
                            all_files_to_process.extend(available_files)

        # Entferne mögliche Duplikate
        all_files_to_process = list(dict.fromkeys(all_files_to_process))
        
        # Aktualisierte Statusanzeige mit korrekter Anzahl
        if normalized_paths:
            status_text = f"""
            Status:
            - Start Transcription: {'✅' if start_transcription else '⏸️'}
            - Gefundene Ordner: {len(normalized_paths)} von {len(paths) if folder_paths else 0}
            - Dateien zum Transkribieren: {len(all_files_to_process)}
            - Pfade gültig: ✅
            """
        else:
            status_text = "Bitte geben Sie mindestens einen gültigen Ordnerpfad ein."
        
        st.info(status_text)

    with file_selection_container:
        if folder_paths:
            for path in normalized_paths:
                if os.path.exists(path):
                    # Alle verfügbaren Audio/Video-Dateien finden
                    available_files = [os.path.join(path, f) for f in os.listdir(path) 
                                     if f.lower().endswith(('.mp4', '.mp3'))]
                    
                    if not skip_existing:
                        st.write(f"Dateien in {path}:")
                        selected_files = st.multiselect(
                            "Dateien auswählen", 
                            available_files, 
                            default=available_files,  # Alle Dateien standardmäßig ausgewählt
                            key=f"files_{path}"
                        )
                        all_files_to_process = selected_files  # Aktualisiere die Gesamtliste
                    else:
                        # Bereits transkribierte Dateien finden
                        existing_transcriptions = {os.path.splitext(f)[0] 
                                                for f in os.listdir(path) 
                                                if f.lower().endswith('.txt')}
                        
                        # Dateien filtern, die noch keine Transkription haben
                        new_files = [f for f in available_files 
                                   if os.path.splitext(os.path.basename(f))[0] not in existing_transcriptions]
                        
                        if new_files:
                            st.write(f"Neue Dateien in {path}:")
                            selected_files = st.multiselect(
                                "Dateien auswählen", 
                                new_files, 
                                default=new_files,  # Alle neuen Dateien standardmäßig ausgewählt
                                key=f"files_{path}"
                            )
                            all_files_to_process.extend(selected_files)
                        else:
                            st.info(f"Alle Dateien in {path} wurden bereits transkribiert!")
                else:
                    st.error(f"Ordner nicht gefunden: {path}")

    with status_container:
        if start_transcription and all_files_to_process and normalized_paths:
            # Setze auto_save auf True, um die Transkriptionen zu speichern
            process_files(all_files_to_process, model_name, language, batch_size, chunk_length, auto_save=True, skip_existing=skip_existing)

# Hauptanwendung
def main():
    # Sidebar für den Workflow
    show_sidebar_workflow()
    
    # Ursprüngliche Tabs bleiben erhalten
    tab1, tab2, tab3 = st.tabs(["YouTube Download", "Transkription", "Knowledge Base Upload"])
    
    with tab1:
        from app0 import show_youtube_tab
        show_youtube_tab()
    
    with tab2:
        show_transcription_tab()
    
    with tab3:
        from app2 import show_upload_tab
        show_upload_tab()

if __name__ == "__main__":
    main()
