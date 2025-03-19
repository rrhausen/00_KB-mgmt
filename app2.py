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
    
    try:
        # Lese den Dateiinhalt direkt
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Erstelle ein temporäres File mit dem Inhalt
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        temp_file.write(content)
        temp_file.close()
        
        # Öffne die Datei neu für den Upload
        file_handle = open(temp_file.name, 'rb')
        
        try:
            files = {
                "file": (file_path.name.replace("%", "Percent").replace("~", ""), file_handle,
                        "text/plain" if file_path.suffix == '.txt' else "application/octet-stream")
            }
            headers = {
                "accept": "application/json",
                "X-API-KEY": api_key,
                "X-API-SECRET": api_secret
            }
            response = requests.post(url, data=metadata, files=files, headers=headers)
            return response
        finally:
            # Schließe und lösche die temporäre Datei
            file_handle.close()
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warnung: Konnte temporäre Datei nicht löschen: {e}")
                
    except Exception as e:
        st.error(f"Fehler beim Upload von {file_path}: {str(e)}")
        raise

def show_upload_tab():
    st.header("Knowledge Base Upload")
    
    # Status der transkribierten Dateien nur anzeigen, wenn vorhanden
    use_transcribed = False
    files_to_process = []
    
    if 'transcribed_files' in st.session_state and st.session_state.transcribed_files:
        st.info(f"Verfügbare transkribierte Dateien: {len(st.session_state.transcribed_files)}")
        
        # Option zum Löschen der Session-Daten
        if st.button("Transkribierte Dateien zurücksetzen"):
            st.session_state.transcribed_files = []
            st.experimental_rerun()
        
        use_transcribed = st.checkbox('Transkribierte Dateien verwenden', value=True)
        if use_transcribed:
            files_to_process = st.session_state.transcribed_files
    
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
        
        if folder_path:  # Nur prüfen wenn ein Pfad eingegeben wurde
            # Normalisiere den Pfad
            folder_path = Path(folder_path)
            
            if folder_path.exists():
                files_to_process = []
                for ext in file_types:
                    files_to_process.extend(list(folder_path.rglob(f'*.{ext}')))
                if not files_to_process:
                    st.warning("Keine Dateien der ausgewählten Typen im Verzeichnis gefunden")
            else:
                st.error("Verzeichnis nicht gefunden")
                return
    
    # Upload-Metadaten
    col1, col2 = st.columns(2)
    with col1:
        project_id = st.text_input('Project ID', value='vc0lgCl8YHKTHQ9ByzYt', key='upload_project_id')
        author_name = st.text_input('Author Name', value='', key='upload_author')
        uploaded_by = st.text_input('Uploaded by', value='trendguru@email.de', key='upload_by')
    with col2:
        added_by = st.text_input('Added by', value='trendguru@email.de', key='upload_added_by')
        category = st.text_input('Category', value='null', key='upload_category')
        last_upload = st.date_input('Last Upload', value=datetime.date.today(), key='upload_date')
    
    if st.button('Upload starten') and folder_path:
        if not files_to_process:
            st.warning("Keine Dateien zum Upload gefunden")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(files_to_process)
        successful_uploads = 0
        failed_uploads = 0
        successful_files = []
        failed_files = []
        
        for idx, file_path in enumerate(files_to_process):
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
                        successful_files.append(file_path.name)
                    else:
                        failed_uploads += 1
                        failed_files.append(f"{file_path.name}: HTTP {response.status_code} - {response.text}")
                
                except Exception as e:
                    failed_uploads += 1
                    failed_files.append(f"{file_path.name}: {str(e)}")
            else:
                failed_uploads += 1
                failed_files.append(f"{file_path.name}: Datei zu groß (>1GB)")
            
            progress_bar.progress((idx + 1) / total_files)
        
        # Zeige Gesamtstatistik
        status_text.text(f"Upload abgeschlossen! Statistik: {successful_uploads} erfolgreich, {failed_uploads} fehlgeschlagen")
        
        # Erfolgreiche Uploads in aufklappbarer Box
        if successful_files:
            with st.expander("Erfolgreich hochgeladene Dateien", expanded=False):
                for file in successful_files:
                    st.write(f"✅ {file}")
        
        # Fehlgeschlagene Uploads in aufklappbarer Box
        if failed_files:
            with st.expander("Fehlgeschlagene Uploads", expanded=True):
                for failed_file in failed_files:
                    st.write(f"❌ {failed_file}")

if __name__ == '__main__':
    show_upload_tab()