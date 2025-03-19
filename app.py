import streamlit as st
from app0 import show_youtube_tab
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
    tab1, tab2, tab3 = st.tabs(["YouTube Download", "Transkription", "Knowledge Base Upload"])
    
    with tab1:
        show_youtube_tab()
    
    with tab2:
        show_transcription_tab()
    
    with tab3:
        show_upload_tab()

if __name__ == "__main__":
    main()