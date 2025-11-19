import streamlit as st
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Streamlit page configuration - MUST be first!
st.set_page_config(
    page_title="Transcription and KB Upload",
    page_icon="🎯",
    layout="wide"
)

# Import tab modules after page config
from app0 import show_youtube_tab
from app1 import show_transcription_tab
from app2 import show_upload_tab

def main():
    st.title("Transcription and Knowledge Base Upload")
    
    # Initialize session state
    if 'transcribed_files' not in st.session_state:
        st.session_state.transcribed_files = []
    if 'use_youtube_api' not in st.session_state:
        st.session_state.use_youtube_api = False

    # Sidebar settings
    with st.sidebar:
        st.header("🛠️ Settings")
        st.session_state.use_youtube_api = st.checkbox(
            "Enable YouTube API",
            value=st.session_state.use_youtube_api,
            help="Enable YouTube API for additional metadata (optional)"
        )
        st.markdown("---")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["YouTube Download", "Transcription", "Knowledge Base Upload"])
    
    with tab1:
        show_youtube_tab()
    
    with tab2:
        show_transcription_tab()
    
    with tab3:
        show_upload_tab()

if __name__ == "__main__":
    main()