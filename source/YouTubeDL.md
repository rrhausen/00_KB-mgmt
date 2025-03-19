# requirements.txt

```text
streamlit
requests
stqdm
yt-dlp


```

# youtube_downloader.py

```python
import streamlit as st
import requests
from datetime import datetime, timedelta
import yt_dlp as youtube_dl
import os
import itertools
import html
import urllib.parse
from stqdm import stqdm

# Funktion zum Einlesen der API-Schlüssel aus einer Datei
def load_api_keys(file_path):
    with open(file_path, 'r') as file:
        keys = [line.strip() for line in file if line.strip()]
    return keys

# Laden der API-Schlüssel aus der Datei
API_KEYS = load_api_keys('YT_API_keys.txt')

# Initialisieren eines Iterators zur zyklischen Verwendung der API-Schlüssel
api_key_iterator = itertools.cycle(API_KEYS)

def get_next_api_key():
    return next(api_key_iterator)

BASE_URL = 'https://www.googleapis.com/youtube/v3'

# Funktion zum Abrufen der Channel-ID aus dem Benutzernamen oder der benutzerdefinierten URL
def get_channel_id(youtube_url):
    youtube_url = urllib.parse.unquote(youtube_url)
    api_key = get_next_api_key()
    
    if '@' in youtube_url:
        handle = youtube_url.split('@')[1]
        url = f'{BASE_URL}/search?part=snippet&type=channel&q={handle}&key={api_key}'
    elif '/c/' in youtube_url:
        username = youtube_url.split('/')[-1]
        url = f'{BASE_URL}/search?part=snippet&type=channel&q={username}&key={api_key}'
    elif '/user/' in youtube_url:
        username = youtube_url.split('/')[-1]
        url = f'{BASE_URL}/channels?part=snippet&forUsername={username}&key={api_key}'
    elif '/channel/' in youtube_url:
        return [{'title': 'Direct Channel', 'channelId': youtube_url.split('/')[-1]}]
    else:
        st.error("Invalid YouTube URL format.")
        return None

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        if 'items' in data and len(data['items']) > 0:
            return [{'title': item['snippet']['title'], 'channelId': item['id']['channelId']} for item in data['items']]
        else:
            st.error("Channel ID not found.")
            return None
    elif response.status_code == 403:
        st.warning(f"API key {api_key} exceeded quota, trying next key...")
        return get_channel_id(youtube_url)
    else:
        st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
        return None

# Funktion zum Abrufen von Videos nach Datumsbereich
def get_videos_by_date(channel_id, start_date, end_date):
    max_results = 50
    all_videos = []
    while True:
        api_key = get_next_api_key()
        url = f'{BASE_URL}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&publishedAfter={start_date}T00:00:00Z&publishedBefore={end_date}T23:59:59Z&maxResults={max_results}'
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            all_videos.extend(data.get('items', []))
            
            while 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
                api_key = get_next_api_key()
                url = f'{BASE_URL}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&publishedAfter={start_date}T00:00:00Z&publishedBefore={end_date}T23:59:59Z&maxResults={max_results}&pageToken={next_page_token}'
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    all_videos.extend(data.get('items', []))
                elif response.status_code == 403:
                    st.warning(f"API key {api_key} exceeded quota, trying next key...")
                else:
                    st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
                    break
            return all_videos
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
        else:
            st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
            return []

# Funktion zum Abrufen aller Videos
def get_all_videos(channel_id):
    max_results = 50
    all_videos = []
    while True:
        api_key = get_next_api_key()
        url = f'{BASE_URL}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}'
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            all_videos.extend(data.get('items', []))
            
            while 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
                api_key = get_next_api_key()
                url = f'{BASE_URL}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}&pageToken={next_page_token}'
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    all_videos.extend(data.get('items', []))
                elif response.status_code == 403:
                    st.warning(f"API key {api_key} exceeded quota, trying next key...")
                else:
                    st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
                    break
            return all_videos
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
        else:
            st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
            return []

# Funktion zum Abrufen der neuesten Videos
def get_latest_videos(channel_id, max_results):
    while True:
        api_key = get_next_api_key()
        url = f'{BASE_URL}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}'
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            return data.get('items', [])
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
        else:
            st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
            return []

# Funktion zum Abrufen der am besten bewerteten Videos
def get_best_rated_videos(channel_id, max_results):
    while True:
        api_key = get_next_api_key()
        url = f'{BASE_URL}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=viewCount&maxResults={max_results}'
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            return data.get('items', [])
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
        else:
            st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
            return []

# Funktion zum Abrufen von Videos aus einer Playlist
def get_videos_by_playlist(playlist_id):
    all_videos = []
    while True:
        api_key = get_next_api_key()
        url = f'{BASE_URL}/playlistItems?key={api_key}&playlistId={playlist_id}&part=snippet,contentDetails&maxResults=50'
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            all_videos.extend(data.get('items', []))
            
            while 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
                api_key = get_next_api_key()
                url = f'{BASE_URL}/playlistItems?key={api_key}&playlistId={playlist_id}&part=snippet,contentDetails&maxResults=50&pageToken={next_page_token}'
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    all_videos.extend(data.get('items', []))
                elif response.status_code == 403:
                    st.warning(f"API key {api_key} exceeded quota, trying next key...")
                else:
                    st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
                    break
            return all_videos
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
        else:
            st.error(f"Error fetching data from YouTube API: {data.get('error', {}).get('message', 'Unknown error')}")
            return []

# Funktion zum Bereinigen von Dateinamen
def sanitize_filename(filename):
    filename = html.unescape(filename)
    filename = filename.replace('"', '')
    return filename.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace(' ', '_')

# Variable zur Steuerung des Download-Abbruchs
cancel_download = False

# Funktion zum Herunterladen von Videos als MP4 oder MP3
def download_video(video_url, video_title, video_id, format='mp3', output_dir='/mnt/youtube'):
    global cancel_download
    video_title = sanitize_filename(video_title)
    output_path = os.path.join(output_dir, f'{video_title}_{video_id}.{format}')

    if os.path.exists(output_path):
        st.write(f"File {output_path} already exists, skipping download.")
        return

    ydl_opts = {
        'format': 'bestaudio/best' if format == 'mp3' else 'best',
        'outtmpl': os.path.join(output_dir, f'{video_title}_{video_id}.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }] if format == 'mp3' else [],
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        if format == 'mp3':
            os.rename(output_path.replace('.webm', '.mp3'), output_path)
    except Exception as e:
        st.error(f"Error downloading video: {e}")

# Funktion zum Abbruch des Downloads
def cancel_download_callback():
    global cancel_download
    cancel_download = True

# Funktion zum Starten des Downloads
def start_download(videos, format, output_dir):
    global cancel_download
    cancel_download = False
    for i, video in enumerate(stqdm(videos)):
        if cancel_download:
            st.warning("Download cancelled.")
            break
        if 'contentDetails' in video:
            video_id = video['contentDetails']['videoId']
        elif 'id' in video and 'videoId' in video['id']:
            video_id = video['id']['videoId']
        else:
            continue
        video_title = video['snippet']['title']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        st.write(f'Downloading {video_url}')
        download_video(video_url, video_title, video_id, format, output_dir)
        st.text(f"{i + 1}/{len(videos)} completed")
    if not cancel_download:
        st.success('Download completed!')

# Streamlit UI
st.title('YouTube Channel Video Downloader')

youtube_url = st.text_input('Enter YouTube Channel URL or ID')
channel_list = None
if youtube_url:
    channel_list = get_channel_id(youtube_url)
    if channel_list:
        st.session_state.channel_list = channel_list
        st.success("Channel(s) fetched. Please select one from the dropdown.")
    else:
        st.error("Failed to fetch Channel ID.")

if 'channel_list' in st.session_state:
    channel_options = [f"{channel['title']} - {channel['channelId']}" for channel in st.session_state.channel_list]
    selected_channel = st.selectbox("Select a Channel", channel_options)
    selected_channel_id = selected_channel.split(' - ')[-1]
    st.session_state.selected_channel_id = selected_channel_id

if 'selected_channel_id' in st.session_state:
    channel_id = st.session_state.selected_channel_id
else:
    channel_id = None

selection = st.radio('Select Videos By', ('Date Range', 'Latest Videos', 'Best Rated Videos', 'Playlist', 'All Videos'))

videos = []

if channel_id:
    if selection == 'Date Range':
        start_date = st.date_input('Start Date', datetime.now())
        end_date = st.date_input('End Date', datetime.now() + timedelta(days=1))
        if st.button('Fetch Videos'):
            videos = get_videos_by_date(channel_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            st.session_state.videos = videos
            st.write(videos)
    elif selection == 'Latest Videos':
        num_videos_options = [10, 20, 50, 100, 200, 300, 500]
        num_videos = st.selectbox('Number of Videos', num_videos_options + ["Custom"])
        if num_videos == "Custom":
            num_videos = st.number_input('Enter number of videos', min_value=1, value=10, step=1)
        if st.button('Fetch Videos'):
            videos = get_latest_videos(channel_id, int(num_videos))
            st.session_state.videos = videos
            st.write(videos)
    elif selection == 'Best Rated Videos':
        num_videos_options = [10, 20, 50, 100, 200, 300, 500]
        num_videos = st.selectbox('Number of Videos', num_videos_options + ["Custom"])
        if num_videos == "Custom":
            num_videos = st.number_input('Enter number of videos', min_value=1, value=10, step=1)
        if st.button('Fetch Videos'):
            videos = get_best_rated_videos(channel_id, int(num_videos))
            st.session_state.videos = videos
            st.write(videos)
    elif selection == 'Playlist':
        playlist_id = st.text_input('Enter Playlist ID')
        if st.button('Fetch Videos'):
            videos = get_videos_by_playlist(playlist_id)
            st.session_state.videos = videos
            st.write(videos)
    elif selection == 'All Videos':
        if st.button('Fetch Videos'):
            videos = get_all_videos(channel_id)
            st.session_state.videos = videos
            st.write(videos)

if 'videos' in st.session_state and st.session_state.videos:
    format = st.radio('Download Format', ('mp3', 'mp4'), index=0)
    output_dir = st.text_input('Enter the directory to save videos', '/mnt/youtube')
    output_dir = output_dir.replace("\\", "/")

    col1, col2 = st.columns(2)
    with col1:
        download_button = st.button('Download Videos')
    with col2:
        cancel_button = st.button('Cancel Download')

    if download_button:
        start_download(st.session_state.videos, format, output_dir)

    if cancel_button:
        cancel_download_callback()
else:
    st.info("No videos to download. Please fetch videos first.")

```

