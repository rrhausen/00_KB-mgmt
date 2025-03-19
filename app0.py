import streamlit as st
import requests
from datetime import datetime, timedelta
import yt_dlp as youtube_dl
import os
import itertools
import html
import urllib.parse
from tqdm import tqdm
import json

def load_api_keys(file_path):
    with open(file_path, 'r') as file:
        keys = [line.strip() for line in file if line.strip()]
    return keys

# Laden der API-Schlüssel aus der Datei
API_KEYS = load_api_keys('YT_API_keys.txt')
api_key_iterator = itertools.cycle(API_KEYS)
BASE_URL = 'https://www.googleapis.com/youtube/v3'

def get_next_api_key():
    return next(api_key_iterator)

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

def sanitize_filename(filename, video_date, video_id):
    date_obj = datetime.strptime(video_date, '%Y-%m-%dT%H:%M:%SZ')
    date_str = date_obj.strftime('%Y-%m-%d')
    
    filename = html.unescape(filename)
    filename = filename.replace('"', '')
    filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = filename.replace('*', '_').replace('?', '_').replace('"', '_')
    filename = filename.replace('<', '_').replace('>', '_').replace('|', '_')
    filename = filename.replace(' ', '_')
    
    return f"{date_str}_{filename}_{video_id}"

cancel_download = False

def download_video(video_url, video_title, video_id, video_date, format_type, output_dir, video_data=None):
    try:
        # Erstelle sauberen Dateinamen mit Datum
        sanitized_filename = sanitize_filename(video_title, video_date, video_id)
        
        ydl_opts = {
            'format': 'bestaudio/best' if format_type == 'mp3' else 'best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }] if format_type == 'mp3' else [],
            'outtmpl': os.path.join(output_dir, f'{sanitized_filename}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        # Speichere Video-Metadaten in strukturiertem Format
        if video_data and 'snippet' in video_data:
            metadata = {
                "videoId": video_data['id'].get('videoId') if 'id' in video_data else None,
                "publishedAt": video_data['snippet'].get('publishedAt'),
                "channelId": video_data['snippet'].get('channelId'),
                "title": video_data['snippet'].get('title'),
                "description": video_data['snippet'].get('description'),
                "thumbnail": video_data['snippet'].get('thumbnails', {}).get('high', {}).get('url')
            }
            
            id_file_path = os.path.join(output_dir, f'{sanitized_filename}.id')
            with open(id_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        return True
    except Exception as e:
        st.error(f"Fehler beim Download von {video_title}: {str(e)}")
        return False

def cancel_download_callback():
    global cancel_download
    cancel_download = True

def start_download(videos, format, output_dir):
    global cancel_download
    cancel_download = False

    progress_bar = st.progress(0)
    counter_text = st.empty()
    
    for i, video in enumerate(videos):
        if cancel_download:
            st.warning("Download cancelled.")
            break

        progress = (i + 1) / len(videos)
        progress_bar.progress(progress)
        counter_text.text(f"Downloading: {i + 1} of {len(videos)} videos")

        if 'contentDetails' in video:
            video_id = video['contentDetails']['videoId']
        elif 'id' in video and 'videoId' in video['id']:
            video_id = video['id']['videoId']
        else:
            continue

        video_title = video['snippet']['title']
        video_date = video['snippet']['publishedAt']
        video_url = f'https://www.youtube.com/watch?v={video_id}'

        download_video(video_url, video_title, video_id, video_date, format, output_dir, video_data=video)

    if not cancel_download:
        counter_text.text(f"Completed: {len(videos)} of {len(videos)} videos")
        st.success('Download completed!')

def get_playlist_id(playlist_url):
    """Extrahiert die Playlist-ID aus einer YouTube-Playlist-URL"""
    try:
        if 'list=' in playlist_url:
            playlist_id = playlist_url.split('list=')[1]
            if '&' in playlist_id:
                playlist_id = playlist_id.split('&')[0]
            return playlist_id.strip()
    except Exception as e:
        st.error(f"Fehler beim Extrahieren der Playlist-ID: {str(e)}")
    return None

def get_channel_playlists(channel_id):
    all_playlists = []
    next_page_token = None
    
    while True:
        api_key = get_next_api_key()
        url = f'{BASE_URL}/playlists?key={api_key}&channelId={channel_id}&part=snippet&maxResults=50'
        
        if next_page_token:
            url += f'&pageToken={next_page_token}'
            
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            playlists = data.get('items', [])
            all_playlists.extend(playlists)
            
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
        elif response.status_code == 403:
            st.warning(f"API key {api_key} exceeded quota, trying next key...")
        else:
            st.error(f"Error fetching playlists: {data.get('error', {}).get('message', 'Unknown error')}")
            break
    
    return all_playlists

def show_youtube_tab():
    st.title('YouTube Channel Video Downloader')

    download_type = st.radio("Download-Typ auswählen", ["Channel", "Playlist"])

    if download_type == "Playlist":
        input_type = st.radio("Playlist auswählen via", ["Playlist URL", "Channel Playlists"])
        
        if input_type == "Playlist URL":
            playlist_url = st.text_input('YouTube Playlist URL eingeben')
            if playlist_url:
                playlist_id = get_playlist_id(playlist_url)
                if playlist_id:
                    st.success(f"Playlist-ID gefunden: {playlist_id}")
                    if st.button('Videos aus Playlist laden'):
                        videos = get_videos_by_playlist(playlist_id)
                        if videos:
                            st.session_state.videos = videos
                            st.success(f"{len(videos)} Videos in der Playlist gefunden")
                        else:
                            st.error("Keine Videos in der Playlist gefunden")
                else:
                    st.error("Ungültige Playlist-URL")
        else:  # Channel Playlists
            channel_url = st.text_input('YouTube Channel URL oder ID eingeben')
            if channel_url:
                channel_list = get_channel_id(channel_url)
                if channel_list:
                    for channel in channel_list:
                        channel_id = channel['channelId']
                        playlists = get_channel_playlists(channel_id)
                        if playlists:
                            st.success(f"{len(playlists)} Playlists gefunden")
                            playlist_options = {f"{pl['snippet']['title']} ({pl['id']})": pl['id'] 
                                             for pl in playlists}
                            selected_playlist = st.selectbox(
                                "Playlist auswählen",
                                options=list(playlist_options.keys())
                            )
                            if st.button('Videos aus ausgewählter Playlist laden'):
                                selected_playlist_id = playlist_options[selected_playlist]
                                videos = get_videos_by_playlist(selected_playlist_id)
                                if videos:
                                    st.session_state.videos = videos
                                    st.success(f"{len(videos)} Videos in der Playlist gefunden")
                                else:
                                    st.error("Keine Videos in der Playlist gefunden")
                        else:
                            st.warning("Keine Playlists in diesem Channel gefunden")
                else:
                    st.error("Channel nicht gefunden")

    else:  # Channel Option
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
                    playlists = get_channel_playlists(channel_id)
                    if playlists:
                        st.success(f"{len(playlists)} Playlists gefunden")
                        playlist_options = {f"{pl['snippet']['title']}": pl['id'] 
                                         for pl in playlists}
                        selected_playlist = st.selectbox(
                            "Playlist auswählen",
                            options=list(playlist_options.keys())
                        )
                        if st.button('Fetch Videos'):
                            selected_playlist_id = playlist_options[selected_playlist]
                            videos = get_videos_by_playlist(selected_playlist_id)
                            if videos:
                                st.session_state.videos = videos
                                st.success(f"{len(videos)} Videos in der Playlist gefunden")
                                st.write(videos)
                            else:
                                st.error("Keine Videos in der Playlist gefunden")
                    else:
                        st.warning("Keine Playlists in diesem Channel gefunden")
                elif selection == 'All Videos':
                    if st.button('Fetch Videos'):
                        videos = get_all_videos(channel_id)
                        st.session_state.videos = videos
                        st.write(videos)

    # Download-Bereich (gemeinsam für Channel und Playlist)
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