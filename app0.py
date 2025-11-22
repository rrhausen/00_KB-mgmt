"""
YouTube Channel Video Downloader
------------------------------
A Streamlit application for downloading videos from YouTube channels and playlists.
Supports both MP3 and MP4 formats with metadata preservation.
"""

# Standard library imports
import os
import json
import html
import itertools
import urllib.parse
from datetime import datetime, timedelta

# Third-party imports
import streamlit as st
import requests
import yt_dlp as youtube_dl
from tqdm import tqdm

#######################
# YouTube API Client
#######################

class YouTubeAPI:
    """
    YouTube Data API v3 Client
    
    Handles API requests with automatic key rotation and error handling.
    Includes methods for fetching videos, playlists, and channel information.
    """
    
    def __init__(self):
        """Initialize the YouTube API client with configuration"""
        self.base_url = 'https://www.googleapis.com/youtube/v3'
        self.api_keys = self.load_api_keys('YT_API_keys.txt')
        self.api_key_iterator = itertools.cycle(self.api_keys)

    def load_api_keys(self, file_path: str) -> list:
        """
        Load YouTube API keys from a file.
        
        Args:
            file_path: Path to the file containing API keys (one per line)
        
        Returns:
            List of API keys
        """
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
            
    def get_next_api_key(self) -> str:
        """Get the next API key from the rotation"""
        return next(self.api_key_iterator)
        
    def make_request(self, url: str) -> tuple[dict | None, str]:
        """
        Make a request to the YouTube API with error handling.
        
        Args:
            url: Full API request URL including parameters
            
        Returns:
            Tuple of (response_data, status)
            Status can be "success", "error", or "quota_exceeded"
        """
        try:
            response = requests.get(url)
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                error = data['error']
                error_msg = error.get('message', 'Unknown error')
                if 'errors' in error:
                    error_details = [f"{err.get('reason')}: {err.get('message')}"
                                   for err in error['errors']]
                    error_msg = f"{error_msg}\n" + "\n".join(error_details)
                raise YouTubeAPIError(error_msg)
            
            # Check HTTP status
            if response.status_code == 403:
                return None, "quota_exceeded"
            elif response.status_code != 200:
                raise YouTubeAPIError(f"HTTP Error: {response.status_code}")
            
            return data, "success"
            
        except YouTubeAPIError as e:
            st.error(f"❌ YouTube API Error: {str(e)}")
            return None, "error"
        except requests.RequestException as e:
            st.error(f"❌ Network Error: {str(e)}")
            return None, "error"
        except ValueError as e:
            st.error(f"❌ Invalid JSON Response: {str(e)}")
            return None, "error"
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            return None, "error"

    def get_videos_by_date(self, channel_id: str, start_date: str, end_date: str) -> list:
        """
        Fetch videos from a channel within a specified date range.
        
        Args:
            channel_id: YouTube channel ID to fetch videos from
            start_date: Start date in RFC 3339 format (YYYY-MM-DDThh:mm:ssZ)
            end_date: End date in RFC 3339 format (YYYY-MM-DDThh:mm:ssZ)
            
        Returns:
            List of video objects containing snippet information
            
        Note:
            Uses pagination to fetch all available videos in the date range
            Handles API quota exceeded scenarios automatically
        """
        max_results = 50
        all_videos = []
        
        st.info(f"🔍 Searching videos for channel {channel_id}\n"
                f"From: {start_date}\n"
                f"To: {end_date}")
        
        page_token = None
        while True:
            # Prepare URL
            api_key = self.get_next_api_key()
            url = (f'{self.base_url}/search'
                   f'?key={api_key}'
                   f'&channelId={channel_id}'
                   f'&part=snippet'
                   f'&type=video'
                   f'&order=date'
                   f'&maxResults={max_results}'
                   f'&publishedAfter={start_date}'
                   f'&publishedBefore={end_date}')
            
            if page_token:
                url += f'&pageToken={page_token}'
            
            # Make request
            data, status = self.make_request(url)
            
            if status == "quota_exceeded":
                continue
            if status == "error":
                break
                
            # Process videos
            new_videos = data.get('items', [])
            if new_videos:
                all_videos.extend(new_videos)
                st.info(f"📄 Found {len(new_videos)} new videos (Total: {len(all_videos)})")
            
            # Check pagination
            if 'nextPageToken' not in data:
                break
            page_token = data['nextPageToken']
        
        # Final status
        if all_videos:
            st.success(f"✅ Found {len(all_videos)} videos in total")
        else:
            st.warning("⚠️ No videos found in the specified time range")
        
        return all_videos

    def get_latest_videos(self, channel_id: str, max_results: int) -> list:
        """
        Fetch the most recent videos from a channel.
        
        Args:
            channel_id: YouTube channel ID to fetch videos from
            max_results: Maximum number of videos to return (1-50)
            
        Returns:
            List of video objects containing snippet information
            
        Note:
            Handles API quota exceeded scenarios automatically
        """
        while True:
            api_key = self.get_next_api_key()
            url = f'{self.base_url}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}'
            
            data, status = self.make_request(url)
            if status == "quota_exceeded":
                continue
            if status == "error":
                return []
                
            return data.get('items', [])

    def get_best_rated_videos(self, channel_id: str, max_results: int) -> list:
        """
        Fetch the most popular videos from a channel.
        
        Args:
            channel_id: YouTube channel ID to fetch videos from
            max_results: Maximum number of videos to return (1-50)
            
        Returns:
            List of video objects sorted by view count
            
        Note:
            Handles API quota exceeded scenarios automatically
        """
        while True:
            api_key = self.get_next_api_key()
            url = f'{self.base_url}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=viewCount&maxResults={max_results}'
            
            data, status = self.make_request(url)
            if status == "quota_exceeded":
                continue
            if status == "error":
                return []
                
            return data.get('items', [])

    def get_videos_by_playlist(self, playlist_id: str) -> list:
        """
        Fetch all videos from a playlist.
        
        Args:
            playlist_id: YouTube playlist ID to fetch videos from
            
        Returns:
            List of playlistItem objects containing video details
            
        Note:
            Uses pagination to fetch all available videos in the playlist
            Handles API quota exceeded scenarios automatically
        """
        all_videos = []
        while True:
            api_key = self.get_next_api_key()
            url = f'{self.base_url}/playlistItems?key={api_key}&playlistId={playlist_id}&part=snippet,contentDetails&maxResults=50'
            
            data, status = self.make_request(url)
            if status == "quota_exceeded":
                continue
            if status == "error":
                break
            
            videos = data.get('items', [])
            all_videos.extend(videos)
            
            if 'nextPageToken' not in data:
                break
            
            while 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
                api_key = self.get_next_api_key()
                url = f'{self.base_url}/playlistItems?key={api_key}&playlistId={playlist_id}&part=snippet,contentDetails&maxResults=50&pageToken={next_page_token}'
                
                data, status = self.make_request(url)
                if status == "quota_exceeded":
                    continue
                if status == "error":
                    break
                
                videos = data.get('items', [])
                all_videos.extend(videos)
        
        return all_videos

    def get_video_details(self, video_id: str) -> dict | None:
        """
        Fetch detailed information for a single video.

        Args:
            video_id: YouTube video ID to fetch details for

        Returns:
            Dictionary containing video details or None if failed
            Includes: title, description, channelTitle, publishedAt, thumbnails, etc.
        """
        api_key = self.get_next_api_key()
        url = f'{self.base_url}/videos?key={api_key}&id={video_id}&part=snippet,contentDetails,statistics,status'

        data, status = self.make_request(url)
        if status == "quota_exceeded":
            # Try with next key
            return self.get_video_details(video_id)
        if status == "error" or not data:
            return None

        items = data.get('items', [])
        if not items:
            return None

        video_item = items[0]

        # Extract video details
        snippet = video_item.get('snippet', {})
        content_details = video_item.get('contentDetails', {})
        statistics = video_item.get('statistics', {})
        status_info = video_item.get('status', {})

        return {
            'videoId': video_id,
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'channelTitle': snippet.get('channelTitle', ''),
            'channelId': snippet.get('channelId', ''),
            'publishedAt': snippet.get('publishedAt', ''),
            'thumbnails': snippet.get('thumbnails', {}),
            'tags': snippet.get('tags', []),
            'categoryId': snippet.get('categoryId', ''),
            'defaultLanguage': snippet.get('defaultLanguage', ''),
            'duration': content_details.get('duration', ''),
            'definition': content_details.get('definition', ''),
            'caption': content_details.get('caption', ''),
            'viewCount': statistics.get('viewCount', '0'),
            'likeCount': statistics.get('likeCount', '0'),
            'commentCount': statistics.get('commentCount', '0'),
            'privacyStatus': status_info.get('privacyStatus', ''),
            'embeddable': status_info.get('embeddable', True),
            'license': status_info.get('license', ''),
            'contentRating': content_details.get('contentRating', {}),
            'regionRestriction': content_details.get('regionRestriction', {})
        }

    def get_channel_id(self, youtube_url: str) -> list:
        """
        Convert various YouTube URL formats to channel ID.
        
        Args:
            youtube_url: URL of YouTube channel (supports @handle, /c/, /user/, and /channel/ formats)
            
        Returns:
            List of dicts containing 'title' and 'channelId' for matching channels
            Returns None if no channel is found or URL is invalid
            
        Note:
            Handles API quota exceeded scenarios automatically by retrying with next key
        """
        youtube_url = urllib.parse.unquote(youtube_url)
        api_key = self.get_next_api_key()
        
        if '@' in youtube_url:
            handle = youtube_url.split('@')[1]
            url = f'{self.base_url}/search?part=snippet&type=channel&q={handle}&key={api_key}'
        elif '/c/' in youtube_url:
            username = youtube_url.split('/')[-1]
            url = f'{self.base_url}/search?part=snippet&type=channel&q={username}&key={api_key}'
        elif '/user/' in youtube_url:
            username = youtube_url.split('/')[-1]
            url = f'{self.base_url}/channels?part=snippet&forUsername={username}&key={api_key}'
        elif '/channel/' in youtube_url:
            return [{'title': 'Direct Channel', 'channelId': youtube_url.split('/')[-1]}]
        else:
            st.error("Invalid YouTube URL format.")
            return None

        data, status = self.make_request(url)
        if status == "quota_exceeded":
            return self.get_channel_id(youtube_url)  # Try again with next key
        if status == "error":
            return None
            
        if 'items' in data and len(data['items']) > 0:
            return [{'title': item['snippet']['title'], 'channelId': item['id']['channelId']} for item in data['items']]
        else:
            st.error("Channel ID not found.")
            return None
            
    def get_channel_playlists(self, channel_id: str) -> list:
        """
        Fetch all playlists from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID to fetch playlists from
            
        Returns:
            List of playlist objects containing snippet information
            
        Note:
            Uses pagination to fetch all available playlists
            Handles API quota exceeded scenarios automatically
        """
        all_playlists = []
        while True:
            api_key = self.get_next_api_key()
            url = f'{self.base_url}/playlists?key={api_key}&channelId={channel_id}&part=snippet&maxResults=50'
            
            data, status = self.make_request(url)
            if status == "quota_exceeded":
                continue
            if status == "error":
                break
            
            playlists = data.get('items', [])
            all_playlists.extend(playlists)
            
            if 'nextPageToken' not in data:
                break
                
            while 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
                api_key = self.get_next_api_key()
                url = f'{self.base_url}/playlists?key={api_key}&channelId={channel_id}&part=snippet&maxResults=50&pageToken={next_page_token}'
                
                data, status = self.make_request(url)
                if status == "quota_exceeded":
                    continue
                if status == "error":
                    break
                
                playlists = data.get('items', [])
                all_playlists.extend(playlists)
        
        return all_playlists

    def get_all_videos(self, channel_id: str) -> list:
        """Fetches all videos from a channel"""
        max_results = 50
        all_videos = []
        while True:
            api_key = self.get_next_api_key()
            url = f'{self.base_url}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}'
            
            data, status = self.make_request(url)
            if status == "quota_exceeded":
                continue
            if status == "error":
                break
            
            videos = data.get('items', [])
            all_videos.extend(videos)
            
            if 'nextPageToken' not in data:
                break
                
            while 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
                api_key = self.get_next_api_key()
                url = f'{self.base_url}/search?key={api_key}&channelId={channel_id}&part=snippet&type=video&order=date&maxResults={max_results}&pageToken={next_page_token}'
                
                data, status = self.make_request(url)
                if status == "quota_exceeded":
                    continue
                if status == "error":
                    break
                
                videos = data.get('items', [])
                all_videos.extend(videos)
        
        return all_videos

# Error Handling
class YouTubeAPIError(Exception):
    """Custom exception for YouTube API errors"""
    pass

# Initialize YouTube API instance
youtube_api = YouTubeAPI()
BASE_URL = youtube_api.base_url

# Global API Functions
def get_video_details(video_id: str) -> dict | None:
    """
    Global wrapper function for YouTubeAPI.get_video_details()

    Args:
        video_id: YouTube video ID

    Returns:
        Video details dictionary or None if failed
    """
    return youtube_api.get_video_details(video_id)

# Utility Functions
def validate_date_format(date_str: str) -> bool:
    """Validates if a string is in RFC 3339 format"""
    try:
        datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
        return True
    except ValueError:
        return False

def sanitize_filename(filename, video_date, video_id):
    """Sanitize filename and prepend date"""
    date_obj = datetime.strptime(video_date, '%Y-%m-%dT%H:%M:%SZ')
    date_str = date_obj.strftime('%Y-%m-%d')
    
    filename = html.unescape(filename)
    filename = filename.replace('"', '')
    filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = filename.replace('*', '_').replace('?', '_').replace('"', '_')
    filename = filename.replace('<', '_').replace('>', '_').replace('|', '_')
    filename = filename.replace(' ', '_')
    
    return f"{date_str}_{filename}_{video_id}"

def get_playlist_id(playlist_url):
    """Extract playlist ID from YouTube URL"""
    try:
        if 'list=' in playlist_url:
            playlist_id = playlist_url.split('list=')[1]
            if '&' in playlist_id:
                playlist_id = playlist_id.split('&')[0]
            return playlist_id.strip()
    except Exception as e:
        st.error(f"Fehler beim Extrahieren der Playlist-ID: {str(e)}")
    return None

# YouTube API Wrapper Functions
def get_next_api_key():
    """Get next API key from the YouTube API instance"""
    return youtube_api.get_next_api_key()

def get_channel_id(youtube_url):
    """Gets channel ID from various URL formats using the YouTube API instance"""
    return youtube_api.get_channel_id(youtube_url)

# YouTube API wrapper functions
def get_videos_by_date(channel_id, start_date, end_date):
    """Fetches videos from a channel within a date range using the YouTube API instance"""
    return youtube_api.get_videos_by_date(channel_id, start_date, end_date)

def get_all_videos(channel_id):
    """Fetches all videos from a channel using the YouTube API instance"""
    return youtube_api.get_all_videos(channel_id)

def get_latest_videos(channel_id, max_results):
    """Fetches latest videos from a channel using the YouTube API instance"""
    return youtube_api.get_latest_videos(channel_id, max_results)

def get_best_rated_videos(channel_id, max_results):
    """Fetches best rated videos from a channel using the YouTube API instance"""
    return youtube_api.get_best_rated_videos(channel_id, max_results)

#######################
# YouTube API Wrappers
#######################

def get_videos_by_playlist(playlist_id):
    """Fetches videos from a playlist using the YouTube API instance"""
    return youtube_api.get_videos_by_playlist(playlist_id)

#######################
# File System Utilities
#######################

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

#######################
# Download Management
#######################

# Download state
cancel_download = False

# Download functions
def normalize_path(path):
    """Convert path to proper UNC format if it's a network path"""
    if path.startswith('\\\\') or path.startswith('//'):
        # Already UNC path or network path
        return path.replace('/', '\\')
    elif path.startswith('\\'):
        # Single backslash network path
        return f"\\\\{path[1:]}"
    else:
        return path

def download_video(video_url, video_title, video_id, video_date, format_type, output_dir, video_data=None):
    """
    Download a video from YouTube with metadata.
    
    Args:
        video_url: YouTube video URL
        video_title: Title of the video
        video_id: YouTube video ID
        video_date: Video publish date
        format_type: 'mp3' or 'mp4'
        output_dir: Directory to save the video
        video_data: Optional metadata dictionary
    """
    try:
        # Normalize and create output directory
        # Ensure proper path format
        output_dir = output_dir.replace('/', '\\')

        # Only add network path prefix for actual network paths
        # Don't modify Windows drive paths (like V:\folder)
        if not output_dir.startswith('\\\\') and not (len(output_dir) > 1 and output_dir[1] == ':'):
            if output_dir.startswith('\\'):
                output_dir = f"\\{output_dir}"
            else:
                output_dir = f"\\\\{output_dir}"

        st.info(f"🔄 Using network path: {output_dir}")
        
        try:
            if not os.path.exists(output_dir):
                st.info(f"📁 Creating directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            # Verify write access
            test_file = os.path.join(output_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
        except Exception as e:
            st.error(f"❌ Cannot access or create directory {output_dir}: {str(e)}")
            return False
        
        # Create sanitized filename
        sanitized_filename = sanitize_filename(video_title, video_date, video_id)
        output_path = os.path.join(output_dir, f'{sanitized_filename}.%(ext)s')
        
        # Prepare download configuration
        st.info(f"💾 Using output directory: {output_dir}")
        
        # Setup paths for download
        final_path = os.path.join(output_dir, f'{sanitized_filename}.{format_type}')
        temp_dir = os.path.join(output_dir, f'.temp_{os.urandom(4).hex()}')
        os.makedirs(temp_dir, exist_ok=True)
        
        progress_placeholder = st.empty()
        def download_progress(d):
            """Simplified progress reporting for download status"""
            if d['status'] == 'downloading' and '_percent_str' in d:
                # Update progress less frequently to reduce clutter
                current = float(d.get('downloaded_bytes', 0))
                total = float(d.get('total_bytes', 100))
                if total > 0 and (current/total * 100) % 10 < 1:  # Update every ~10%
                    progress_placeholder.info(f"⬇️ {d.get('_percent_str', '0%')}")
            elif d['status'] == 'finished':
                progress_placeholder.success("✅ Download complete")
        
        # Configure download options with reduced verbosity for workflow mode
        workflow_mode = st.session_state.get('workflow_mode', False)
        if not workflow_mode:
            st.info(f"📥 Preparing download configuration...")
        
        # Clear progress display before starting
        if progress_placeholder:
            progress_placeholder.empty()

        # Configure youtube-dl options
        # Configure options based on format type
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best' if format_type == 'mp4' else 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s').replace('\\', '/'),
            'progress_hooks': [download_progress],
            'retries': 3,
            'postprocessor_args': ['-v', 'warning'],
            'cachedir': temp_dir,
            'no_warnings': True,
            'quiet': st.session_state.get('workflow_mode', False)
        }
        
        # Add format-specific options
        if format_type == 'mp3':
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        
        # Start download process
        downloaded_file = None
        success = False
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded file
            temp_files = os.listdir(temp_dir)
            downloaded_files = [f for f in temp_files if f.endswith(f'.{format_type}')]
            
            if not downloaded_files:
                st.error(f"❌ No {format_type.upper()} file found after download")
                return False
                
            downloaded_file = os.path.join(temp_dir, downloaded_files[0])
            
            # Handle file move with backup
            st.info(f"📦 Moving file to: {os.path.basename(final_path)}")
            
            # Handle file move with backup
            try:
                if os.path.exists(final_path):
                    backup_path = f"{final_path}.bak"
                    st.warning(f"⚠️ Target file exists, creating backup")
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(final_path, backup_path)
                
                os.replace(downloaded_file, final_path)
                st.success(f"✅ File moved successfully")
            except Exception as e:
                st.error(f"❌ Error moving file: {str(e)}")
                return False
            
            # Save metadata with error handling
            if video_data and 'snippet' in video_data:
                metadata = {
                    "videoId": video_data['id'].get('videoId') if 'id' in video_data else None,
                    "publishedAt": video_data['snippet'].get('publishedAt'),
                    "channelId": video_data['snippet'].get('channelId'),
                    "title": video_data['snippet'].get('title'),
                    "description": video_data['snippet'].get('description'),
                    "thumbnail": video_data['snippet'].get('thumbnails', {}).get('high', {}).get('url')
                }
                
                metadata_path = os.path.join(output_dir, f'{sanitized_filename}.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ Successfully downloaded: {os.path.basename(final_path)}")
            return True

        except youtube_dl.utils.DownloadError as e:
            st.error(f"❌ YouTube download error: {str(e)}")
            return False
        except (OSError, IOError) as e:
            st.error(f"❌ File system error: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return False
        finally:
            # Cleanup temporary files
            try:
                if downloaded_file and os.path.exists(downloaded_file):
                    os.remove(downloaded_file)
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                st.info("🧹 Cleaned up temporary files")
            except Exception as e:
                st.warning(f"⚠️ Could not clean up some temporary files: {str(e)}")
    except youtube_dl.utils.DownloadError as e:
        st.error(f"❌ Download error for {video_title}: {str(e)}")
        return False
    except FileNotFoundError as e:
        st.error(f"❌ Directory error: {str(e)}\nPlease check if the output directory exists and is accessible")
        return False
    except PermissionError as e:
        st.error(f"❌ Permission error: {str(e)}\nPlease check directory permissions")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error downloading {video_title}: {str(e)}")
        return False

def cancel_download_callback():
    global cancel_download
    cancel_download = True

def start_download(videos, format, output_dir):
    """Start batch download of videos with progress tracking"""
    global cancel_download
    cancel_download = False

    total_videos = len(videos)
    successful = 0
    failed = []

    # Create status containers
    summary = st.empty()  # For main progress
    details = st.expander("🔍 Download Details", expanded=False)  # For detailed progress
    progress_bar = st.progress(0)

    # Initialize progress display
    summary.info(f"⏳ Processing {total_videos} videos...")
    
    for i, video in enumerate(videos):
        if cancel_download:
            summary.warning("⚠️ Download cancelled by user.")
            break

        # Update main progress
        progress = (i + 1) / total_videos
        progress_bar.progress(progress)
        summary.info(f"⏳ Processing video {i + 1} of {total_videos}")

        # Process in detail expander
        with details:
            try:
                # Extract video information - support multiple structures
                video_id = None
                video_title = ""
                video_date = ""

                # API response structure (flat)
                if 'videoId' in video:
                    video_id = video['videoId']
                    video_title = video.get('title', 'Unknown Title')
                    video_date = video.get('publishedAt', '')
                # Playlist/channel structure (nested)
                elif 'contentDetails' in video:
                    video_id = video['contentDetails']['videoId']
                    video_title = video['snippet']['title']
                    video_date = video['snippet']['publishedAt']
                # Alternative nested structure
                elif 'id' in video and 'videoId' in video['id']:
                    video_id = video['id']['videoId']
                    video_title = video['snippet']['title']
                    video_date = video['snippet']['publishedAt']

                if not video_id:
                    st.error("❌ Invalid video format")
                    failed.append("Unknown video format")
                    continue
                video_url = f'https://www.youtube.com/watch?v={video_id}'

                # Show current video status
                st.info(f"⬇️ Processing: {video_title}")

                # Attempt download
                if download_video(video_url, video_title, video_id, video_date, format, output_dir, video_data=video):
                    successful += 1
                    st.success(f"✅ Downloaded: {video_title}")
                else:
                    failed.append(video_title)
                    st.error(f"❌ Failed: {video_title}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                failed.append(f"Error: {str(e)}")

        # Update summary stats
        summary.info(f"📊 Progress: {successful}/{total_videos} successful, {len(failed)} failed")

    # Show final summary
    if not cancel_download:
        # Clear progress displays
        progress_bar.empty()
        details.empty()
        
        # Create final summary
        if successful == total_videos:
            summary.success(f'✅ Successfully downloaded all {total_videos} videos!')
        else:
            summary.warning(f'⚠️ Download completed with issues')
            
            # Show detailed results in collapsible sections
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Successful", successful)
            with col2:
                st.metric("❌ Failed", len(failed))
            
            if failed:
                with st.expander("❌ Failed Downloads", expanded=False):
                    for title in failed:
                        st.error(f"- {title}")

#######################
# URL Utilities
#######################

def process_video_url(url: str) -> tuple[str | None, dict | None]:
    """Process video URL and return video ID and metadata"""
    if not url:
        return None, None
        
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        
        if not video_id:
            st.error("❌ Ungültige YouTube URL")
            return None, None
            
        # Create basic metadata matching API structure
        metadata = {
            'contentDetails': {'videoId': video_id},
            'snippet': {
                'title': f'video_{video_id}',
                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'description': '',
                'channelTitle': 'Unknown Channel'
            }
        }
        return video_id, metadata
        
    except Exception as e:
        st.error(f"❌ Fehler beim Verarbeiten der URL: {str(e)}")
        return None, None

def get_playlist_id(playlist_url: str) -> str:
    """Extract playlist ID from a YouTube playlist URL"""
    try:
        if 'list=' in playlist_url:
            playlist_id = playlist_url.split('list=')[1]
            if '&' in playlist_id:
                playlist_id = playlist_id.split('&')[0]
            return playlist_id.strip()
    except Exception as e:
        st.error(f"Error extracting playlist ID: {str(e)}")
    return None

#######################
# User Interface
#######################

def handle_direct_url(use_api: bool) -> None:
    """Handle direct URL input and processing"""
    video_url = st.text_input(
        'YouTube Video URL eingeben',
        placeholder='https://www.youtube.com/watch?v=...',
        help="Einzelne Video-URL eingeben"
    )
    
    if video_url:
        video_id, metadata = process_video_url(video_url)
        if video_id:
            if use_api:
                try:
                    api_metadata = get_video_details(video_id)
                    if api_metadata:
                        metadata = api_metadata
                except Exception as e:
                    st.warning(f"⚠️ API-Abruf fehlgeschlagen: {str(e)}")
            st.session_state.videos = [metadata]
            st.success(f"✅ Video ID erkannt: {video_id}")

def handle_download_section():
    """Display and handle the download configuration section"""
    if 'videos' not in st.session_state or not st.session_state.videos:
        st.info("ℹ️ Keine Videos zum Download ausgewählt")
        return

    st.markdown("---")
    st.subheader("⬇️ Download Einstellungen")
    
    format = st.radio('Format', ('mp3', 'mp4'), index=0)
    output_dir = st.text_input(
        'Netzwerk-Pfad eingeben',
        '\\\\192.168.2.10\\Videos\\YouTube\\Tiago_Forte',
        help="Windows Netzwerk-Pfad Format (\\\\server\\share)"
    )
    
    if not output_dir:
        return

    normalized_path = normalize_path(output_dir)
    if not os.path.exists(normalized_path):
        st.error("❌ Verzeichnis nicht verfügbar")
        return

    st.success(f"✅ Ausgabeverzeichnis: {normalized_path}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Verbindung testen'):
            try:
                test_file = os.path.join(normalized_path, '.write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                st.success("✅ Schreibzugriff OK")
            except Exception as e:
                st.error(f"❌ Fehler: {str(e)}")
    
    with col2:
        if st.button('Download starten'):
            start_download(st.session_state.videos, format, normalized_path)
    
    with col3:
        if st.button('Abbrechen'):
            cancel_download_callback()
            st.warning("⚠️ Download abgebrochen")

def show_youtube_tab():
    """Main user interface for the YouTube Channel Video Downloader"""
    st.title('YouTube Channel Video Downloader')

    # Initialize session state variables
    if 'use_youtube_api' not in st.session_state:
        st.session_state.use_youtube_api = False
    if 'download_type' not in st.session_state:
        st.session_state.download_type = "Direct URL"
        
    use_api = st.session_state.use_youtube_api
    download_type = st.session_state.download_type

    # Display appropriate interface based on API mode
    if not use_api:
        st.info("ℹ️ Simple Mode - Nur direkte Video-Downloads")
        st.markdown("""
        **Verfügbar:**
        - Einzelne Video-Downloads
        - MP3/MP4 Konvertierung
        
        💡 _API-Funktionen deaktiviert_
        """)
        
        # Simple URL input for non-API mode
        video_url = st.text_input(
            'YouTube Video URL eingeben',
            placeholder='https://www.youtube.com/watch?v=...'
        )
        
        if video_url:
            video_id, metadata = process_video_url(video_url)
            if video_id:
                st.session_state.videos = [metadata]
                st.success(f"✅ Video ID: {video_id}")
                handle_download_section()
    
    else:
        st.success("✅ API Mode - Alle Funktionen verfügbar")
        st.session_state.download_type = st.radio(
            "Download-Typ",
            ["Direct URL", "Channel", "Playlist"],
            index=["Direct URL", "Channel", "Playlist"].index(st.session_state.download_type)
        )
        download_type = st.session_state.download_type
        
        if st.session_state.get('download_type') == "Direct URL":
            handle_direct_url(use_api)
            if 'videos' in st.session_state and st.session_state.videos:
                handle_download_section()
                
        elif st.session_state.get('download_type') == "Channel":
            youtube_url = st.text_input('Channel URL/ID')
            if youtube_url:
                channel_list = get_channel_id(youtube_url)
                if channel_list:
                    st.session_state.channel_list = channel_list
                    st.success(f"✅ {len(channel_list)} Channel(s) gefunden")
                    handle_download_section()
                    
        elif st.session_state.get('download_type') == "Playlist":
            st.info("🎵 Playlist-Download")
            playlist_url = st.text_input('Playlist URL')
            if playlist_url:
                playlist_id = get_playlist_id(playlist_url)
                if playlist_id and st.button('Videos laden'):
                    videos = get_videos_by_playlist(playlist_id)
                    if videos:
                        st.session_state.videos = videos
                        st.success(f"✅ {len(videos)} Videos gefunden")
                        handle_download_section()
    
    # API mode handling
    if not use_api:
        st.info("ℹ️ Simple Mode - Nur direkte Video-Downloads")
    else:
        st.success("✅ API Mode - Alle Funktionen verfügbar")
        
        # Channel and Playlist options only in API mode
        if download_type == "Channel":
            youtube_url = st.text_input('Enter YouTube Channel URL or ID')
            channel_list = None
            if youtube_url:
                channel_list = get_channel_id(youtube_url)
                if channel_list:
                    st.session_state.channel_list = channel_list
                    st.success("Channel(s) fetched. Please select one from the dropdown.")
                else:
                    st.error("Failed to fetch Channel ID.")

# Handle URL input based on mode
if 'use_youtube_api' not in st.session_state:
    st.session_state.use_youtube_api = False

if not st.session_state.use_youtube_api:
    # Simple direct URL mode
    video_url = st.text_input(
        'YouTube Video URL eingeben',
        placeholder='https://www.youtube.com/watch?v=...',
        help="Einzelne Video-URL eingeben"
    )
    if video_url:
        video_id, metadata = process_video_url(video_url)
        if video_id:
            st.session_state.videos = [metadata]

elif st.session_state.use_youtube_api and st.session_state.get('download_type') == "Direct URL":
    # Direct URL with API features
    video_url = st.text_input(
        'YouTube Video URL eingeben',
        placeholder='https://www.youtube.com/watch?v=...',
        help="URL eines einzelnen Videos eingeben"
    )
    if video_url:
        video_id, basic_metadata = process_video_url(video_url)
        if video_id and use_api:
            try:
                api_metadata = get_video_details(video_id)
                if api_metadata:
                    st.session_state.videos = [api_metadata]
                else:
                    st.session_state.videos = [basic_metadata]
            except Exception as e:
                st.warning(f"⚠️ API-Abruf fehlgeschlagen, verwende Basis-Metadaten: {str(e)}")
                st.session_state.videos = [basic_metadata]
else:
    video_url = None

    if st.session_state.get('download_type') == "Direct URL":
        video_url = st.text_input('YouTube Video URL eingeben',
                                placeholder='https://www.youtube.com/watch?v=...')
        
        if video_url:
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                # Extract video ID from URL
                video_id = None
                if 'youtube.com/watch?v=' in video_url:
                    video_id = video_url.split('watch?v=')[1].split('&')[0]
                elif 'youtu.be/' in video_url:
                    video_id = video_url.split('youtu.be/')[1].split('?')[0]
                
                if video_id:
                    st.success(f"✅ Video ID erkannt: {video_id}")
                    
                    # Simplified metadata when API is disabled
                    if not use_api:
                        video_data = {
                            'id': {'videoId': video_id},
                            'snippet': {
                                'title': f'video_{video_id}',
                                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                            }
                        }
                        st.session_state.videos = [video_data]
                else:
                    st.error("❌ Ungültige YouTube URL")
            else:
                st.error("❌ Bitte geben Sie eine gültige YouTube URL ein")

    elif download_type == "Playlist" and use_api:
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
                    # Datumseingabe
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            'Start Datum',
                            value=datetime.now() - timedelta(days=7),
                            max_value=datetime.now()
                        )
                    with col2:
                        end_date = st.date_input(
                            'End Datum',
                            value=datetime.now(),
                            max_value=datetime.now()
                        )
                    
                    # Validiere Datumsbereich
                    if start_date > end_date:
                        st.error("⚠️ Start Datum muss vor dem End Datum liegen!")
                    else:
                        if st.button('Fetch Videos'):
                            # Konvertiere zu UTC Timestamps
                            start_datetime = datetime.combine(start_date, datetime.min.time())
                            end_datetime = datetime.combine(end_date, datetime.max.time())
                            start_utc = start_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                            end_utc = end_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                            
                            st.info(f"🔍 Suche Videos vom {start_date.strftime('%d.%m.%Y')} bis {end_date.strftime('%d.%m.%Y')}")
                            videos = get_videos_by_date(channel_id, start_utc, end_utc)
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
        # Download configuration
        format = st.radio('Download Format', ('mp3', 'mp4'), index=0)
        
        # Network path input with validation
        output_dir = st.text_input(
            'Enter network share path (e.g., \\\\server\\share\\folder)',
            '\\\\192.168.2.10\\Videos\\YouTube\\Tiago_Forte',
            help="Use Windows network path format (\\\\server\\share)"
        )
        
        # Normalize and validate path
        if output_dir:
            try:
                normalized_path = normalize_path(output_dir)
                if os.path.exists(normalized_path):
                    st.success(f"✅ Output directory is accessible: {normalized_path}")
                else:
                    try:
                        os.makedirs(normalized_path, exist_ok=True)
                        st.success(f"✅ Created output directory: {normalized_path}")
                    except Exception as e:
                        st.error(f"❌ Cannot access or create directory: {str(e)}")
                        st.info("💡 Make sure you have proper network share permissions")
            except Exception as e:
                st.error(f"❌ Invalid path format: {str(e)}")
        
        # Path validation and download controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_button = st.button('Test Connection')
        with col2:
            download_button = st.button('Download Videos', disabled=not os.path.exists(normalize_path(output_dir)))
        with col3:
            cancel_button = st.button('Cancel Download')

        # Test connection when requested
        if test_button:
            try:
                test_path = normalize_path(output_dir)
                if os.path.exists(test_path):
                    # Try to create a test file
                    test_file = os.path.join(test_path, '.test_write')
                    try:
                        with open(test_file, 'w') as f:
                            f.write('test')
                        os.remove(test_file)
                        st.success(f"✅ Successfully tested write access to {test_path}")
                    except Exception as e:
                        st.error(f"❌ Cannot write to directory: {str(e)}")
                else:
                    st.error(f"❌ Directory not accessible: {test_path}")
            except Exception as e:
                st.error(f"❌ Error testing connection: {str(e)}")

        # Start download if path is valid
        if download_button and os.path.exists(normalize_path(output_dir)):
            start_download(st.session_state.videos, format, normalize_path(output_dir))
        elif download_button:
            st.error("❌ Please test the connection first")

        if cancel_button:
            cancel_download_callback()
            st.warning("⚠️ Download cancelled")
    else:
        st.info("No videos to download. Please fetch videos first.")

if __name__ == "__main__":
    show_youtube_tab()