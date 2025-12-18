import os
import sys
import json
import argparse
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# SCOPES for full management
SCOPES = [
    'https://www.googleapis.com/auth/youtube.force-ssl',
    'https://www.googleapis.com/auth/youtube.upload'
]

class YouTubeToolkit:
    def __init__(self, channel_name='default', secrets_file='client_secrets.json'):
        self.secrets_file = secrets_file
        # Use a unique token file for each channel
        self.token_file = f'yt_token_{channel_name}.pickle'
        self.youtube = self._authenticate()

    def _authenticate(self):
        # The key is prompt='consent' to force the permissions screen where 
        # brand account selection happens.
        creds = None
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.secrets_file, SCOPES)
                # FORCE the brand selector by using prompt='consent'
                creds = flow.run_local_server(
                    port=0, 
                    prompt='consent', 
                    access_type='offline'
                )
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('youtube', 'v3', credentials=creds)

    def get_current_channel(self):
        """Prints the title and ID of the channel currently in use."""
        try:
            request = self.youtube.channels().list(part="snippet", mine=True)
            response = request.execute()
            if not response.get('items'):
                print("[-] No channel found for this account.")
                return None
            
            channel = response['items'][0]
            print(f"[i] Logged in as: {channel['snippet']['title']} ({channel['id']})")
            return channel['id']
        except Exception as e:
            print(f"[-] Authentication failed: {e}")
            return None
    
    def list_my_channels(self):
        """Lists all channels the current token has access to."""
        # 'mine=True' returns the channel the token was authorized for
        request = self.youtube.channels().list(
            part="snippet,contentDetails,statistics",
            mine=True
        )
        response = request.execute()
        
        for item in response.get('items', []):
            print(f"--- Channel Found ---")
            print(f"Title: {item['snippet']['title']}")
            print(f"ID:    {item['id']}")
            print(f"URL:   https://youtube.com/channel/{item['id']}")

    def upload_video(self, file_path, title, description, category="22", privacy="private"):
        """Uploads a video ensuring the byte-stream is closed correctly."""
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'categoryId': category
            },
            'status': {
                'privacyStatus': privacy,
                'selfDeclaredMadeForKids': False
            }
        }
        
        # 1. Use a chunk size that is a multiple of 256KB (Required by Google)
        # 1024 * 1024 = 1MB. This is safer than -1 for "closing" the connection.
        media = MediaFileUpload(
            file_path, 
            mimetype='video/mp4', 
            chunksize=1024*1024, 
            resumable=True
        )
        
        request = self.youtube.videos().insert(
            part="snippet,status", 
            body=body, 
            media_body=media
        )
        
        print(f"[*] Uploading {file_path}...")
        
        response = None
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    # This ensures you see the progress correctly
                    print(f"    Uploaded: {int(status.progress() * 100)}%", end="\r")
                
                if response is not None:
                    # This is the "Close" signal
                    print(f"\n[+] Finalizing upload...")
                    if 'id' in response:
                        print(f"[+] Success! Video ID: {response['id']}")
                        return response['id']
                    else:
                        print(f"[-] Upload failed with unexpected response: {response}")
            except Exception as e:
                print(f"\n[-] A connection error occurred: {e}")
                # Optional: implement a sleep and retry here
                break

        return None

    def download_transcript(self, video_id, lang='de', output_file=None):
        """Downloads the SRT/Caption track for a video."""
        results = self.youtube.captions().list(part="snippet", videoId=video_id).execute()
        
        caption_id = None
        for item in results.get('items', []):
            if item['snippet']['language'] == lang:
                caption_id = item['id']
                break
        
        if not caption_id:
            print(f"[-] No caption track found for language: {lang}")
            return None

        request = self.youtube.captions().download(id=caption_id, tfmt='srt')
        fh = io.FileIO(output_file or f"{video_id}_{lang}.srt", 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        print(f"[+] Transcript saved to {fh.name}")
        return fh.name

    def upload_audio_as_video(self, audio_path, title, description):
        """Workaround: Uses FFmpeg to create a silent video from audio, then uploads."""
        video_path = f"{audio_path}.mp4"
        print("[*] Converting audio to compatible video format (FFmpeg)...")
        # Creates a black frame video with the audio track
        os.system(f'ffmpeg -loop 1 -i black_frame.png -i "{audio_path}" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest "{video_path}"')
        
        vid_id = self.upload_video(video_path, title, description)
        os.remove(video_path) # Cleanup
        return vid_id

# --- CLI INTERFACE ---
if __name__ == "__main__":
    # 1. Main parser
    parser = argparse.ArgumentParser(description="YouTube API Toolkit")
    # Move --channel here so it works for ALL commands
    parser.add_argument("--channel", default="default", help="Identity/Token profile name")
    
    subparsers = parser.add_subparsers(dest="command")

    # 2. Whoami command
    subparsers.add_parser("whoami", help="Check current authenticated channel")

    # 3. Upload command
    up = subparsers.add_parser("upload", help="Upload a video")
    up.add_argument("file", help="Path to video file")
    up.add_argument("--title", required=True)
    up.add_argument("--desc", default="Uploaded via Toolkit")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize Toolkit
    yt = YouTubeToolkit(channel_name=args.channel)

    if args.command == "whoami":
        yt.get_current_channel()
    elif args.command == "upload":
        yt.upload_video(args.file, args.title, args.desc)