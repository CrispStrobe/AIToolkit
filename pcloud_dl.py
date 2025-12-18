import os
import re
import json
import sys
import argparse
import requests
from datetime import datetime
from tqdm import tqdm

class PCloudDownloader:
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://e.pcloud.link/"
        })
        
        # Auto-detect Cluster
        self.is_eu = "e.pcloud.link" in url or "eapi.pcloud.com" in url
        self.api_host = "eapi.pcloud.com" if self.is_eu else "api.pcloud.com"
        
        # Extract Code
        match = re.search(r"code=([a-zA-Z0-9]+)", url)
        self.code = match.group(1) if match else None

    def scan_link(self):
        """Extracts metadata from the pCloud landing page."""
        if not self.code:
            return None, []
        
        try:
            response = self.session.get(self.url, timeout=15)
            if response.status_code != 200:
                return None, []

            # Locate the embedded JSON object in HTML
            pattern = r"var publinkData = (\{.*?\});"
            match = re.search(pattern, response.text, re.DOTALL)
            if not match:
                return None, []

            data = json.loads(match.group(1))
            meta = data.get("metadata", {})
            folder_name = meta.get("name", "pCloud_Shared")
            
            # If folder, get 'contents'. If single file, wrap metadata in a list.
            files = meta.get("contents", [])
            if not files and not meta.get("isfolder"):
                files = [meta]
                
            return folder_name, files
        except Exception as e:
            print(f"[!] Scan error: {e}")
            return None, []

    def get_download_url(self, fileid):
        """Requests a temporary download link for a specific file ID."""
        api_url = f"https://{self.api_host}/getpublinkdownload?code={self.code}&fileid={fileid}"
        try:
            res = self.session.get(api_url).json()
            if res.get("result") == 0:
                return f"https://{res['hosts'][0]}{res['path']}"
        except:
            pass
        return None

    def download_stream(self, file_info, target_path):
        """Streams the file to disk with a progress bar."""
        url = self.get_download_url(file_info['fileid'])
        if not url:
            return False

        with self.session.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f, tqdm(
                desc=file_info['name'],
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return True

def main():
    parser = argparse.ArgumentParser(description="pCloud Public Link CLI Downloader")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Command: scan
    scan_p = subparsers.add_parser("scan", help="Verify link and list files")
    scan_p.add_argument("url", help="pCloud public link URL")

    # Command: download
    dl_p = subparsers.add_parser("download", help="Download files from link")
    dl_p.add_argument("url", help="pCloud public link URL")
    dl_p.add_argument("-o", "--output", default=".", help="Output directory")
    dl_p.add_argument("-s", "--subpath", help="Relative subpath (e.g. pcloud/shared/myfiles)")
    dl_p.add_argument("--select", nargs="+", help="Only download files containing these keywords")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    downloader = PCloudDownloader(args.url)
    folder_name, files = downloader.scan_link()

    if not folder_name:
        print("[-] Failed to scan link. Is it valid/public?")
        return

    # CLI Output for Scan
    if args.command == "scan":
        print(f"\n[+] Link: {folder_name}")
        print(f"[+] Total Files: {len(files)}")
        print("-" * 50)
        for f in files:
            print(f"  - {f['name']} ({f['size']/1e6:.2f} MB)")
        return

    # CLI Output for Download
    if args.command == "download":
        # Resolve target directory
        if args.subpath:
            target_dir = os.path.join(args.output, args.subpath)
        else:
            date_prefix = datetime.now().strftime("%Y-%m-%d")
            target_dir = os.path.join(args.output, f"pcloud/shared/{date_prefix}_{folder_name}")
        
        os.makedirs(target_dir, exist_ok=True)
        print(f"[*] Target Directory: {target_dir}")

        # Filter files if requested
        to_download = files
        if args.select:
            to_download = [f for f in files if any(k.lower() in f['name'].lower() for k in args.select)]
        
        print(f"[*] Downloading {len(to_download)} files...")
        for f in to_download:
            path = os.path.join(target_dir, f['name'])
            try:
                downloader.download_stream(f, path)
            except Exception as e:
                print(f"[!] Error downloading {f['name']}: {e}")

if __name__ == "__main__":
    main()