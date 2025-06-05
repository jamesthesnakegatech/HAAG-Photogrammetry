#!/usr/bin/env python3
"""
OneDrive Download Workarounds for BlendedMVS Dataset
Various methods to try downloading from OneDrive
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlparse
import base64


class OneDriveDownloader:
    """Enhanced OneDrive downloader with multiple strategies"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def method1_direct_download_link(self, share_url: str) -> str:
        """
        Method 1: Convert OneDrive share URL to direct download URL
        
        OneDrive share URLs can sometimes be converted to direct download links
        by modifying the URL structure.
        """
        print("🔧 Method 1: Trying direct download link conversion...")
        
        # Extract the file ID from the share URL
        # Format: https://1drv.ms/u/s!{FileID}?e={param}
        match = re.search(r's!([A-Za-z0-9\-_]+)', share_url)
        if match:
            file_id = match.group(1)
            
            # Try different OneDrive direct download formats
            direct_urls = [
                f"https://api.onedrive.com/v1.0/shares/u!{file_id}/root/content",
                f"https://onedrive.live.com/download?cid={file_id}",
                f"https://storage.live.com/items/{file_id}?authkey={file_id}"
            ]
            
            for url in direct_urls:
                print(f"  Trying: {url[:50]}...")
                if self._test_url(url):
                    return url
                    
        return None
        
    def method2_embed_link(self, share_url: str) -> str:
        """
        Method 2: Use OneDrive embed link format
        
        Sometimes embed links work when direct links don't
        """
        print("🔧 Method 2: Trying embed link format...")
        
        # Try to get the embed version
        embed_url = share_url.replace('/redir?', '/embed?')
        
        # Try to extract download URL from embed page
        try:
            response = requests.get(embed_url, allow_redirects=True)
            if response.status_code == 200:
                # Look for download URL in the response
                download_match = re.search(r'downloadUrl":"([^"]+)"', response.text)
                if download_match:
                    download_url = download_match.group(1).replace('\\u0026', '&')
                    return download_url
        except:
            pass
            
        return None
        
    def method3_decode_base64_url(self, share_url: str) -> str:
        """
        Method 3: Decode base64 encoded URLs
        
        Some OneDrive URLs contain base64 encoded actual URLs
        """
        print("🔧 Method 3: Checking for base64 encoded URLs...")
        
        # Look for base64 encoded parts in the URL
        if 'redeem=' in share_url:
            encoded = share_url.split('redeem=')[-1]
            try:
                decoded = base64.b64decode(encoded).decode('utf-8')
                print(f"  Decoded URL: {decoded}")
                return decoded
            except:
                pass
                
        return None
        
    def method4_rclone(self, share_url: str, output_file: str):
        """
        Method 4: Use rclone (if installed)
        
        rclone is a command-line tool that supports OneDrive
        """
        print("🔧 Method 4: Trying rclone...")
        
        # Check if rclone is installed
        try:
            subprocess.run(['rclone', 'version'], capture_output=True, check=True)
        except:
            print("  ❌ rclone not installed")
            print("  Install with: brew install rclone")
            return False
            
        # Create temporary rclone config for public OneDrive link
        config_content = f"""
[onedrive_public]
type = onedrive
drive_type = personal
"""
        
        config_file = self.output_dir / 'rclone_temp.conf'
        with open(config_file, 'w') as f:
            f.write(config_content)
            
        try:
            cmd = [
                'rclone', 'copy',
                f'--config={config_file}',
                share_url,
                str(self.output_dir)
            ]
            subprocess.run(cmd, check=True)
            return True
        except:
            return False
        finally:
            config_file.unlink(missing_ok=True)
            
    def method5_selenium(self, share_url: str, output_file: str):
        """
        Method 5: Use Selenium to automate browser download
        
        This requires selenium and a web driver
        """
        print("🔧 Method 5: Browser automation with Selenium...")
        
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            print("  ❌ Selenium not installed")
            print("  Install with: pip install selenium")
            return False
            
        # Setup Chrome options
        options = webdriver.ChromeOptions()
        prefs = {"download.default_directory": str(self.output_dir)}
        options.add_experimental_option("prefs", prefs)
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(share_url)
            
            # Wait for download button and click
            download_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download')]"))
            )
            download_button.click()
            
            # Wait for download to start
            time.sleep(5)
            
            print("  ✅ Download started via browser automation")
            print("  Check your downloads folder")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Selenium failed: {e}")
            return False
        finally:
            if 'driver' in locals():
                driver.quit()
                
    def _test_url(self, url: str) -> bool:
        """Test if a URL is accessible"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def download_with_method(self, method_func, share_url: str, output_file: str):
        """Try a download method"""
        result = method_func(share_url)
        
        if isinstance(result, str) and result:  # Got a URL
            print(f"  ✅ Got URL: {result[:50]}...")
            return self._download_url(result, output_file)
        elif isinstance(result, bool):  # Method returned success/failure
            return result
        else:
            return False
            
    def _download_url(self, url: str, output_file: str) -> bool:
        """Download from a direct URL"""
        try:
            # Try with requests
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_file, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
                            
                print("\n  ✅ Download successful")
                return True
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            
        return False


def create_alternative_sources():
    """Create a file with alternative download sources"""
    
    alternatives = """
# Alternative Download Sources for BlendedMVS

## 1. Request from Authors
Contact the dataset authors directly:
- Check the paper: https://arxiv.org/abs/1911.10127
- GitHub issues: https://github.com/YoYo000/BlendedMVS/issues

## 2. Academic Torrents
Sometimes large datasets are shared via Academic Torrents:
- Search: https://academictorrents.com

## 3. University Mirrors
Some universities host mirrors of popular datasets:
- ETH Zurich datasets
- Stanford datasets
- CMU datasets

## 4. Cloud Storage Services
The authors might provide alternative links:
- Google Drive
- Dropbox
- AWS S3

## 5. Use Existing Downloads
If colleagues have downloaded the dataset:
- Use rsync/scp to transfer
- Create a local network share

## 6. Download via Institution
If you're at a university:
- Use institutional network (often bypasses restrictions)
- Request IT department assistance
- Use campus computing clusters

## 7. Download Managers
Use specialized download managers:
- JDownloader2 (supports OneDrive)
- Internet Download Manager (IDM)
- Free Download Manager

## 8. Command Line Tools
Try alternative CLI tools:
```bash
# aria2c - powerful download utility
brew install aria2
aria2c -x 16 -s 16 "URL"

# youtube-dl (supports some cloud services)
brew install youtube-dl
youtube-dl "URL"
```

## 9. Browser Extensions
- OneDrive Direct Link Generator
- Universal Bypass
- Direct Download Link Generator

## 10. Python Libraries
```python
# pyOneDrive
pip install pyonedrive

# cloudscraper (bypasses some protections)
pip install cloudscraper
```
"""
    
    with open("blendedmvs_alternative_sources.md", "w") as f:
        f.write(alternatives)
        
    print("✅ Created blendedmvs_alternative_sources.md with alternative download methods")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OneDrive Download Workarounds")
    parser.add_argument("share_url", nargs='?', help="OneDrive share URL")
    parser.add_argument("--output", "-o", help="Output filename")
    parser.add_argument("--method", "-m", type=int, choices=[1,2,3,4,5],
                       help="Specific method to try (1-5)")
    parser.add_argument("--alternatives", action="store_true",
                       help="Create file with alternative download sources")
    
    args = parser.parse_args()
    
    if args.alternatives:
        create_alternative_sources()
        return 0
        
    if not args.share_url:
        print("❌ Please provide a OneDrive share URL")
        return 1
        
    downloader = OneDriveDownloader()
    
    # BlendedMVS URLs
    urls = {
        "part1": "https://1drv.ms/u/s!Ag8Dbz2Aqc81gVLILxpohZLEYiIa?e=MhwYSR",
        "part2": "https://1drv.ms/u/s!Ag8Dbz2Aqc81gVHCxmURGz0UBGns?e=Tnw2KY"
    }
    
    # Determine which file we're downloading
    output_file = args.output
    if not output_file:
        if "gVLI" in args.share_url:
            output_file = "BlendedMVS_low_res_part1.zip"
        elif "gVHC" in args.share_url:
            output_file = "BlendedMVS_low_res_part2.zip"
        else:
            output_file = "download.zip"
            
    print(f"🎯 Attempting to download: {output_file}")
    print(f"📁 Output directory: {downloader.output_dir.absolute()}\n")
    
    # Methods to try
    methods = [
        (1, downloader.method1_direct_download_link),
        (2, downloader.method2_embed_link),
        (3, downloader.method3_decode_base64_url),
        (4, lambda url: downloader.method4_rclone(url, output_file)),
        (5, lambda url: downloader.method5_selenium(url, output_file))
    ]
    
    if args.method:
        # Try specific method only
        methods = [(m[0], m[1]) for m in methods if m[0] == args.method]
        
    # Try each method
    for method_num, method_func in methods:
        print(f"\n{'='*60}")
        success = downloader.download_with_method(method_func, args.share_url, output_file)
        
        if success:
            print(f"\n✅ Successfully downloaded using Method {method_num}")
            return 0
            
    print("\n❌ All automated methods failed")
    print("\n" + "="*60)
    print("📋 Manual Download Required")
    print("="*60)
    print(f"\n1. Open this URL in your browser:")
    print(f"   {args.share_url}")
    print(f"\n2. Click the Download button")
    print(f"\n3. Save as: {downloader.output_dir / output_file}")
    print("\n💡 Run with --alternatives flag for more download options")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
