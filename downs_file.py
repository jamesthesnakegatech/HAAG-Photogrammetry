#!/usr/bin/env python3
"""
BlendedMVS Dataset Downloader
Downloads the BlendedMVS dataset files from OneDrive
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import time


class BlendedMVSDownloader:
    """Download BlendedMVS dataset from OneDrive links"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OneDrive download links for BlendedMVS
        self.download_links = {
            "low_res_part1": {
                "name": "BlendedMVS_low_res_part1.zip",
                "url": "https://1drv.ms/u/s!Ag8Dbz2Aqc81gVLILxpohZLEYiIa?e=MhwYSR",
                "size": "81.5 GB",
                "description": "BlendedMVS Low-res Part 1 (768×576)"
            },
            "low_res_part2": {
                "name": "BlendedMVS_low_res_part2.zip",
                "url": "https://1drv.ms/u/s!Ag8Dbz2Aqc81gVHCxmURGz0UBGns?e=Tnw2KY",
                "size": "80.0 GB",
                "description": "BlendedMVS Low-res Part 2 (768×576)"
            },
            "high_res": {
                "name": "BlendedMVS_high_res.zip",
                "url": "https://1drv.ms/u/s!Ag8Dbz2Aqc81ezb9OciQ4zKwJ_w?e=afFOTi",
                "size": "156 GB",
                "description": "BlendedMVS High-res (2048×1536)"
            },
            "textured_meshes": {
                "name": "BlendedMVS_textured_meshes.zip",
                "url": "https://1drv.ms/u/s!Ag8Dbz2Aqc81fkvi2X9Mmzan0FI?e=7x2WoS",
                "size": "9.42 GB",
                "description": "Textured mesh models"
            },
            "other_images": {
                "name": "BlendedMVS_other_images.zip",
                "url": "https://1drv.ms/u/s!Ag8Dbz2Aqc81gVMgQoHpAJP4jlwo?e=wVOWqD",
                "size": "7.56 GB",
                "description": "Other images"
            }
        }
        
    def download_with_wget(self, url: str, output_file: str) -> bool:
        """
        Download file using wget
        
        Args:
            url: Download URL
            output_file: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Downloading {output_file}...")
        
        try:
            # Use wget with resume capability
            cmd = [
                "wget",
                "-c",  # Continue partial downloads
                "-O", output_file,
                "--no-check-certificate",
                url
            ]
            
            result = subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ wget failed: {e}")
            return False
        except FileNotFoundError:
            print("❌ wget not found. Please install wget:")
            print("   macOS: brew install wget")
            print("   Linux: sudo apt-get install wget")
            return False
            
    def download_with_curl(self, url: str, output_file: str) -> bool:
        """
        Download file using curl
        
        Args:
            url: Download URL
            output_file: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Downloading {output_file} with curl...")
        
        try:
            cmd = [
                "curl",
                "-L",  # Follow redirects
                "-C", "-",  # Resume partial downloads
                "-o", output_file,
                url
            ]
            
            result = subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ curl failed: {e}")
            return False
            
    def print_download_instructions(self):
        """Print manual download instructions"""
        print("\n" + "="*60)
        print("📋 Manual Download Instructions")
        print("="*60)
        print("\nPlease download the following files manually from OneDrive:")
        print("\n1. Go to each link in your browser")
        print("2. Click the download button")
        print("3. Save the files to:", self.output_dir.absolute())
        print("\nDownload links:\n")
        
        for key, info in self.download_links.items():
            print(f"📁 {info['description']} ({info['size']})")
            print(f"   Filename: {info['name']}")
            print(f"   URL: {info['url']}")
            print()
            
    def create_download_script(self):
        """Create a shell script with download commands"""
        script_path = self.output_dir / "download_blendedmvs.sh"
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# BlendedMVS Dataset Download Script\n")
            f.write("# This script attempts to download the dataset files\n\n")
            
            f.write("echo 'Starting BlendedMVS dataset download...'\n")
            f.write("echo 'Note: OneDrive links may require manual download'\n\n")
            
            for key, info in self.download_links.items():
                f.write(f"# {info['description']}\n")
                f.write(f"echo 'Downloading {info['name']} ({info['size']})...'\n")
                f.write(f"wget -c -O {info['name']} '{info['url']}' || ")
                f.write(f"curl -L -C - -o {info['name']} '{info['url']}'\n\n")
                
            f.write("echo 'Download attempts complete!'\n")
            f.write("echo 'If any downloads failed, please download manually from the URLs above'\n")
            
        # Make script executable
        script_path.chmod(0o755)
        print(f"✅ Created download script: {script_path}")
        
    def create_dataset_info(self):
        """Create a JSON file with dataset information"""
        import json
        
        info = {
            "dataset": "BlendedMVS",
            "description": "Large-scale dataset for generalized multi-view stereo networks",
            "stats": {
                "scenes": 113,
                "training_samples": "17k+",
                "architectures": True,
                "sculptures": True,
                "small_objects": True
            },
            "downloads": self.download_links,
            "structure": {
                "PID_format": "PIDxxx where xxx is the project number",
                "subdirectories": [
                    "blended_images - Regular and masked images",
                    "cams - Camera parameters and pair.txt",
                    "rendered_depth_maps - PFM depth files"
                ]
            }
        }
        
        info_path = self.output_dir / "blendedmvs_dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"✅ Created dataset info: {info_path}")
        
    def download_files(self, file_keys: list = None):
        """
        Attempt to download files
        
        Args:
            file_keys: List of file keys to download, or None for all
        """
        if file_keys is None:
            file_keys = list(self.download_links.keys())
            
        print("🚀 Starting BlendedMVS dataset download")
        print(f"📁 Output directory: {self.output_dir.absolute()}\n")
        
        # Note about OneDrive limitations
        print("⚠️  Note: OneDrive direct downloads may not work with wget/curl")
        print("   You may need to download manually through your browser\n")
        
        success_count = 0
        
        for key in file_keys:
            if key not in self.download_links:
                print(f"❌ Unknown file key: {key}")
                continue
                
            info = self.download_links[key]
            output_file = self.output_dir / info['name']
            
            print(f"\n{'='*60}")
            print(f"📦 {info['description']} ({info['size']})")
            print(f"{'='*60}")
            
            # Check if file already exists
            if output_file.exists():
                print(f"✅ File already exists: {output_file}")
                success_count += 1
                continue
                
            # Try downloading
            success = self.download_with_wget(info['url'], str(output_file))
            
            if not success:
                print("   Trying with curl...")
                success = self.download_with_curl(info['url'], str(output_file))
                
            if success:
                success_count += 1
            else:
                print(f"❌ Failed to download {info['name']}")
                print(f"   Please download manually from: {info['url']}")
                
        print(f"\n✅ Successfully downloaded {success_count}/{len(file_keys)} files")
        
        if success_count < len(file_keys):
            self.print_download_instructions()


def main():
    parser = argparse.ArgumentParser(description="Download BlendedMVS Dataset")
    parser.add_argument("--output-dir", "-o", default=".", 
                       help="Output directory for downloads (default: current directory)")
    parser.add_argument("--low-res", action="store_true",
                       help="Download only low-res dataset")
    parser.add_argument("--high-res", action="store_true",
                       help="Download only high-res dataset")
    parser.add_argument("--meshes", action="store_true",
                       help="Download only textured meshes")
    parser.add_argument("--create-script", action="store_true",
                       help="Create download shell script")
    parser.add_argument("--info-only", action="store_true",
                       help="Show download information only")
    
    args = parser.parse_args()
    
    downloader = BlendedMVSDownloader(args.output_dir)
    
    # Create dataset info file
    downloader.create_dataset_info()
    
    if args.info_only:
        downloader.print_download_instructions()
        return 0
        
    if args.create_script:
        downloader.create_download_script()
        print("\nYou can now run: bash download_blendedmvs.sh")
        return 0
        
    # Determine which files to download
    file_keys = []
    
    if args.low_res:
        file_keys.extend(["low_res_part1", "low_res_part2"])
    elif args.high_res:
        file_keys.append("high_res")
    elif args.meshes:
        file_keys.append("textured_meshes")
    else:
        # Default: download low-res dataset
        print("ℹ️  No specific dataset selected. Downloading low-res by default.")
        print("   Use --high-res for high resolution or --meshes for textured meshes\n")
        file_keys.extend(["low_res_part1", "low_res_part2"])
        
    # Start download
    downloader.download_files(file_keys)
    
    print("\n✅ Download process complete!")
    print("\nNext steps:")
    print("1. If downloads failed, use the manual download links above")
    print("2. Extract the downloaded zip files")
    print("3. Run the setup helper to generate list files")
    print("4. Start processing with the SuGaR pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
