#!/usr/bin/env python3
"""
BlendedMVS Setup Helper
This script helps set up the BlendedMVS dataset and generates project lists
"""

import os
import sys
from pathlib import Path
import json
import argparse


def generate_project_lists(dataset_root: str):
    """
    Generate project list files based on the dataset structure
    
    Args:
        dataset_root: Root directory of the BlendedMVS dataset
    """
    dataset_path = Path(dataset_root)
    
    # Find all PID directories
    pid_dirs = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name.startswith('PID'):
            # Check if it has the expected subdirectories
            if (item / 'blended_images').exists() and (item / 'cams').exists():
                pid_dirs.append(item.name)
    
    # Sort PIDs numerically
    pid_dirs.sort(key=lambda x: int(x[3:]) if x[3:].isdigit() else float('inf'))
    
    print(f"Found {len(pid_dirs)} valid project directories")
    
    # Generate BlendedMVS_list.txt (typically PIDs 0-112 for the original 113 scenes)
    blendedmvs_pids = [pid for pid in pid_dirs if pid[3:].isdigit() and int(pid[3:]) <= 112]
    
    with open(dataset_path / 'BlendedMVS_list.txt', 'w') as f:
        for pid in blendedmvs_pids:
            f.write(f"{pid}\n")
    print(f"Created BlendedMVS_list.txt with {len(blendedmvs_pids)} projects")
    
    # Generate BlendedMVG_list.txt (all PIDs)
    with open(dataset_path / 'BlendedMVG_list.txt', 'w') as f:
        for pid in pid_dirs:
            f.write(f"{pid}\n")
    print(f"Created BlendedMVG_list.txt with {len(pid_dirs)} projects")
    
    # Create a custom list for testing (first 5 projects)
    test_pids = pid_dirs[:5]
    with open(dataset_path / 'BlendedMVS_test_list.txt', 'w') as f:
        for pid in test_pids:
            f.write(f"{pid}\n")
    print(f"Created BlendedMVS_test_list.txt with {len(test_pids)} projects for testing")
    
    return pid_dirs


def check_dataset_structure(dataset_root: str):
    """
    Check the dataset structure and report any issues
    
    Args:
        dataset_root: Root directory of the BlendedMVS dataset
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        print(f"❌ Dataset root directory not found: {dataset_root}")
        return False
    
    print(f"📁 Checking dataset structure at: {dataset_root}")
    
    # Check for any PID directories
    pid_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('PID')]
    
    if not pid_dirs:
        print("❌ No PID directories found!")
        print("\nExpected structure:")
        print("BlendedMVS_dataset/")
        print("├── PID0/")
        print("│   ├── blended_images/")
        print("│   ├── cams/")
        print("│   └── rendered_depth_maps/")
        print("├── PID1/")
        print("└── ...")
        return False
    
    print(f"✅ Found {len(pid_dirs)} PID directories")
    
    # Check the structure of the first PID
    sample_pid = pid_dirs[0]
    print(f"\n📂 Checking structure of {sample_pid.name}:")
    
    required_dirs = ['blended_images', 'cams', 'rendered_depth_maps']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = sample_pid / dir_name
        if dir_path.exists():
            # Count files
            files = list(dir_path.iterdir())
            print(f"  ✅ {dir_name}: {len(files)} files")
        else:
            print(f"  ❌ {dir_name}: Missing!")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories in {sample_pid.name}: {', '.join(missing_dirs)}")
        return False
    
    # Check for important files
    pair_file = sample_pid / 'cams' / 'pair.txt'
    if pair_file.exists():
        print(f"  ✅ pair.txt found")
    else:
        print(f"  ❌ pair.txt missing!")
    
    # Check for camera files
    cam_files = list((sample_pid / 'cams').glob('*_cam.txt'))
    print(f"  ✅ Found {len(cam_files)} camera files")
    
    # Check for images
    images = list((sample_pid / 'blended_images').glob('*.jpg'))
    masked_images = [img for img in images if '_masked' in img.name]
    regular_images = [img for img in images if '_masked' not in img.name]
    
    print(f"  ✅ Found {len(regular_images)} regular images")
    print(f"  ✅ Found {len(masked_images)} masked images")
    
    return True


def create_sample_project_info(dataset_root: str, project_id: str):
    """
    Create a JSON file with information about a specific project
    
    Args:
        dataset_root: Root directory of the BlendedMVS dataset
        project_id: Project ID to analyze
    """
    dataset_path = Path(dataset_root)
    project_path = dataset_path / project_id
    
    if not project_path.exists():
        print(f"❌ Project {project_id} not found!")
        return
    
    info = {
        'project_id': project_id,
        'path': str(project_path),
        'images': {},
        'cameras': {},
        'statistics': {}
    }
    
    # Count files
    images = list((project_path / 'blended_images').glob('*.jpg'))
    regular_images = [img for img in images if '_masked' not in img.name]
    masked_images = [img for img in images if '_masked' in img.name]
    
    info['statistics']['num_images'] = len(regular_images)
    info['statistics']['num_masked_images'] = len(masked_images)
    
    # Get camera count
    cam_files = list((project_path / 'cams').glob('*_cam.txt'))
    info['statistics']['num_cameras'] = len(cam_files)
    
    # Get depth map count
    depth_files = list((project_path / 'rendered_depth_maps').glob('*.pfm'))
    info['statistics']['num_depth_maps'] = len(depth_files)
    
    # Sample image info
    if regular_images:
        sample_img = regular_images[0]
        info['images']['sample'] = sample_img.name
        info['images']['format'] = 'XXXXXXXX.jpg (8-digit ID)'
    
    # Save info
    info_file = dataset_path / f'{project_id}_info.json'
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Created project info file: {info_file}")
    print(f"   - Images: {info['statistics']['num_images']}")
    print(f"   - Cameras: {info['statistics']['num_cameras']}")
    print(f"   - Depth maps: {info['statistics']['num_depth_maps']}")


def main():
    parser = argparse.ArgumentParser(description="BlendedMVS Setup Helper")
    parser.add_argument("dataset_root", help="Root directory of BlendedMVS dataset")
    parser.add_argument("--generate-lists", action="store_true",
                       help="Generate project list files")
    parser.add_argument("--check-structure", action="store_true",
                       help="Check dataset structure")
    parser.add_argument("--project-info", type=str,
                       help="Generate info for specific project (e.g., PID0)")
    parser.add_argument("--all", action="store_true",
                       help="Run all checks and generate all files")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_root)
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory not found: {args.dataset_root}")
        print("\nPlease ensure you have:")
        print("1. Downloaded the BlendedMVS dataset from OneDrive")
        print("2. Extracted it to the correct location")
        print("3. Provided the correct path to this script")
        return 1
    
    print(f"🔍 BlendedMVS Setup Helper")
    print(f"📁 Dataset location: {dataset_path.absolute()}")
    print()
    
    # Run requested operations
    if args.all or args.check_structure:
        print("=" * 60)
        print("Checking dataset structure...")
        print("=" * 60)
        if not check_dataset_structure(args.dataset_root):
            return 1
        print()
    
    if args.all or args.generate_lists:
        print("=" * 60)
        print("Generating project lists...")
        print("=" * 60)
        pid_dirs = generate_project_lists(args.dataset_root)
        print()
        
        if pid_dirs and (args.all or args.project_info):
            # Generate info for first project as sample
            print("=" * 60)
            print("Generating sample project info...")
            print("=" * 60)
            create_sample_project_info(args.dataset_root, pid_dirs[0])
    
    elif args.project_info:
        print("=" * 60)
        print(f"Generating info for {args.project_info}...")
        print("=" * 60)
        create_sample_project_info(args.dataset_root, args.project_info)
    
    print("\n✅ Setup helper completed!")
    print("\nNext steps:")
    print("1. If project lists were generated, you can now use them with the pipeline")
    print("2. Start with a test run using BlendedMVS_test_list.txt")
    print("3. Run the full pipeline on individual projects or in batch mode")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
