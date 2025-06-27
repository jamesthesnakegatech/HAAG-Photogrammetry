"""
VGGT + NeuS2 Pipeline
Combines Visual Geometry Grounded Transformer with Neural Surface Reconstruction

Pipeline:
1. VGGT: Extract camera parameters, depth maps, and point clouds from input images
2. NeuS2: Use VGGT outputs for fast neural surface reconstruction
"""

import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import subprocess
import shutil

class VGGTNeuS2Pipeline:
    def __init__(self, vggt_path: str, neus2_path: str, output_dir: str = "output"):
        """
        Initialize the pipeline with paths to both repositories
        
        Args:
            vggt_path: Path to VGGT repository
            neus2_path: Path to NeuS2 repository
            output_dir: Output directory for results
        """
        self.vggt_path = Path(vggt_path)
        self.neus2_path = Path(neus2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def prepare_images(self, image_folder: str, mask_folder: Optional[str] = None) -> Dict:
        """
        Prepare input images for processing
        
        Args:
            image_folder: Folder containing input images
            mask_folder: Optional folder containing masks for each image
            
        Returns:
            Dictionary with image paths and metadata
        """
        image_folder = Path(image_folder)
        images = sorted(list(image_folder.glob("*.jpg")) + 
                       list(image_folder.glob("*.png")) + 
                       list(image_folder.glob("*.JPG")))
        
        data = {
            "images": [str(img) for img in images],
            "n_images": len(images),
            "image_folder": str(image_folder)
        }
        
        if mask_folder:
            mask_folder = Path(mask_folder)
            masks = sorted(list(mask_folder.glob("*.png")))
            data["masks"] = [str(mask) for mask in masks]
            
        return data
    
    def run_vggt(self, image_data: Dict) -> Dict:
        """
        Run VGGT to extract 3D attributes from images
        
        Args:
            image_data: Dictionary containing image paths
            
        Returns:
            Dictionary with VGGT outputs (cameras, depth maps, point clouds)
        """
        print("Running VGGT for 3D attribute extraction...")
        
        vggt_output = self.output_dir / "vggt_results"
        vggt_output.mkdir(exist_ok=True)
        
        # Prepare VGGT input
        vggt_input = {
            "images": image_data["images"],
            "output_dir": str(vggt_output)
        }
        
        # Save input configuration
        with open(vggt_output / "input_config.json", "w") as f:
            json.dump(vggt_input, f, indent=2)
        
        # Run VGGT inference
        vggt_script = f"""
import sys
sys.path.append('{self.vggt_path}')
import torch
from vggt import VGGT
import numpy as np
from PIL import Image
import json

# Load configuration
with open('{vggt_output}/input_config.json') as f:
    config = json.load(f)

# Initialize VGGT model
model = VGGT.from_pretrained('vggt-large')
model.eval()

# Process images
images = []
for img_path in config['images']:
    img = Image.open(img_path).convert('RGB')
    images.append(np.array(img))

# Run inference
with torch.no_grad():
    outputs = model(images)

# Extract results
cameras = outputs['cameras']  # Camera parameters
depths = outputs['depths']     # Depth maps
points = outputs['points']     # Point maps
tracks = outputs['tracks']     # 3D point tracks

# Save results
np.save('{vggt_output}/cameras.npy', cameras.cpu().numpy())
np.save('{vggt_output}/depths.npy', depths.cpu().numpy())
np.save('{vggt_output}/points.npy', points.cpu().numpy())
np.save('{vggt_output}/tracks.npy', tracks.cpu().numpy())

# Extract camera intrinsics and extrinsics
intrinsics = cameras[:, :9].reshape(-1, 3, 3)
extrinsics = cameras[:, 9:].reshape(-1, 4, 4)

np.save('{vggt_output}/intrinsics.npy', intrinsics.cpu().numpy())
np.save('{vggt_output}/extrinsics.npy', extrinsics.cpu().numpy())

print("VGGT processing complete!")
"""
        
        # Execute VGGT script
        script_path = vggt_output / "run_vggt.py"
        with open(script_path, "w") as f:
            f.write(vggt_script)
            
        subprocess.run([sys.executable, str(script_path)], check=True)
        
        # Load results
        results = {
            "cameras": np.load(vggt_output / "cameras.npy"),
            "depths": np.load(vggt_output / "depths.npy"),
            "points": np.load(vggt_output / "points.npy"),
            "tracks": np.load(vggt_output / "tracks.npy"),
            "intrinsics": np.load(vggt_output / "intrinsics.npy"),
            "extrinsics": np.load(vggt_output / "extrinsics.npy"),
        }
        
        return results
    
    def prepare_neus2_data(self, image_data: Dict, vggt_results: Dict) -> str:
        """
        Prepare data in NeuS2 format using VGGT outputs
        
        Args:
            image_data: Original image data
            vggt_results: Results from VGGT
            
        Returns:
            Path to NeuS2 data directory
        """
        print("Preparing data for NeuS2...")
        
        neus2_data = self.output_dir / "neus2_data"
        neus2_data.mkdir(exist_ok=True)
        
        # Copy images
        images_dir = neus2_data / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i, img_path in enumerate(image_data["images"]):
            dst = images_dir / f"{i:04d}.png"
            shutil.copy2(img_path, dst)
        
        # Copy masks if available
        if "masks" in image_data:
            masks_dir = neus2_data / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            for i, mask_path in enumerate(image_data["masks"]):
                dst = masks_dir / f"{i:04d}.png"
                shutil.copy2(mask_path, dst)
        
        # Create transform.json for NeuS2
        frames = []
        
        for i in range(len(image_data["images"])):
            # Get camera parameters from VGGT
            K = vggt_results["intrinsics"][i]
            RT = vggt_results["extrinsics"][i]
            
            # Extract focal length and principal point
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Convert to NeuS2 format
            frame = {
                "file_path": f"./images/{i:04d}.png",
                "transform_matrix": RT.tolist(),
                "fl_x": float(fx),
                "fl_y": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "w": image_data.get("width", 1920),
                "h": image_data.get("height", 1080)
            }
            
            if "masks" in image_data:
                frame["mask_path"] = f"./masks/{i:04d}.png"
            
            frames.append(frame)
        
        # Add depth maps from VGGT as additional supervision
        depths_dir = neus2_data / "depths"
        depths_dir.mkdir(exist_ok=True)
        
        for i, depth in enumerate(vggt_results["depths"]):
            # Normalize and save depth maps
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = (depth_normalized * 65535).astype(np.uint16)
            cv2.imwrite(str(depths_dir / f"{i:04d}.png"), depth_img)
            
            frames[i]["depth_path"] = f"./depths/{i:04d}.png"
        
        # Create transform.json
        transform_data = {
            "camera_model": "OPENCV",
            "frames": frames,
            "from_na": True,
            "has_mask": "masks" in image_data,
            "has_depth": True,  # We have depth from VGGT
            "aabb_scale": 16,  # Adjust based on scene scale
        }
        
        with open(neus2_data / "transform.json", "w") as f:
            json.dump(transform_data, f, indent=2)
        
        # Save point cloud from VGGT for initialization
        points = vggt_results["points"].reshape(-1, 3)
        np.savetxt(neus2_data / "points_init.txt", points)
        
        return str(neus2_data)
    
    def run_neus2(self, data_path: str, config: str = "dtu.json", n_steps: int = 15000) -> Dict:
        """
        Run NeuS2 for surface reconstruction
        
        Args:
            data_path: Path to prepared NeuS2 data
            config: Configuration file name
            n_steps: Number of training steps
            
        Returns:
            Dictionary with NeuS2 outputs
        """
        print(f"Running NeuS2 surface reconstruction for {n_steps} steps...")
        
        experiment_name = "vggt_neus2_experiment"
        
        # Run NeuS2 training
        cmd = [
            sys.executable,
            str(self.neus2_path / "scripts" / "run.py"),
            "--scene", f"{data_path}/transform.json",
            "--name", experiment_name,
            "--network", config,
            "--n_steps", str(n_steps)
        ]
        
        subprocess.run(cmd, check=True, cwd=str(self.neus2_path))
        
        # Get output path
        neus2_output = self.neus2_path / "output" / experiment_name
        
        results = {
            "mesh_path": str(neus2_output / "mesh.ply"),
            "checkpoint_path": str(neus2_output / "checkpoints"),
            "logs_path": str(neus2_output / "logs"),
            "experiment_name": experiment_name
        }
        
        # Copy results to pipeline output
        final_output = self.output_dir / "final_results"
        final_output.mkdir(exist_ok=True)
        
        if (neus2_output / "mesh.ply").exists():
            shutil.copy2(neus2_output / "mesh.ply", final_output / "mesh.ply")
            
        return results
    
    def run_pipeline(self, image_folder: str, mask_folder: Optional[str] = None,
                    neus2_config: str = "dtu.json", n_steps: int = 15000) -> Dict:
        """
        Run the complete VGGT + NeuS2 pipeline
        
        Args:
            image_folder: Folder containing input images
            mask_folder: Optional folder containing masks
            neus2_config: NeuS2 configuration file
            n_steps: Number of training steps for NeuS2
            
        Returns:
            Dictionary with all results
        """
        print("Starting VGGT + NeuS2 Pipeline...")
        
        # Step 1: Prepare images
        image_data = self.prepare_images(image_folder, mask_folder)
        print(f"Found {image_data['n_images']} images")
        
        # Step 2: Run VGGT
        vggt_results = self.run_vggt(image_data)
        print("VGGT processing complete")
        
        # Step 3: Prepare NeuS2 data
        neus2_data_path = self.prepare_neus2_data(image_data, vggt_results)
        print(f"NeuS2 data prepared at: {neus2_data_path}")
        
        # Step 4: Run NeuS2
        neus2_results = self.run_neus2(neus2_data_path, neus2_config, n_steps)
        print("NeuS2 reconstruction complete")
        
        # Compile final results
        final_results = {
            "vggt_results": vggt_results,
            "neus2_results": neus2_results,
            "output_dir": str(self.output_dir),
            "mesh_path": str(self.output_dir / "final_results" / "mesh.ply")
        }
        
        # Save summary
        with open(self.output_dir / "pipeline_summary.json", "w") as f:
            json.dump({
                "n_images": image_data["n_images"],
                "neus2_config": neus2_config,
                "n_steps": n_steps,
                "mesh_path": final_results["mesh_path"],
                "experiment_name": neus2_results["experiment_name"]
            }, f, indent=2)
        
        print(f"\nPipeline complete! Results saved to: {self.output_dir}")
        print(f"Final mesh: {final_results['mesh_path']}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description="VGGT + NeuS2 Pipeline")
    parser.add_argument("--images", required=True, help="Path to input images folder")
    parser.add_argument("--masks", help="Path to masks folder (optional)")
    parser.add_argument("--vggt_path", required=True, help="Path to VGGT repository")
    parser.add_argument("--neus2_path", required=True, help="Path to NeuS2 repository")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--config", default="dtu.json", help="NeuS2 config file")
    parser.add_argument("--steps", type=int, default=15000, help="Training steps")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VGGTNeuS2Pipeline(
        vggt_path=args.vggt_path,
        neus2_path=args.neus2_path,
        output_dir=args.output
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        image_folder=args.images,
        mask_folder=args.masks,
        neus2_config=args.config,
        n_steps=args.steps
    )
    
    return results


if __name__ == "__main__":
    main()
