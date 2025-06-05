#!/usr/bin/env python3
"""
SuGaR 3D Reconstruction Pipeline
This pipeline helps you process your data through the SuGaR framework
for 3D mesh reconstruction from images/videos.
"""

import os
import subprocess
import argparse
from pathlib import Path
import json
import shutil
from typing import Optional, List, Dict


class SuGaRPipeline:
    """Complete pipeline for SuGaR 3D reconstruction"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "./output",
                 sugar_path: str = "./SuGaR"):
        """
        Initialize the SuGaR pipeline
        
        Args:
            data_path: Path to input data (images directory or video file)
            output_dir: Output directory for results
            sugar_path: Path to cloned SuGaR repository
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.sugar_path = Path(sugar_path)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths for different stages
        self.colmap_dir = self.output_dir / "colmap_data"
        self.images_dir = self.colmap_dir / "input"
        self.gs_checkpoint_dir = self.output_dir / "gaussian_splatting"
        self.sugar_checkpoint_dir = self.output_dir / "sugar_model"
        
    def setup_environment(self):
        """Clone and setup SuGaR repository"""
        print("🔧 Setting up SuGaR environment...")
        
        # Clone SuGaR if not already present
        if not self.sugar_path.exists():
            subprocess.run([
                "git", "clone", 
                "https://github.com/Anttwo/SuGaR.git", 
                "--recursive"
            ], check=True)
            
        # Run installation script
        os.chdir(self.sugar_path)
        subprocess.run(["python", "install.py"], check=True)
        print("✅ Environment setup complete!")
        
    def prepare_images(self, fps: int = 2):
        """
        Prepare images from video or copy existing images
        
        Args:
            fps: Frames per second to extract from video
        """
        print("📸 Preparing images...")
        
        # Create images directory
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        if self.data_path.is_file() and self.data_path.suffix in ['.mp4', '.avi', '.mov', '.mkv']:
            # Extract frames from video
            print(f"Extracting frames from video at {fps} FPS...")
            subprocess.run([
                "ffmpeg", "-i", str(self.data_path),
                "-qscale:v", "1", "-qmin", "1",
                "-vf", f"fps={fps}",
                str(self.images_dir / "%04d.jpg")
            ], check=True)
            
        elif self.data_path.is_dir():
            # Copy images from directory
            print("Copying images from directory...")
            for img in self.data_path.glob("*"):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, self.images_dir)
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
            
        # Count images
        num_images = len(list(self.images_dir.glob("*")))
        print(f"✅ Prepared {num_images} images")
        
    def run_colmap(self, skip_matching: bool = False):
        """
        Run COLMAP to estimate camera poses
        
        Args:
            skip_matching: Skip feature matching if already done
        """
        print("📷 Running COLMAP for camera pose estimation...")
        
        os.chdir(self.sugar_path)
        
        cmd = [
            "python", "gaussian_splatting/convert.py",
            "-s", str(self.colmap_dir)
        ]
        
        if skip_matching:
            cmd.append("--skip_matching")
            
        subprocess.run(cmd, check=True)
        print("✅ COLMAP reconstruction complete!")
        
    def check_colmap_models(self) -> bool:
        """
        Check if COLMAP produced multiple models and handle accordingly
        
        Returns:
            True if manual intervention is needed
        """
        sparse_dir = self.colmap_dir / "distorted" / "sparse"
        if not sparse_dir.exists():
            return False
            
        models = list(sparse_dir.iterdir())
        if len(models) > 1:
            print("⚠️  Multiple COLMAP models detected!")
            print("Please check the following directories and keep only the largest model:")
            for model in models:
                print(f"  - {model}")
            print("\nRename the largest model directory to '0' and run with --skip-colmap-matching")
            return True
            
        return False
        
    def run_sugar_pipeline(self, 
                          regularization: str = "dn_consistency",
                          poly_level: str = "high",
                          refinement_time: str = "medium",
                          export_obj: bool = True,
                          export_ply: bool = True,
                          from_gs_checkpoint: Optional[str] = None):
        """
        Run the complete SuGaR optimization pipeline
        
        Args:
            regularization: Type of regularization ("dn_consistency", "density", or "sdf")
            poly_level: Mesh resolution ("high" for 1M vertices, "low" for 200k)
            refinement_time: Refinement duration ("short", "medium", or "long")
            export_obj: Export textured mesh as OBJ
            export_ply: Export refined Gaussians as PLY
            from_gs_checkpoint: Optional path to existing Gaussian Splatting checkpoint
        """
        print(f"🚀 Running SuGaR pipeline with {regularization} regularization...")
        
        os.chdir(self.sugar_path)
        
        cmd = [
            "python", "train_full_pipeline.py",
            "-s", str(self.colmap_dir),
            "-r", regularization,
            "--refinement_time", refinement_time,
            f"--{poly_level}_poly", "True"
        ]
        
        if export_obj:
            cmd.extend(["--export_obj", "True"])
            
        if export_ply:
            cmd.extend(["--export_ply", "True"])
            
        if from_gs_checkpoint:
            cmd.extend(["--gs_output_dir", from_gs_checkpoint])
            
        subprocess.run(cmd, check=True)
        print("✅ SuGaR optimization complete!")
        
    def extract_results(self):
        """Extract and organize results"""
        print("📦 Extracting results...")
        
        # Copy results to organized output directory
        results_dir = self.output_dir / "final_results"
        results_dir.mkdir(exist_ok=True)
        
        # Find and copy mesh files
        sugar_output = self.sugar_path / "output"
        
        # Copy refined mesh
        mesh_dir = sugar_output / "refined_mesh"
        if mesh_dir.exists():
            shutil.copytree(mesh_dir, results_dir / "mesh", dirs_exist_ok=True)
            
        # Copy PLY files
        ply_dir = sugar_output / "refined_ply"
        if ply_dir.exists():
            shutil.copytree(ply_dir, results_dir / "gaussians", dirs_exist_ok=True)
            
        print(f"✅ Results saved to {results_dir}")
        
    def run_viewer(self, ply_path: Optional[str] = None):
        """
        Launch the SuGaR viewer
        
        Args:
            ply_path: Path to PLY file to visualize
        """
        if not ply_path:
            # Find the most recent PLY file
            ply_files = list((self.sugar_path / "output" / "refined_ply").rglob("*.ply"))
            if ply_files:
                ply_path = str(ply_files[-1])
            else:
                print("❌ No PLY file found!")
                return
                
        print(f"🖥️  Launching viewer for {ply_path}")
        os.chdir(self.sugar_path)
        subprocess.run(["python", "run_viewer.py", "-p", ply_path])
        
    def create_config(self, config_path: str = "pipeline_config.json"):
        """Create a configuration file for the pipeline"""
        config = {
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "sugar_path": str(self.sugar_path),
            "colmap_dir": str(self.colmap_dir),
            "pipeline_settings": {
                "regularization": "dn_consistency",
                "poly_level": "high",
                "refinement_time": "medium",
                "export_obj": True,
                "export_ply": True,
                "video_fps": 2
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"📝 Configuration saved to {config_path}")
        

def main():
    parser = argparse.ArgumentParser(description="SuGaR 3D Reconstruction Pipeline")
    parser.add_argument("data_path", help="Path to input data (images directory or video file)")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--sugar-path", default="./SuGaR", help="Path to SuGaR repository")
    parser.add_argument("--regularization", default="dn_consistency", 
                       choices=["dn_consistency", "density", "sdf"],
                       help="Regularization method")
    parser.add_argument("--poly-level", default="high", choices=["high", "low"],
                       help="Mesh resolution")
    parser.add_argument("--refinement-time", default="medium", 
                       choices=["short", "medium", "long"],
                       help="Refinement duration")
    parser.add_argument("--video-fps", type=int, default=2,
                       help="FPS for video frame extraction")
    parser.add_argument("--skip-setup", action="store_true",
                       help="Skip environment setup")
    parser.add_argument("--skip-colmap", action="store_true",
                       help="Skip COLMAP if already done")
    parser.add_argument("--skip-colmap-matching", action="store_true",
                       help="Skip COLMAP matching (use if fixing multiple models)")
    parser.add_argument("--from-gs-checkpoint", type=str,
                       help="Path to existing Gaussian Splatting checkpoint")
    parser.add_argument("--view-results", action="store_true",
                       help="Launch viewer after completion")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SuGaRPipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sugar_path=args.sugar_path
    )
    
    try:
        # Setup environment
        if not args.skip_setup:
            pipeline.setup_environment()
        
        # Prepare images
        pipeline.prepare_images(fps=args.video_fps)
        
        # Run COLMAP
        if not args.skip_colmap:
            pipeline.run_colmap(skip_matching=args.skip_colmap_matching)
            
            # Check for multiple models
            if pipeline.check_colmap_models():
                print("\n⚠️  Please resolve the multiple models issue and re-run with --skip-colmap-matching")
                return
        
        # Run SuGaR pipeline
        pipeline.run_sugar_pipeline(
            regularization=args.regularization,
            poly_level=args.poly_level,
            refinement_time=args.refinement_time,
            from_gs_checkpoint=args.from_gs_checkpoint
        )
        
        # Extract results
        pipeline.extract_results()
        
        # Save configuration
        pipeline.create_config()
        
        # Launch viewer if requested
        if args.view_results:
            pipeline.run_viewer()
            
        print("\n🎉 Pipeline completed successfully!")
        print(f"Results are in: {pipeline.output_dir / 'final_results'}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
