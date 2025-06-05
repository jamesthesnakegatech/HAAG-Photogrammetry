#!/usr/bin/env python3
"""
NeuS2 + SuGaR Comparison Pipeline
Runs both methods on BlendedMVS data and compares results
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime


class NeuS2Pipeline:
    """Pipeline for NeuS2 processing"""
    
    def __init__(self, dataset_dir: str = ".", neus2_path: str = "./NeuS2"):
        self.dataset_dir = Path(dataset_dir)
        self.neus2_path = Path(neus2_path)
        self.output_dir = self.dataset_dir / "output_neus2"
        self.output_dir.mkdir(exist_ok=True)
        
    def setup_neus2(self):
        """Setup NeuS2 environment"""
        if not self.neus2_path.exists():
            print("🔧 Setting up NeuS2...")
            
            # Clone NeuS2
            subprocess.run([
                "git", "clone",
                "https://github.com/19reborn/NeuS2.git"
            ], check=True)
            
            # Install dependencies
            os.chdir(self.neus2_path)
            
            # Create conda environment
            print("📦 Creating NeuS2 conda environment...")
            subprocess.run([
                "conda", "create", "-n", "neus2", "python=3.8", "-y"
            ], check=True)
            
            # Install requirements
            install_cmd = """
            conda activate neus2 && \
            pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html && \
            pip install -r requirements.txt && \
            pip install ninja imageio PyMCubes trimesh pymeshlab
            """
            
            subprocess.run(install_cmd, shell=True, check=True)
            
            print("✅ NeuS2 setup complete")
            
    def convert_blendedmvs_to_neus2(self, scene_id: str) -> Path:
        """Convert BlendedMVS format to NeuS2 format"""
        print(f"🔄 Converting {scene_id} to NeuS2 format...")
        
        scene_path = self.dataset_dir / scene_id
        neus2_data_path = self.neus2_path / "data" / scene_id
        neus2_data_path.mkdir(parents=True, exist_ok=True)
        
        # Create images directory
        images_out = neus2_data_path / "images"
        images_out.mkdir(exist_ok=True)
        
        # Copy images (non-masked only)
        blended_images = scene_path / "blended_images"
        for img in blended_images.glob("*.jpg"):
            if "_masked" not in img.name:
                shutil.copy2(img, images_out / img.name)
                
        # Convert camera parameters to NeuS2 format
        self._convert_cameras_to_neus2(scene_id, neus2_data_path)
        
        return neus2_data_path
        
    def _convert_cameras_to_neus2(self, scene_id: str, output_path: Path):
        """Convert BlendedMVS cameras to NeuS2 format"""
        scene_path = self.dataset_dir / scene_id
        cams_path = scene_path / "cams"
        
        # Create cameras.npz file for NeuS2
        camera_dict = {}
        
        # Get all camera files
        cam_files = sorted(cams_path.glob("*_cam.txt"))
        
        for i, cam_file in enumerate(cam_files):
            cam_id = int(cam_file.stem.split('_')[0])
            
            with open(cam_file, 'r') as f:
                lines = f.readlines()
                
            # Parse extrinsics
            extrinsics = np.array([
                [float(x) for x in lines[1].split()],
                [float(x) for x in lines[2].split()],
                [float(x) for x in lines[3].split()],
                [float(x) for x in lines[4].split()]
            ])
            
            # Parse intrinsics
            intrinsics = np.array([
                [float(x) for x in lines[7].split()],
                [float(x) for x in lines[8].split()],
                [float(x) for x in lines[9].split()]
            ])
            
            # NeuS2 expects world_mat and scale_mat
            camera_dict[f'world_mat_{i}'] = extrinsics
            camera_dict[f'scale_mat_{i}'] = np.eye(4)  # Identity scale
            
        # Save cameras
        np.savez(output_path / "cameras.npz", **camera_dict)
        
        # Create camera config
        config = {
            "n_images": len(cam_files),
            "image_width": 768,
            "image_height": 576,
            "scale_mat_scale": 1.0
        }
        
        with open(output_path / "camera_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
    def run_neus2(self, scene_id: str, iterations: int = 50000):
        """Run NeuS2 reconstruction"""
        print(f"🚀 Running NeuS2 for {scene_id}...")
        
        # Convert data format
        data_path = self.convert_blendedmvs_to_neus2(scene_id)
        
        # Prepare config
        config_path = self._create_neus2_config(scene_id, data_path, iterations)
        
        # Run NeuS2
        os.chdir(self.neus2_path)
        
        start_time = time.time()
        
        cmd = [
            "python", "train.py",
            "--conf", str(config_path),
            "--case", scene_id,
            "--mode", "train"
        ]
        
        # Run in conda environment
        conda_cmd = f"conda activate neus2 && {' '.join(cmd)}"
        subprocess.run(conda_cmd, shell=True, check=True)
        
        training_time = time.time() - start_time
        
        # Extract mesh
        print("📦 Extracting mesh from NeuS2...")
        extract_cmd = [
            "python", "train.py",
            "--conf", str(config_path),
            "--case", scene_id,
            "--mode", "validate_mesh",
            "--resolution", "512"
        ]
        
        conda_extract_cmd = f"conda activate neus2 && {' '.join(extract_cmd)}"
        subprocess.run(conda_extract_cmd, shell=True, check=True)
        
        # Copy results
        neus2_exp_path = self.neus2_path / "exp" / scene_id
        output_path = self.output_dir / scene_id
        output_path.mkdir(exist_ok=True)
        
        # Find and copy mesh
        mesh_files = list(neus2_exp_path.glob("**/*.ply"))
        if mesh_files:
            shutil.copy2(mesh_files[0], output_path / "mesh.ply")
            
        # Save timing info
        with open(output_path / "timing.json", 'w') as f:
            json.dump({"training_time": training_time}, f)
            
        print(f"✅ NeuS2 complete for {scene_id} (Time: {training_time/60:.1f} min)")
        
        return output_path
        
    def _create_neus2_config(self, scene_id: str, data_path: Path, iterations: int) -> Path:
        """Create NeuS2 configuration file"""
        config = {
            "general": {
                "base_exp_dir": f"./exp/{scene_id}",
                "data_dir": str(data_path),
                "resolution": 512
            },
            "train": {
                "learning_rate": 5e-4,
                "num_iterations": iterations,
                "warm_up_iter": 1000,
                "batch_size": 512
            },
            "model": {
                "sdf_network": {
                    "d_out": 257,
                    "d_in": 3,
                    "d_hidden": 256,
                    "n_layers": 8,
                    "skip_in": [4],
                    "bias": 0.5,
                    "scale": 1.0
                }
            }
        }
        
        config_path = self.neus2_path / f"conf/{scene_id}.conf"
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return config_path


class ComparisonPipeline:
    """Pipeline for comparing NeuS2 and SuGaR results"""
    
    def __init__(self, dataset_dir: str = "."):
        self.dataset_dir = Path(dataset_dir)
        self.comparison_dir = self.dataset_dir / "comparison_results"
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Initialize sub-pipelines
        from blendedmvs_sugar_pipeline import BlendedMVSSuGaRPipeline
        self.sugar_pipeline = BlendedMVSSuGaRPipeline(dataset_dir)
        self.neus2_pipeline = NeuS2Pipeline(dataset_dir)
        
    def setup_all(self):
        """Setup both pipelines"""
        print("🔧 Setting up all pipelines...")
        self.sugar_pipeline.setup_sugar()
        self.neus2_pipeline.setup_neus2()
        
    def process_scene_both_methods(self, scene_id: str, quality: str = "medium"):
        """Process a scene with both methods"""
        print(f"\n{'='*60}")
        print(f"Processing {scene_id} with both methods")
        print(f"{'='*60}\n")
        
        results = {
            "scene_id": scene_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process with SuGaR
        print("\n📌 Running SuGaR...")
        sugar_start = time.time()
        try:
            self.sugar_pipeline.process_scene(scene_id, quality)
            sugar_time = time.time() - sugar_start
            results["sugar"] = {
                "status": "success",
                "time": sugar_time,
                "output_path": str(self.dataset_dir / "output" / scene_id)
            }
        except Exception as e:
            results["sugar"] = {
                "status": "failed",
                "error": str(e)
            }
            
        # Process with NeuS2
        print("\n📌 Running NeuS2...")
        neus2_start = time.time()
        try:
            iterations = {"fast": 20000, "medium": 50000, "high": 100000}[quality]
            self.neus2_pipeline.run_neus2(scene_id, iterations)
            neus2_time = time.time() - neus2_start
            results["neus2"] = {
                "status": "success",
                "time": neus2_time,
                "output_path": str(self.dataset_dir / "output_neus2" / scene_id)
            }
        except Exception as e:
            results["neus2"] = {
                "status": "failed",
                "error": str(e)
            }
            
        # Save results
        with open(self.comparison_dir / f"{scene_id}_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    def analyze_comparison(self, scene_id: str) -> Dict:
        """Analyze and compare results from both methods"""
        print(f"\n📊 Analyzing comparison for {scene_id}...")
        
        comparison = {
            "scene_id": scene_id,
            "metrics": {}
        }
        
        # Load meshes
        sugar_mesh_path = self.dataset_dir / "output" / scene_id / "sugar_output" / "mesh"
        neus2_mesh_path = self.dataset_dir / "output_neus2" / scene_id / "mesh.ply"
        
        try:
            import trimesh
            
            # Find SuGaR mesh
            sugar_meshes = list(sugar_mesh_path.glob("*.obj")) if sugar_mesh_path.exists() else []
            if sugar_meshes:
                sugar_mesh = trimesh.load(sugar_meshes[0])
                comparison["sugar_mesh"] = self._analyze_mesh(sugar_mesh)
            
            # Load NeuS2 mesh
            if neus2_mesh_path.exists():
                neus2_mesh = trimesh.load(neus2_mesh_path)
                comparison["neus2_mesh"] = self._analyze_mesh(neus2_mesh)
                
            # Compare meshes
            if sugar_meshes and neus2_mesh_path.exists():
                comparison["cross_comparison"] = self._compare_meshes(sugar_mesh, neus2_mesh)
                
        except ImportError:
            print("⚠️  trimesh not installed")
            
        # Load timing information
        sugar_timing = self._load_timing("sugar", scene_id)
        neus2_timing = self._load_timing("neus2", scene_id)
        
        comparison["timing"] = {
            "sugar": sugar_timing,
            "neus2": neus2_timing
        }
        
        # Save comparison
        with open(self.comparison_dir / f"{scene_id}_analysis.json", 'w') as f:
            json.dump(comparison, f, indent=2)
            
        return comparison
        
    def _analyze_mesh(self, mesh) -> Dict:
        """Analyze mesh properties"""
        return {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": mesh.is_watertight,
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area),
            "bounds": {
                "min": mesh.bounds[0].tolist(),
                "max": mesh.bounds[1].tolist()
            }
        }
        
    def _compare_meshes(self, mesh1, mesh2) -> Dict:
        """Compare two meshes"""
        # Sample points from both meshes
        points1, _ = trimesh.sample.sample_surface(mesh1, 10000)
        points2, _ = trimesh.sample.sample_surface(mesh2, 10000)
        
        # Compute Chamfer distance
        from scipy.spatial import cKDTree
        
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        dist1, _ = tree1.query(points2)
        dist2, _ = tree2.query(points1)
        
        chamfer_dist = np.mean(dist1) + np.mean(dist2)
        
        return {
            "chamfer_distance": float(chamfer_dist),
            "vertex_ratio": len(mesh1.vertices) / len(mesh2.vertices),
            "face_ratio": len(mesh1.faces) / len(mesh2.faces),
            "volume_ratio": mesh1.volume / mesh2.volume if mesh1.is_watertight and mesh2.is_watertight else None
        }
        
    def _load_timing(self, method: str, scene_id: str) -> Optional[float]:
        """Load timing information"""
        if method == "sugar":
            # Estimate from process time
            return None  # Would need to track this
        else:
            timing_file = self.dataset_dir / "output_neus2" / scene_id / "timing.json"
            if timing_file.exists():
                with open(timing_file, 'r') as f:
                    return json.load(f)["training_time"]
        return None
        
    def create_comparison_notebook(self):
        """Create Jupyter notebook for comparison"""
        notebook_content = '''# NeuS2 vs SuGaR Comparison Notebook

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# Load comparison results
comparison_dir = Path("comparison_results")
results = []

for file in comparison_dir.glob("*_analysis.json"):
    with open(file, 'r') as f:
        results.append(json.load(f))

# Create comparison DataFrame
data = []
for r in results:
    row = {"scene_id": r["scene_id"]}
    
    if "sugar_mesh" in r:
        row.update({f"sugar_{k}": v for k, v in r["sugar_mesh"].items()})
    if "neus2_mesh" in r:
        row.update({f"neus2_{k}": v for k, v in r["neus2_mesh"].items()})
    if "cross_comparison" in r:
        row.update(r["cross_comparison"])
        
    data.append(row)

df = pd.DataFrame(data)

# Visualization 1: Mesh Quality Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('NeuS2 vs SuGaR Mesh Quality Comparison', fontsize=16)

# Vertex count
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['sugar_vertices'], width, label='SuGaR')
ax.bar(x + width/2, df['neus2_vertices'], width, label='NeuS2')
ax.set_xlabel('Scene')
ax.set_ylabel('Vertex Count')
ax.set_title('Vertex Count Comparison')
ax.legend()

# Surface area
ax = axes[0, 1]
ax.bar(x - width/2, df['sugar_surface_area'], width, label='SuGaR')
ax.bar(x + width/2, df['neus2_surface_area'], width, label='NeuS2')
ax.set_xlabel('Scene')
ax.set_ylabel('Surface Area')
ax.set_title('Surface Area Comparison')
ax.legend()

# Watertight status
ax = axes[1, 0]
sugar_watertight = df['sugar_watertight'].sum()
neus2_watertight = df['neus2_watertight'].sum()
ax.bar(['SuGaR', 'NeuS2'], [sugar_watertight, neus2_watertight])
ax.set_ylabel('Number of Watertight Meshes')
ax.set_title('Watertight Mesh Count')

# Chamfer distance
ax = axes[1, 1]
ax.bar(x, df['chamfer_distance'])
ax.set_xlabel('Scene')
ax.set_ylabel('Chamfer Distance')
ax.set_title('Cross-Method Chamfer Distance')

plt.tight_layout()
plt.show()

# Timing comparison
if 'timing' in results[0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sugar_times = [r['timing'].get('sugar', 0) for r in results]
    neus2_times = [r['timing'].get('neus2', 0) for r in results]
    
    ax.bar(x - width/2, sugar_times, width, label='SuGaR')
    ax.bar(x + width/2, neus2_times, width, label='NeuS2')
    ax.set_xlabel('Scene')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Processing Time Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Summary statistics
print("Summary Statistics:")
print("-" * 50)
print(f"Average Chamfer Distance: {df['chamfer_distance'].mean():.4f}")
print(f"Average Vertex Ratio (SuGaR/NeuS2): {df['vertex_ratio'].mean():.2f}")
print(f"Average Face Ratio (SuGaR/NeuS2): {df['face_ratio'].mean():.2f}")
'''
        
        with open(self.comparison_dir / "comparison_notebook.py", 'w') as f:
            f.write(notebook_content)
            
        print(f"✅ Created comparison notebook: {self.comparison_dir / 'comparison_notebook.py'}")


def main():
    parser = argparse.ArgumentParser(description="NeuS2 + SuGaR Comparison Pipeline")
    parser.add_argument("--setup", action="store_true", help="Setup both methods")
    parser.add_argument("--process", type=str, help="Process specific scene with both methods")
    parser.add_argument("--analyze", type=str, help="Analyze comparison for specific scene")
    parser.add_argument("--batch", type=str, help="Batch process from list file")
    parser.add_argument("--quality", default="medium", 
                       choices=["fast", "medium", "high"],
                       help="Processing quality")
    parser.add_argument("--create-notebook", action="store_true",
                       help="Create comparison notebook")
    
    args = parser.parse_args()
    
    pipeline = ComparisonPipeline()
    
    if args.setup:
        pipeline.setup_all()
        
    if args.process:
        pipeline.process_scene_both_methods(args.process, args.quality)
        pipeline.analyze_comparison(args.process)
        
    if args.analyze:
        pipeline.analyze_comparison(args.analyze)
        
    if args.batch:
        with open(args.batch, 'r') as f:
            scene_list = [line.strip() for line in f if line.strip()]
            
        for scene_id in scene_list:
            try:
                pipeline.process_scene_both_methods(scene_id, args.quality)
                pipeline.analyze_comparison(scene_id)
            except Exception as e:
                print(f"❌ Error processing {scene_id}: {e}")
                
    if args.create_notebook:
        pipeline.create_comparison_notebook()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
