#!/usr/bin/env python3
"""
OpenMVS Integration Pipeline with Hyperparameter Tuning
Adds OpenMVS to the existing SuGaR and NeuS2 comparison framework
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import pandas as pd
import trimesh


class OpenMVSPipeline:
    """Pipeline for OpenMVS processing with hyperparameter tuning"""
    
    def __init__(self, dataset_dir: str = "./mipnerf360", output_dir: str = "./mipnerf360_output"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.openmvs_dir = self.output_dir / "openmvs_results"
        self.openmvs_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenMVS hyperparameter space
        self.openmvs_hyperparam_space = {
            # Dense reconstruction parameters
            "resolution_level": [0, 1, 2],  # 0=full, 1=half, 2=quarter
            "number_views": [3, 5, 8],  # Min number of views for reconstruction
            "number_views_fuse": [2, 3, 4],  # Min views for fusion
            "max_resolution": [2048, 3072, 4096],  # Max image resolution
            
            # Depth map parameters
            "min_resolution": [320, 640, 960],  # Min resolution for depth
            "num_scales": [3, 5, 7],  # Number of resolution scales
            "scale_step": [0.5, 0.7, 0.9],  # Scale step between resolutions
            "confidence": [0.5, 0.7, 0.9],  # Min confidence for depth values
            "max_threads": [0, 8, 16],  # 0=auto
            
            # Dense point cloud parameters
            "geometric_iters": [0, 1, 2],  # Geometric consistency iterations
            "optimize": [0, 1, 7],  # Optimization level (bitmask)
            "densify": [0, 1],  # Densify point cloud
            "min_point_distance": [0.0, 0.5, 1.0],  # Min distance between points
            "estimate_colors": [0, 1],  # Estimate point colors
            "estimate_normals": [0, 1, 2],  # 0=no, 1=yes, 2=refine
            
            # Mesh reconstruction parameters
            "reconstruct_mesh": [0, 1],  # Use Delaunay or Poisson
            "smooth": [0, 1, 3, 5],  # Smoothing iterations
            "thickness_factor": [0.5, 1.0, 2.0],  # Thickness factor
            "quality_factor": [0.5, 1.0, 1.5],  # Quality factor
            "decimate": [0.0, 0.5, 0.9],  # Decimation (0=none, 1=all)
            
            # Mesh refinement parameters
            "scales": [1, 2, 3],  # Number of scales for refinement
            "gradient_step": [25.0, 45.0, 90.0],  # Gradient descent step
            "cuda_device": [-1, 0],  # -1=CPU, 0+=GPU
            
            # Scene-specific parameters
            "remove_spurious": [0, 20, 100],  # Remove small components
            "remove_spikes": [0, 1],  # Remove spike artifacts
            "close_holes": [0, 30, 100],  # Close holes up to N edges
            "smooth_mesh": [0, 1, 2],  # Final smoothing iterations
        }
        
    def install_openmvs(self):
        """Install OpenMVS if not already installed"""
        print("🔧 Installing OpenMVS...")
        
        install_script = """#!/bin/bash
# Install OpenMVS and dependencies

# Check if OpenMVS is already installed
if command -v DensifyPointCloud &> /dev/null; then
    echo "✅ OpenMVS already installed"
    exit 0
fi

echo "📦 Installing OpenMVS dependencies..."

# Install dependencies based on OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        cmake build-essential git \
        libpng-dev libjpeg-dev libtiff-dev \
        libglu1-mesa-dev libglew-dev libglfw3-dev \
        libatlas-base-dev libsuitesparse-dev \
        libboost-all-dev \
        libopencv-dev \
        libcgal-dev \
        libvtk7-dev \
        libceres-dev
        
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew install \
        cmake eigen boost \
        opencv cgal vtk \
        ceres-solver glew glfw
fi

# Clone and build OpenMVS
echo "📥 Cloning OpenMVS..."
git clone https://github.com/cdcseacave/openMVS.git --recursive
cd openMVS

# Create build directory
mkdir build && cd build

# Configure
echo "🔨 Building OpenMVS..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DVCG_ROOT="$PWD/../vcglib"

# Build
make -j$(nproc)

# Install
sudo make install

echo "✅ OpenMVS installation complete!"
"""
        
        script_path = self.output_dir / "install_openmvs.sh"
        with open(script_path, 'w') as f:
            f.write(install_script)
        script_path.chmod(0o755)
        
        # Run installation
        subprocess.run([str(script_path)], check=True)
        
    def convert_colmap_to_openmvs(self, scene_path: Path, output_path: Path):
        """Convert COLMAP format to OpenMVS format"""
        print(f"🔄 Converting COLMAP to OpenMVS format...")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create conversion script
        convert_script = f"""#!/usr/bin/env python3
import sys
import os
sys.path.append('/usr/local/bin/OpenMVS/python')

# Import OpenMVS Python bindings (if available)
# Otherwise use InterfaceCOLMAP tool

import subprocess

# Use InterfaceCOLMAP tool
cmd = [
    "InterfaceCOLMAP",
    "-i", "{scene_path}",
    "-o", "{output_path}/scene.mvs",
    "--image-folder", "{scene_path}/images"
]

subprocess.run(cmd, check=True)
"""
        
        script_path = output_path / "convert.py"
        with open(script_path, 'w') as f:
            f.write(convert_script)
            
        # Run conversion
        subprocess.run(["python3", str(script_path)], check=True)
        
        return output_path / "scene.mvs"
        
    def run_openmvs_with_hyperparams(self, scene_name: str, hyperparams: Dict) -> Dict:
        """Run OpenMVS with specific hyperparameters"""
        print(f"🚀 Running OpenMVS on {scene_name} with custom hyperparameters...")
        
        scene_path = self.dataset_dir / scene_name
        output_path = self.openmvs_dir / f"{scene_name}_{self._hyperparam_hash(hyperparams)}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "scene_path": str(scene_path),
            "output_path": str(output_path),
            "hyperparameters": hyperparams,
            "method": "openmvs"
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        start_time = time.time()
        
        try:
            # Convert COLMAP to OpenMVS format
            mvs_scene = self.convert_colmap_to_openmvs(scene_path, output_path)
            
            # 1. Dense point cloud reconstruction
            dense_cmd = [
                "DensifyPointCloud",
                "-i", str(mvs_scene),
                "-o", str(output_path / "dense.mvs"),
                "--resolution-level", str(hyperparams.get("resolution_level", 1)),
                "--number-views", str(hyperparams.get("number_views", 5)),
                "--max-resolution", str(hyperparams.get("max_resolution", 3072)),
                "--min-resolution", str(hyperparams.get("min_resolution", 640)),
                "--num-scales", str(hyperparams.get("num_scales", 5)),
                "--scale-step", str(hyperparams.get("scale_step", 0.7)),
                "--confidence", str(hyperparams.get("confidence", 0.7)),
                "--geometric-iters", str(hyperparams.get("geometric_iters", 1)),
                "--optimize", str(hyperparams.get("optimize", 7)),
                "--estimate-normals", str(hyperparams.get("estimate_normals", 1))
            ]
            
            if hyperparams.get("cuda_device", -1) >= 0:
                dense_cmd.extend(["--cuda-device", str(hyperparams["cuda_device"])])
                
            subprocess.run(dense_cmd, check=True)
            
            # 2. Mesh reconstruction
            mesh_cmd = [
                "ReconstructMesh",
                "-i", str(output_path / "dense.mvs"),
                "-o", str(output_path / "mesh.mvs"),
                "--smooth", str(hyperparams.get("smooth", 1)),
                "--thickness-factor", str(hyperparams.get("thickness_factor", 1.0)),
                "--quality-factor", str(hyperparams.get("quality_factor", 1.0)),
                "--decimate", str(hyperparams.get("decimate", 0.0)),
                "--remove-spurious", str(hyperparams.get("remove_spurious", 20)),
                "--close-holes", str(hyperparams.get("close_holes", 30))
            ]
            
            if hyperparams.get("remove_spikes", 0):
                mesh_cmd.append("--remove-spikes")
                
            subprocess.run(mesh_cmd, check=True)
            
            # 3. Mesh refinement
            if hyperparams.get("scales", 1) > 1:
                refine_cmd = [
                    "RefineMesh",
                    "-i", str(output_path / "mesh.mvs"),
                    "-o", str(output_path / "refined_mesh.mvs"),
                    "--scales", str(hyperparams.get("scales", 2)),
                    "--gradient-step", str(hyperparams.get("gradient_step", 45.0))
                ]
                
                if hyperparams.get("cuda_device", -1) >= 0:
                    refine_cmd.extend(["--cuda-device", str(hyperparams["cuda_device"])])
                    
                subprocess.run(refine_cmd, check=True)
                final_mesh = output_path / "refined_mesh.mvs"
            else:
                final_mesh = output_path / "mesh.mvs"
                
            # 4. Texture mesh (optional)
            if hyperparams.get("texture_mesh", True):
                texture_cmd = [
                    "TextureMesh",
                    "-i", str(final_mesh),
                    "-o", str(output_path / "textured_mesh.mvs"),
                    "--decimate", str(hyperparams.get("texture_decimate", 0.0)),
                    "--smooth", str(hyperparams.get("texture_smooth", 0))
                ]
                
                subprocess.run(texture_cmd, check=True)
                
            # Export to standard formats
            export_cmd = [
                "InterfaceMVS",
                "-i", str(final_mesh),
                "-o", str(output_path / "mesh.ply")
            ]
            subprocess.run(export_cmd, check=True)
            
            processing_time = time.time() - start_time
            
            # Evaluate results
            metrics = self.evaluate_reconstruction(output_path, scene_name)
            metrics["processing_time"] = processing_time
            metrics["hyperparameters"] = hyperparams
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"❌ OpenMVS failed for {scene_name}: {e}")
            return {"error": str(e), "hyperparameters": hyperparams}
            
    def evaluate_reconstruction(self, output_path: Path, scene_name: str) -> Dict:
        """Evaluate OpenMVS reconstruction quality"""
        metrics = {
            "scene": scene_name,
            "method": "openmvs"
        }
        
        # Find mesh file
        mesh_files = list(output_path.glob("*.ply"))
        if not mesh_files:
            mesh_files = list(output_path.glob("*.obj"))
            
        if not mesh_files:
            return {"error": "No mesh found"}
            
        try:
            mesh = trimesh.load(mesh_files[0])
            
            # Basic metrics
            metrics["vertices"] = len(mesh.vertices)
            metrics["faces"] = len(mesh.faces)
            metrics["watertight"] = mesh.is_watertight
            metrics["volume"] = float(mesh.volume) if mesh.is_watertight else 0
            metrics["surface_area"] = float(mesh.area)
            
            # Mesh quality metrics
            face_areas = mesh.area_faces
            metrics["face_area_std"] = float(np.std(face_areas))
            metrics["face_area_mean"] = float(np.mean(face_areas))
            metrics["degenerate_faces"] = int(np.sum(face_areas < 1e-6))
            
            # Vertex distribution
            vertex_neighbors = [len(mesh.vertex_neighbors[i]) for i in range(len(mesh.vertices))]
            metrics["vertex_degree_mean"] = float(np.mean(vertex_neighbors))
            metrics["vertex_degree_std"] = float(np.std(vertex_neighbors))
            
            # Scene extent
            bounds = mesh.bounds
            scene_extent = bounds[1] - bounds[0]
            metrics["scene_extent"] = scene_extent.tolist()
            metrics["max_extent"] = float(np.max(scene_extent))
            
            # Compute quality score
            metrics["quality_score"] = self._compute_quality_score(metrics)
            
            # OpenMVS-specific metrics
            if (output_path / "dense.mvs.log").exists():
                # Parse log for additional metrics
                with open(output_path / "dense.mvs.log", 'r') as f:
                    log_content = f.read()
                    # Extract point cloud size, processing stats, etc.
                    
        except Exception as e:
            metrics["error"] = str(e)
            
        return metrics
        
    def _compute_quality_score(self, metrics: Dict) -> float:
        """Compute quality score for OpenMVS reconstruction"""
        score = 100.0
        
        # Watertightness is important for OpenMVS
        if not metrics.get("watertight", False):
            score -= 15
            
        # Check vertex count
        ideal_vertices = 600000
        vertex_ratio = metrics["vertices"] / ideal_vertices
        if vertex_ratio < 0.3:
            score -= 25
        elif vertex_ratio > 3.0:
            score -= 10
            
        # Face quality
        if metrics.get("degenerate_faces", 0) > metrics["faces"] * 0.01:
            score -= 15
            
        # Vertex distribution quality
        ideal_degree = 6  # For well-distributed mesh
        degree_diff = abs(metrics.get("vertex_degree_mean", 6) - ideal_degree)
        score -= min(10, degree_diff * 2)
        
        # Scene completeness (based on extent)
        if metrics.get("max_extent", 0) < 5.0:
            score -= 10  # Might be incomplete
            
        return max(0, min(100, score))
        
    def _hyperparam_hash(self, hyperparams: Dict) -> str:
        """Create a hash for hyperparameter combination"""
        import hashlib
        param_str = json.dumps(hyperparams, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]


class UnifiedBenchmarkPipeline:
    """Unified pipeline for SuGaR, NeuS2, and OpenMVS comparison"""
    
    def __init__(self, dataset_dir: str = "./mipnerf360", output_dir: str = "./mipnerf360_output"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        # Import existing pipelines
        from mipnerf360_benchmark_pipeline import MipNeRF360Pipeline, MipNeRF360HyperparamOptimizer
        
        # Initialize all three pipelines
        self.sugar_neus2_pipeline = MipNeRF360Pipeline(dataset_dir, output_dir)
        self.openmvs_pipeline = OpenMVSPipeline(dataset_dir, output_dir)
        self.optimizer = MipNeRF360HyperparamOptimizer(self.sugar_neus2_pipeline)
        
        # Combined results directory
        self.comparison_dir = self.output_dir / "three_way_comparison"
        self.comparison_dir.mkdir(exist_ok=True)
        
    def run_three_way_comparison(self, scene_name: str, optimization: str = "grid"):
        """Run all three methods on a scene with hyperparameter optimization"""
        print(f"\n{'='*60}")
        print(f"Three-Way Comparison: {scene_name}")
        print(f"Methods: SuGaR, NeuS2, OpenMVS")
        print(f"{'='*60}\n")
        
        results = {}
        
        # 1. Run SuGaR
        print("\n📌 Running SuGaR...")
        if optimization == "grid":
            sugar_params = {
                "regularization_type": ["sdf", "dn_consistency"],
                "sh_degree": [3, 4],
                "refinement_iterations": [7000]
            }
            sugar_results = self._grid_search("sugar", scene_name, sugar_params)
        else:
            sugar_results = self.sugar_neus2_pipeline.run_sugar_with_hyperparams(scene_name, {})
            
        results["sugar"] = sugar_results
        
        # 2. Run NeuS2
        print("\n📌 Running NeuS2...")
        if optimization == "grid":
            neus2_params = {
                "learning_rate": [5e-4],
                "num_iterations": [50000],
                "n_samples": [128]
            }
            neus2_results = self._grid_search("neus2", scene_name, neus2_params)
        else:
            neus2_results = self.sugar_neus2_pipeline.run_neus2_with_hyperparams(scene_name, {})
            
        results["neus2"] = neus2_results
        
        # 3. Run OpenMVS
        print("\n📌 Running OpenMVS...")
        if optimization == "grid":
            openmvs_params = {
                "resolution_level": [0, 1],
                "number_views": [5],
                "smooth": [1, 3],
                "remove_spurious": [20, 50]
            }
            openmvs_results = self._grid_search("openmvs", scene_name, openmvs_params)
        else:
            openmvs_results = self.openmvs_pipeline.run_openmvs_with_hyperparams(scene_name, {})
            
        results["openmvs"] = openmvs_results
        
        # Compare results
        self.analyze_three_way_comparison(scene_name, results)
        
        return results
        
    def _grid_search(self, method: str, scene_name: str, param_grid: Dict):
        """Run grid search for any method"""
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        best_score = -float('inf')
        best_result = None
        
        for combo in combinations:
            hyperparams = dict(zip(param_names, combo))
            
            if method == "sugar":
                result = self.sugar_neus2_pipeline.run_sugar_with_hyperparams(scene_name, hyperparams)
            elif method == "neus2":
                result = self.sugar_neus2_pipeline.run_neus2_with_hyperparams(scene_name, hyperparams)
            else:  # openmvs
                result = self.openmvs_pipeline.run_openmvs_with_hyperparams(scene_name, hyperparams)
                
            score = result.get("quality_score", -float('inf'))
            if score > best_score:
                best_score = score
                best_result = result
                
        return best_result
        
    def analyze_three_way_comparison(self, scene_name: str, results: Dict):
        """Analyze and visualize three-way comparison"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Create comparison dataframe
        comparison_data = []
        for method, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                comparison_data.append({
                    "method": method,
                    "quality_score": result.get("quality_score", 0),
                    "processing_time": result.get("processing_time", 0),
                    "vertices": result.get("vertices", 0),
                    "faces": result.get("faces", 0),
                    "watertight": result.get("watertight", False),
                    "surface_area": result.get("surface_area", 0)
                })
                
        df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Three-Way Comparison: {scene_name}', fontsize=16)
        
        # 1. Quality scores
        ax = axes[0, 0]
        df.plot(x='method', y='quality_score', kind='bar', ax=ax, legend=False)
        ax.set_ylabel('Quality Score')
        ax.set_title('Overall Quality')
        ax.set_ylim(0, 100)
        
        # 2. Processing time
        ax = axes[0, 1]
        df['time_minutes'] = df['processing_time'] / 60
        df.plot(x='method', y='time_minutes', kind='bar', ax=ax, legend=False, color='orange')
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Processing Time')
        
        # 3. Mesh complexity
        ax = axes[0, 2]
        df[['vertices', 'faces']].plot(kind='bar', ax=ax)
        ax.set_xticklabels(df['method'], rotation=0)
        ax.set_ylabel('Count')
        ax.set_title('Mesh Complexity')
        ax.legend(['Vertices', 'Faces'])
        
        # 4. Time vs Quality scatter
        ax = axes[1, 0]
        for method in df['method']:
            row = df[df['method'] == method]
            ax.scatter(row['time_minutes'], row['quality_score'], 
                      label=method, s=200, alpha=0.7)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Quality Score')
        ax.set_title('Time vs Quality Trade-off')
        ax.legend()
        
        # 5. Method characteristics radar
        ax = axes[1, 1]
        categories = ['Quality', 'Speed', 'Watertight', 'Detail']
        radar_data = []
        
        for _, row in df.iterrows():
            method_scores = [
                row['quality_score'] / 100,
                1 - (row['time_minutes'] / df['time_minutes'].max()),
                1.0 if row['watertight'] else 0.0,
                min(1.0, row['vertices'] / 1000000)  # Normalize to 1M vertices
            ]
            radar_data.append((row['method'], method_scores))
            
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        for method, scores in radar_data:
            scores += scores[:1]
            ax.plot(angles, scores, 'o-', linewidth=2, label=method)
            ax.fill(angles, scores, alpha=0.25)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Method Characteristics')
        ax.legend()
        ax.grid(True)
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for _, row in df.iterrows():
            summary_data.append([
                row['method'].upper(),
                f"{row['quality_score']:.1f}",
                f"{row['time_minutes']:.1f}",
                "✓" if row['watertight'] else "✗",
                f"{row['vertices']/1000:.0f}k"
            ])
            
        table = ax.table(cellText=summary_data,
                        colLabels=['Method', 'Score', 'Time(m)', 'Watertight', 'Vertices'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save results
        save_path = self.comparison_dir / f"{scene_name}_three_way_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed comparison
        comparison_report = f"""
# Three-Way Comparison Report: {scene_name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Method | Quality Score | Time (min) | Vertices | Watertight |
|--------|--------------|------------|----------|------------|
"""
        
        for _, row in df.iterrows():
            comparison_report += f"| {row['method'].upper()} | {row['quality_score']:.1f} | {row['time_minutes']:.1f} | {row['vertices']:,} | {'Yes' if row['watertight'] else 'No'} |\n"
            
        comparison_report += f"""

## Best Method by Criteria

- **Highest Quality**: {df.loc[df['quality_score'].idxmax(), 'method'].upper()}
- **Fastest**: {df.loc[df['time_minutes'].idxmin(), 'method'].upper()}
- **Best Time/Quality**: {self._best_time_quality_tradeoff(df).upper()}

## Method Strengths

### SuGaR
- Fast processing
- Good for view-dependent effects
- Neural representation advantages

### NeuS2
- High quality surfaces
- Good topology
- Robust to noise

### OpenMVS
- Traditional MVS approach
- Often produces watertight meshes
- No training required

## Recommendations

"""
        
        # Add scene-specific recommendations
        if "outdoor" in scene_name.lower() or scene_name in ["bicycle", "garden", "treehill"]:
            comparison_report += "- For this outdoor scene, consider SuGaR or NeuS2 for better background handling\n"
        else:
            comparison_report += "- For this indoor scene, OpenMVS may provide good results with faster processing\n"
            
        # Save report
        report_path = self.comparison_dir / f"{scene_name}_comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(comparison_report)
            
        print(f"\n📊 Comparison saved to: {save_path}")
        print(f"📄 Report saved to: {report_path}")
        
    def _best_time_quality_tradeoff(self, df: pd.DataFrame) -> str:
        """Determine best method considering time/quality tradeoff"""
        # Simple scoring: quality_score / sqrt(time_minutes)
        df['tradeoff_score'] = df['quality_score'] / np.sqrt(df['time_minutes'] + 1)
        return df.loc[df['tradeoff_score'].idxmax(), 'method']
        
    def run_full_benchmark(self, scenes: List[str], optimization: str = "grid"):
        """Run three-way comparison on multiple scenes"""
        all_results = []
        
        for scene in scenes:
            print(f"\n{'#'*70}")
            print(f"# Processing Scene: {scene}")
            print(f"{'#'*70}")
            
            results = self.run_three_way_comparison(scene, optimization)
            all_results.append({
                "scene": scene,
                "results": results
            })
            
        # Create final summary
        self.create_benchmark_summary(all_results)
        
    def create_benchmark_summary(self, all_results: List[Dict]):
        """Create summary of all benchmark results"""
        # Aggregate results
        summary_data = []
        
        for scene_result in all_results:
            scene = scene_result["scene"]
            for method, result in scene_result["results"].items():
                if isinstance(result, dict) and "error" not in result:
                    summary_data.append({
                        "scene": scene,
                        "method": method,
                        "quality_score": result.get("quality_score", 0),
                        "processing_time": result.get("processing_time", 0) / 60,
                        "vertices": result.get("vertices", 0),
                        "watertight": result.get("watertight", False)
                    })
                    
        df = pd.DataFrame(summary_data)
        
        # Create summary visualizations
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Three-Way Method Comparison Summary', fontsize=16)
        
        # 1. Average scores by method
        ax = axes[0, 0]
        method_scores = df.groupby('method')['quality_score'].agg(['mean', 'std'])
        method_scores.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
        ax.set_ylabel('Quality Score')
        ax.set_title('Average Quality by Method')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # 2. Processing time comparison
        ax = axes[0, 1]
        df.boxplot(column='processing_time', by='method', ax=ax)
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Processing Time Distribution')
        
        # 3. Success rate (watertight meshes)
        ax = axes[1, 0]
        watertight_rate = df.groupby('method')['watertight'].mean() * 100
        watertight_rate.plot(kind='bar', ax=ax)
        ax.set_ylabel('Watertight Rate (%)')
        ax.set_title('Mesh Quality (Watertight %)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # 4. Scene difficulty
        ax = axes[1, 1]
        scene_difficulty = df.groupby('scene')['quality_score'].mean().sort_values()
        scene_difficulty.plot(kind='barh', ax=ax)
        ax.set_xlabel('Average Quality Score')
        ax.set_title('Scene Difficulty Ranking')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "benchmark_summary.png", dpi=300)
        plt.show()
        
        # Save summary report
        report = f"""
# Three-Way Benchmark Summary

## Overall Performance

| Method | Avg Quality | Avg Time (min) | Watertight Rate |
|--------|-------------|----------------|-----------------|
"""
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            report += f"| {method.upper()} | {method_df['quality_score'].mean():.1f} ± {method_df['quality_score'].std():.1f} | "
            report += f"{method_df['processing_time'].mean():.1f} ± {method_df['processing_time'].std():.1f} | "
            report += f"{method_df['watertight'].mean()*100:.0f}% |\n"
            
        with open(self.comparison_dir / "benchmark_summary.md", 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Three-Way Comparison: SuGaR vs NeuS2 vs OpenMVS")
    parser.add_argument("--install-openmvs", action="store_true", help="Install OpenMVS")
    parser.add_argument("--scenes", nargs='+', 
                       default=["bicycle"],
                       help="Scenes to process")
    parser.add_argument("--optimization", 
                       choices=["none", "grid", "bayesian"],
                       default="grid",
                       help="Hyperparameter optimization method")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with minimal hyperparameters")
    
    args = parser.parse_args()
    
    pipeline = UnifiedBenchmarkPipeline()
    
    if args.install_openmvs:
        pipeline.openmvs_pipeline.install_openmvs()
        
    if args.quick:
        args.optimization = "none"
        
    if len(args.scenes) == 1:
        # Single scene comparison
        pipeline.run_three_way_comparison(args.scenes[0], args.optimization)
    else:
        # Multi-scene benchmark
        pipeline.run_full_benchmark(args.scenes, args.optimization)
        
    print("\n✅ Three-way comparison complete!")


if __name__ == "__main__":
    main()
