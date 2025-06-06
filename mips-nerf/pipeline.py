#!/usr/bin/env python3
"""
Mip-NeRF 360 Benchmark Pipeline with Hyperparameter Tuning
Integrates Mip-NeRF 360 dataset with SuGaR and NeuS2, includes hyperparameter optimization
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
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
import optuna
import requests
from tqdm import tqdm
import zipfile


class MipNeRF360Pipeline:
    """Pipeline for Mip-NeRF 360 dataset processing"""
    
    def __init__(self, dataset_dir: str = "./mipnerf360", output_dir: str = "./mipnerf360_output"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mip-NeRF 360 scenes
        self.scenes = {
            "outdoor": ["bicycle", "flowers", "garden", "stump", "treehill"],
            "indoor": ["room", "counter", "kitchen", "bonsai"]
        }
        
        # Dataset URLs
        self.dataset_urls = {
            "360_v2": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
            "360_v2_extra": "http://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
        }
        
        # Hyperparameter search spaces optimized for unbounded scenes
        self.sugar_hyperparam_space = {
            # Core parameters
            "sh_degree": [2, 3, 4],  # Spherical harmonics degree
            "lambda_dssim": [0.1, 0.2, 0.3],  # SSIM loss weight
            "lambda_dist": [0.0, 0.1, 0.2],  # Distortion loss
            
            # Densification parameters
            "densification_interval": [100, 200, 300],
            "opacity_reset_interval": [2000, 3000, 4000],
            "densify_grad_threshold": [0.0001, 0.0002, 0.0005],
            "percent_dense": [0.005, 0.01, 0.02],
            
            # Regularization parameters
            "lambda_normal": [0.01, 0.05, 0.1],  # Normal consistency
            "lambda_dist_ratio": [0.1, 0.5, 1.0],  # Distance ratio loss
            "regularization_type": ["density", "sdf", "dn_consistency"],
            
            # Background handling (important for unbounded scenes)
            "background_type": ["white", "black", "random"],
            "near_plane": [0.01, 0.1, 0.5],
            "far_plane": [10.0, 50.0, 100.0],
            
            # Refinement
            "refinement_iterations": [2000, 7000, 15000],
            "refinement_lr": [1e-5, 1e-4, 1e-3]
        }
        
        self.neus2_hyperparam_space = {
            # Training parameters
            "learning_rate": [1e-4, 5e-4, 1e-3],
            "num_iterations": [20000, 50000, 100000],
            "batch_size": [512, 1024, 2048],
            
            # Sampling parameters
            "n_samples": [64, 128, 256],
            "n_importance": [64, 128, 256],
            "up_sample_steps": [1, 2, 4],
            "perturb": [0.0, 1.0],
            
            # Network architecture
            "sdf_network": {
                "d_hidden": [128, 256, 512],
                "n_layers": [4, 8, 12],
                "skip_in": [[4], [4, 6], [4, 6, 8]],
                "bias": [0.1, 0.5, 1.0],
                "scale": [1.0, 2.0, 4.0],
                "geometric_init": [True, False]
            },
            
            # Background model (crucial for unbounded scenes)
            "background_network": {
                "d_hidden": [64, 128],
                "n_layers": [2, 4],
                "background_type": ["nerf++", "mipnerf360"]
            },
            
            # Variance network
            "variance_network": {
                "init_val": [0.1, 0.3, 0.5]
            }
        }
        
    def download_mipnerf360(self, scenes_only: List[str] = None):
        """Download Mip-NeRF 360 dataset"""
        print("📥 Downloading Mip-NeRF 360 dataset...")
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download main dataset
        zip_path = self.dataset_dir / "360_v2.zip"
        if not zip_path.exists():
            print("Downloading 360_v2.zip (~25GB)...")
            self._download_file(self.dataset_urls["360_v2"], zip_path)
            
        # Extract dataset
        if not (self.dataset_dir / "360_v2").exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.dataset_dir)
                
        # Move scenes to root level for easier access
        v2_dir = self.dataset_dir / "360_v2"
        if v2_dir.exists():
            for scene_dir in v2_dir.iterdir():
                if scene_dir.is_dir():
                    dest = self.dataset_dir / scene_dir.name
                    if not dest.exists():
                        shutil.move(str(scene_dir), str(dest))
                        
        print("✅ Mip-NeRF 360 dataset ready!")
        
    def _download_file(self, url: str, dest_path: Path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    def prepare_scene_for_processing(self, scene_name: str) -> Path:
        """Prepare Mip-NeRF 360 scene for processing"""
        print(f"🔧 Preparing {scene_name} for processing...")
        
        scene_path = self.dataset_dir / scene_name
        if not scene_path.exists():
            raise ValueError(f"Scene {scene_name} not found in {self.dataset_dir}")
            
        # Mip-NeRF 360 scenes come in COLMAP format
        # Check if we need to reorganize
        if not (scene_path / "sparse").exists():
            # Create COLMAP structure
            sparse_dir = scene_path / "sparse" / "0"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            
            # Move COLMAP files if they exist
            for file in ["cameras.bin", "images.bin", "points3D.bin"]:
                if (scene_path / file).exists():
                    shutil.move(str(scene_path / file), str(sparse_dir / file))
                    
        # Create images directory if needed
        if not (scene_path / "images").exists() and (scene_path / "images_360").exists():
            shutil.move(str(scene_path / "images_360"), str(scene_path / "images"))
            
        return scene_path
        
    def run_sugar_with_hyperparams(self, scene_name: str, hyperparams: Dict) -> Dict:
        """Run SuGaR with specific hyperparameters"""
        print(f"🚀 Running SuGaR on {scene_name} with custom hyperparameters...")
        
        scene_path = self.prepare_scene_for_processing(scene_name)
        output_path = self.output_dir / "sugar_results" / f"{scene_name}_{self._hyperparam_hash(hyperparams)}"
        
        # Create custom config
        config = {
            "scene_path": str(scene_path),
            "output_path": str(output_path),
            "hyperparameters": hyperparams,
            "dataset_type": "mipnerf360"
        }
        
        config_path = output_path / "config.json"
        output_path.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Run SuGaR with custom parameters
        start_time = time.time()
        
        # Build command with all hyperparameters
        cmd = [
            "python", "train_full_pipeline.py",
            "-s", str(scene_path),
            "-r", hyperparams.get("regularization_type", "sdf"),  # SDF often better for unbounded
            "--sh_degree", str(hyperparams.get("sh_degree", 3)),
            "--lambda_dssim", str(hyperparams.get("lambda_dssim", 0.2)),
            "--lambda_dist", str(hyperparams.get("lambda_dist", 0.1)),
            "--densification_interval", str(hyperparams.get("densification_interval", 100)),
            "--opacity_reset_interval", str(hyperparams.get("opacity_reset_interval", 3000)),
            "--densify_grad_threshold", str(hyperparams.get("densify_grad_threshold", 0.0002)),
            "--percent_dense", str(hyperparams.get("percent_dense", 0.01)),
            "--refinement_iterations", str(hyperparams.get("refinement_iterations", 7000))
        ]
        
        # Add background handling for unbounded scenes
        if hyperparams.get("background_type") == "white":
            cmd.append("--white_background")
            
        try:
            # Change to SuGaR directory
            sugar_dir = Path("./SuGaR")
            subprocess.run(cmd, check=True, cwd=sugar_dir)
            processing_time = time.time() - start_time
            
            # Evaluate results
            metrics = self.evaluate_reconstruction(output_path, scene_name, "sugar")
            metrics["processing_time"] = processing_time
            metrics["hyperparameters"] = hyperparams
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"❌ SuGaR failed for {scene_name}: {e}")
            return {"error": str(e), "hyperparameters": hyperparams}
            
    def run_neus2_with_hyperparams(self, scene_name: str, hyperparams: Dict) -> Dict:
        """Run NeuS2 with specific hyperparameters"""
        print(f"🚀 Running NeuS2 on {scene_name} with custom hyperparameters...")
        
        scene_path = self.prepare_scene_for_processing(scene_name)
        output_path = self.output_dir / "neus2_results" / f"{scene_name}_{self._hyperparam_hash(hyperparams)}"
        
        # Create NeuS2 config for unbounded scenes
        neus2_config = self._create_neus2_config_unbounded(scene_name, scene_path, hyperparams)
        config_path = output_path / "config.json"
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(neus2_config, f, indent=2)
            
        # Run NeuS2
        start_time = time.time()
        
        cmd = [
            "python", "train.py",
            "--conf", str(config_path),
            "--case", scene_name,
            "--mode", "train"
        ]
        
        try:
            neus2_dir = Path("./NeuS2")
            subprocess.run(cmd, check=True, cwd=neus2_dir)
            
            # Extract mesh
            extract_cmd = cmd[:-2] + ["--mode", "validate_mesh", "--resolution", "512"]
            subprocess.run(extract_cmd, check=True, cwd=neus2_dir)
            
            processing_time = time.time() - start_time
            
            # Evaluate results
            metrics = self.evaluate_reconstruction(output_path, scene_name, "neus2")
            metrics["processing_time"] = processing_time
            metrics["hyperparameters"] = hyperparams
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"❌ NeuS2 failed for {scene_name}: {e}")
            return {"error": str(e), "hyperparameters": hyperparams}
            
    def evaluate_reconstruction(self, output_path: Path, scene_name: str, method: str) -> Dict:
        """Evaluate reconstruction quality for unbounded scenes"""
        metrics = {
            "scene": scene_name,
            "method": method,
            "scene_type": "outdoor" if scene_name in self.scenes["outdoor"] else "indoor"
        }
        
        # Load mesh
        if method == "sugar":
            mesh_files = list((output_path / "refined_mesh").glob("*.obj"))
        else:
            mesh_files = list(output_path.glob("*.ply"))
            
        if not mesh_files:
            return {"error": "No mesh found"}
            
        try:
            import trimesh
            mesh = trimesh.load(mesh_files[0])
            
            # Basic metrics
            metrics["vertices"] = len(mesh.vertices)
            metrics["faces"] = len(mesh.faces)
            metrics["watertight"] = mesh.is_watertight
            metrics["volume"] = float(mesh.volume) if mesh.is_watertight else 0
            metrics["surface_area"] = float(mesh.area)
            
            # Bounding box for unbounded scenes
            bounds = mesh.bounds
            scene_extent = bounds[1] - bounds[0]
            metrics["scene_extent"] = scene_extent.tolist()
            metrics["max_extent"] = float(np.max(scene_extent))
            
            # Quality metrics specific to unbounded scenes
            face_areas = mesh.area_faces
            metrics["face_area_std"] = float(np.std(face_areas))
            metrics["degenerate_faces"] = int(np.sum(face_areas < 1e-6))
            
            # For outdoor scenes, check background handling
            if metrics["scene_type"] == "outdoor":
                # Check if mesh extends far enough (unbounded)
                if metrics["max_extent"] < 10.0:
                    metrics["background_score"] = 0.5  # Might be truncated
                else:
                    metrics["background_score"] = 1.0
            
            # Compute quality score
            metrics["quality_score"] = self._compute_quality_score(metrics)
            
        except Exception as e:
            metrics["error"] = str(e)
            
        return metrics
        
    def _compute_quality_score(self, metrics: Dict) -> float:
        """Compute quality score for unbounded scenes"""
        score = 100.0
        
        # Penalize non-watertight meshes
        if not metrics.get("watertight", False):
            score -= 20
            
        # Check vertex count (unbounded scenes need more vertices)
        ideal_vertices = 800000 if metrics["scene_type"] == "outdoor" else 500000
        vertex_ratio = metrics["vertices"] / ideal_vertices
        if vertex_ratio < 0.5:
            score -= 20
        elif vertex_ratio > 2.0:
            score -= 10
            
        # Penalize high face area variance
        area_cv = metrics["face_area_std"] / (metrics["surface_area"] / metrics["faces"])
        score -= min(10, area_cv * 10)
        
        # Bonus for good background handling in outdoor scenes
        if metrics["scene_type"] == "outdoor":
            score += metrics.get("background_score", 0.5) * 10
            
        return max(0, min(100, score))
        
    def _hyperparam_hash(self, hyperparams: Dict) -> str:
        """Create a hash for hyperparameter combination"""
        import hashlib
        param_str = json.dumps(hyperparams, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
        
    def _create_neus2_config_unbounded(self, scene_name: str, scene_path: Path, hyperparams: Dict) -> Dict:
        """Create NeuS2 configuration for unbounded scenes"""
        config = {
            "general": {
                "base_exp_dir": f"./exp/{scene_name}_unbounded",
                "data_dir": str(scene_path),
                "resolution": 512,
                "unbounded": True  # Important for Mip-NeRF 360 scenes
            },
            "train": {
                "learning_rate": hyperparams.get("learning_rate", 5e-4),
                "num_iterations": hyperparams.get("num_iterations", 50000),
                "batch_size": hyperparams.get("batch_size", 1024),
                "validate_resolution_level": 4,
                "warm_up_iter": 1000,
                "anneal_end_iter": 50000,
                "use_white_background": False,
                "save_freq": 10000
            },
            "model": {
                "sdf_network": {
                    "d_out": 257,
                    "d_in": 3,
                    "d_hidden": hyperparams.get("sdf_network", {}).get("d_hidden", 256),
                    "n_layers": hyperparams.get("sdf_network", {}).get("n_layers", 8),
                    "skip_in": hyperparams.get("sdf_network", {}).get("skip_in", [4]),
                    "bias": hyperparams.get("sdf_network", {}).get("bias", 0.5),
                    "scale": hyperparams.get("sdf_network", {}).get("scale", 2.0),  # Larger scale for unbounded
                    "geometric_init": hyperparams.get("sdf_network", {}).get("geometric_init", True),
                    "weight_norm": True
                },
                "variance_network": {
                    "init_val": hyperparams.get("variance_network", {}).get("init_val", 0.3)
                },
                "rendering_network": {
                    "d_feature": 256,
                    "mode": "idr",
                    "d_out": 3,
                    "d_hidden": 256,
                    "n_layers": 4
                },
                # Background network for unbounded scenes
                "background_network": {
                    "d_hidden": hyperparams.get("background_network", {}).get("d_hidden", 128),
                    "n_layers": hyperparams.get("background_network", {}).get("n_layers", 4),
                    "background_type": hyperparams.get("background_network", {}).get("background_type", "mipnerf360")
                }
            },
            "dataset": {
                "n_samples": hyperparams.get("n_samples", 128),
                "n_importance": hyperparams.get("n_importance", 128),
                "n_outside": 32,  # Important for unbounded scenes
                "up_sample_steps": hyperparams.get("up_sample_steps", 2),
                "perturb": hyperparams.get("perturb", 1.0),
                "scale_mat": 1.0
            }
        }
        
        return config


class MipNeRF360HyperparamOptimizer:
    """Hyperparameter optimization for Mip-NeRF 360 scenes"""
    
    def __init__(self, pipeline: MipNeRF360Pipeline):
        self.pipeline = pipeline
        self.results_dir = pipeline.output_dir / "hyperparam_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def grid_search(self, method: str, scene_name: str, param_grid: Dict) -> pd.DataFrame:
        """Perform grid search over hyperparameters"""
        print(f"\n🔍 Grid Search for {method} on {scene_name}")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Total combinations: {len(combinations)}")
        
        results = []
        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(param_names, combo))
            print(f"\n[{i+1}/{len(combinations)}] Testing: {hyperparams}")
            
            if method == "sugar":
                metrics = self.pipeline.run_sugar_with_hyperparams(scene_name, hyperparams)
            else:
                metrics = self.pipeline.run_neus2_with_hyperparams(scene_name, hyperparams)
                
            results.append(metrics)
            
            # Save intermediate results
            with open(self.results_dir / f"{method}_{scene_name}_grid_search.json", 'w') as f:
                json.dump(results, f, indent=2)
                
        return pd.DataFrame(results)
        
    def bayesian_optimization(self, method: str, scene_name: str, n_trials: int = 50):
        """Perform Bayesian optimization using Optuna"""
        print(f"\n🎯 Bayesian Optimization for {method} on {scene_name}")
        
        # Determine if outdoor scene (needs different parameters)
        is_outdoor = scene_name in self.pipeline.scenes["outdoor"]
        
        def objective(trial):
            # Sample hyperparameters based on scene type
            if method == "sugar":
                hyperparams = {
                    "sh_degree": trial.suggest_int("sh_degree", 2, 4),
                    "lambda_dssim": trial.suggest_float("lambda_dssim", 0.1, 0.3),
                    "lambda_dist": trial.suggest_float("lambda_dist", 0.0, 0.2),
                    "densification_interval": trial.suggest_int("densification_interval", 100, 300),
                    "opacity_reset_interval": trial.suggest_int("opacity_reset_interval", 2000, 4000),
                    "densify_grad_threshold": trial.suggest_float("densify_grad_threshold", 0.0001, 0.0005, log=True),
                    "percent_dense": trial.suggest_float("percent_dense", 0.005, 0.02),
                    "lambda_normal": trial.suggest_float("lambda_normal", 0.01, 0.1, log=True),
                    "regularization_type": trial.suggest_categorical("regularization_type", 
                                                                    ["sdf", "dn_consistency"] if is_outdoor else ["density", "dn_consistency"])
                }
                
                # Add unbounded-specific parameters for outdoor scenes
                if is_outdoor:
                    hyperparams["background_type"] = trial.suggest_categorical("background_type", ["white", "black"])
                    hyperparams["far_plane"] = trial.suggest_float("far_plane", 10.0, 100.0)
                    
                metrics = self.pipeline.run_sugar_with_hyperparams(scene_name, hyperparams)
            else:
                hyperparams = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
                    "num_iterations": trial.suggest_int("num_iterations", 20000, 100000, step=10000),
                    "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
                    "n_samples": trial.suggest_categorical("n_samples", [64, 128, 256]),
                    "n_importance": trial.suggest_categorical("n_importance", [64, 128, 256]),
                    "sdf_network": {
                        "d_hidden": trial.suggest_categorical("d_hidden", [128, 256, 512]),
                        "n_layers": trial.suggest_int("n_layers", 4, 12),
                        "bias": trial.suggest_float("bias", 0.1, 1.0),
                        "scale": trial.suggest_float("scale", 1.0, 4.0) if is_outdoor else 1.0
                    }
                }
                
                # Add background network for outdoor scenes
                if is_outdoor:
                    hyperparams["background_network"] = {
                        "d_hidden": trial.suggest_categorical("bg_d_hidden", [64, 128]),
                        "n_layers": trial.suggest_int("bg_n_layers", 2, 4),
                        "background_type": trial.suggest_categorical("bg_type", ["nerf++", "mipnerf360"])
                    }
                    
                metrics = self.pipeline.run_neus2_with_hyperparams(scene_name, hyperparams)
                
            # Return objective value (minimize negative score)
            if "error" in metrics:
                return float('inf')
            return -metrics.get("quality_score", 0)
            
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{method}_{scene_name}",
            storage=f"sqlite:///{self.results_dir}/{method}_{scene_name}_optuna.db",
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Save results
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            "method": method,
            "scene": scene_name,
            "scene_type": "outdoor" if is_outdoor else "indoor",
            "best_params": best_params,
            "best_score": -best_value,
            "n_trials": len(study.trials)
        }
        
        with open(self.results_dir / f"{method}_{scene_name}_best_params.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return study
        
    def analyze_results(self, method: str, scene_name: str):
        """Analyze hyperparameter optimization results"""
        print(f"\n📊 Analyzing results for {method} on {scene_name}")
        
        # Load results
        results_file = self.results_dir / f"{method}_{scene_name}_grid_search.json"
        if not results_file.exists():
            print("No results found")
            return
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        df = pd.DataFrame(results)
        
        # Remove failed runs
        df = df[~df['error'].notna()]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Hyperparameter Analysis for {method} on {scene_name}')
        
        # 1. Score distribution
        ax = axes[0, 0]
        df['quality_score'].hist(bins=20, ax=ax)
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        
        # 2. Processing time vs score
        ax = axes[0, 1]
        ax.scatter(df['processing_time'] / 60, df['quality_score'])
        ax.set_xlabel('Processing Time (minutes)')
        ax.set_ylabel('Quality Score')
        ax.set_title('Time vs Quality Trade-off')
        
        # 3. Scene extent (important for unbounded)
        ax = axes[1, 0]
        max_extents = df['scene_extent'].apply(lambda x: max(x) if isinstance(x, list) else 0)
        ax.scatter(max_extents, df['quality_score'])
        ax.set_xlabel('Max Scene Extent')
        ax.set_ylabel('Quality Score')
        ax.set_title('Scene Size vs Quality')
        
        # 4. Best configurations
        ax = axes[1, 1]
        top_5 = df.nlargest(5, 'quality_score')
        y_pos = np.arange(len(top_5))
        ax.barh(y_pos, top_5['quality_score'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Config {i}" for i in range(len(top_5))])
        ax.set_xlabel('Quality Score')
        ax.set_title('Top 5 Configurations')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{method}_{scene_name}_analysis.png")
        plt.show()
        
        # Print best configuration
        best_config = df.loc[df['quality_score'].idxmax()]
        print(f"\n🏆 Best Configuration:")
        print(f"Score: {best_config['quality_score']:.2f}")
        print(f"Time: {best_config['processing_time']/60:.1f} minutes")
        print(f"Scene Extent: {best_config.get('max_extent', 'N/A')}")
        print(f"Parameters: {json.dumps(best_config['hyperparameters'], indent=2)}")
        
        return df


class MipNeRF360BenchmarkRunner:
    """Main runner for Mip-NeRF 360 benchmark with hyperparameter tuning"""
    
    def __init__(self):
        self.pipeline = MipNeRF360Pipeline()
        self.optimizer = MipNeRF360HyperparamOptimizer(self.pipeline)
        
    def run_full_benchmark(self, scenes: List[str], methods: List[str], optimization: str = "grid"):
        """Run full benchmark with hyperparameter optimization"""
        
        results_summary = []
        
        for scene in scenes:
            for method in methods:
                print(f"\n{'='*60}")
                print(f"Processing {scene} with {method}")
                print(f"Scene type: {'outdoor' if scene in self.pipeline.scenes['outdoor'] else 'indoor'}")
                print(f"{'='*60}")
                
                if optimization == "grid":
                    # Quick grid search with reduced space
                    if method == "sugar":
                        param_grid = {
                            "regularization_type": ["sdf", "dn_consistency"],
                            "sh_degree": [3, 4],
                            "lambda_dssim": [0.1, 0.2],
                            "refinement_iterations": [2000, 7000]
                        }
                    else:
                        param_grid = {
                            "learning_rate": [1e-4, 5e-4],
                            "num_iterations": [30000, 50000],
                            "n_samples": [128, 256]
                        }
                    
                    df = self.optimizer.grid_search(method, scene, param_grid)
                    
                elif optimization == "bayesian":
                    # Bayesian optimization
                    study = self.optimizer.bayesian_optimization(method, scene, n_trials=20)
                    
                else:
                    # Single run with default parameters
                    if method == "sugar":
                        metrics = self.pipeline.run_sugar_with_hyperparams(scene, {})
                    else:
                        metrics = self.pipeline.run_neus2_with_hyperparams(scene, {})
                    df = pd.DataFrame([metrics])
                    
                # Analyze results
                self.optimizer.analyze_results(method, scene)
                
                # Get best result
                if not df.empty and 'quality_score' in df.columns:
                    best_idx = df['quality_score'].idxmax()
                    best_result = df.loc[best_idx].to_dict()
                    best_result['scene'] = scene
                    best_result['method'] = method
                    results_summary.append(best_result)
                
        # Create final report
        self.create_benchmark_report(results_summary)
        
    def create_benchmark_report(self, results: List[Dict]):
        """Create comprehensive benchmark report"""
        df = pd.DataFrame(results)
        
        report = f"""
# Mip-NeRF 360 Benchmark Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

The Mip-NeRF 360 dataset contains challenging unbounded scenes with complex lighting and geometry:
- **Outdoor scenes**: bicycle, flowers, garden, stump, treehill
- **Indoor scenes**: room, counter, kitchen, bonsai

## Summary Statistics

### By Method
{df.groupby('method')['quality_score'].describe() if 'quality_score' in df else 'No quality scores available'}

### By Scene Type
{df.groupby('scene_type')['quality_score'].describe() if 'scene_type' in df else 'No scene type data'}

### By Scene
{df.groupby('scene')['quality_score'].describe() if 'quality_score' in df else 'No scene data'}

## Best Configurations

"""
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            if not method_df.empty and 'quality_score' in method_df:
                best = method_df.loc[method_df['quality_score'].idxmax()]
                
                report += f"""
### {method.upper()}
- Best Score: {best.get('quality_score', 'N/A')}
- Scene: {best['scene']}
- Scene Type: {best.get('scene_type', 'N/A')}
- Processing Time: {best.get('processing_time', 0)/60:.1f} minutes
- Max Scene Extent: {best.get('max_extent', 'N/A')}
- Hyperparameters:
```json
{json.dumps(best.get('hyperparameters', {}), indent=2)}
```
"""
        
        # Save report
        report_path = self.pipeline.output_dir / "mipnerf360_benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"\n📄 Report saved to: {report_path}")
        
        # Create comparison visualization
        if not df.empty:
            self.create_comparison_plots(df)
        
    def create_comparison_plots(self, df: pd.DataFrame):
        """Create comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Method comparison by scene type
        ax = axes[0, 0]
        if 'scene_type' in df and 'quality_score' in df:
            df.boxplot(column='quality_score', by=['method', 'scene_type'], ax=ax)
            ax.set_title('Score Distribution by Method and Scene Type')
            ax.set_xlabel('Method, Scene Type')
            ax.set_ylabel('Quality Score')
        
        # 2. Indoor vs Outdoor performance
        ax = axes[0, 1]
        if 'scene_type' in df and 'quality_score' in df:
            scene_type_means = df.groupby(['scene_type', 'method'])['quality_score'].mean().unstack()
            scene_type_means.plot(kind='bar', ax=ax)
            ax.set_title('Average Score: Indoor vs Outdoor')
            ax.set_xlabel('Scene Type')
            ax.set_ylabel('Average Quality Score')
            ax.legend(title='Method')
        
        # 3. Time-Quality trade-off
        ax = axes[1, 0]
        if 'processing_time' in df and 'quality_score' in df:
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                ax.scatter(method_df['processing_time']/60, method_df['quality_score'], 
                          label=method, s=100, alpha=0.6)
            ax.set_xlabel('Processing Time (minutes)')
            ax.set_ylabel('Quality Score')
            ax.set_title('Time vs Quality Trade-off')
            ax.legend()
        
        # 4. Scene difficulty ranking
        ax = axes[1, 1]
        if 'quality_score' in df:
            scene_means = df.groupby('scene')['quality_score'].mean().sort_values()
            scene_means.plot(kind='barh', ax=ax)
            ax.set_title('Scene Difficulty (Lower = Harder)')
            ax.set_xlabel('Average Quality Score')
            ax.set_ylabel('Scene')
        
        plt.suptitle('Mip-NeRF 360 Benchmark Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.pipeline.output_dir / "mipnerf360_benchmark_comparison.png", dpi=300)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Mip-NeRF 360 Benchmark with Hyperparameter Tuning")
    parser.add_argument("--download", action="store_true", help="Download Mip-NeRF 360 dataset")
    parser.add_argument("--scenes", nargs='+', 
                       default=["bicycle", "garden", "room"],
                       help="Scenes to process")
    parser.add_argument("--methods", nargs='+', 
                       default=["sugar", "neus2"],
                       help="Methods to benchmark")
    parser.add_argument("--optimization", 
                       choices=["none", "grid", "bayesian"],
                       default="grid",
                       help="Hyperparameter optimization method")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with minimal hyperparameters")
    
    args = parser.parse_args()
    
    runner = MipNeRF360BenchmarkRunner()
    
    if args.download:
        runner.pipeline.download_mipnerf360()
        
    if args.quick:
        # Quick test mode
        args.scenes = args.scenes[:1]
        args.optimization = "none"
        
    runner.run_full_benchmark(args.scenes, args.methods, args.optimization)
    
    print("\n✅ Mip-NeRF 360 benchmark complete!")


if __name__ == "__main__":
    main()
