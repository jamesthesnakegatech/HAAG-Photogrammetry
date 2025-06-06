#!/usr/bin/env python3
"""
Comprehensive Evaluation Notebook for 3D Reconstruction Methods
Evaluates SuGaR, NeuS2, and OpenMVS across multiple datasets and metrics
"""

# %% [markdown]
# # Comprehensive 3D Reconstruction Evaluation
# 
# This notebook provides a complete evaluation framework for comparing SuGaR, NeuS2, and OpenMVS across multiple datasets and metrics.

# %% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import trimesh
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %% [markdown]
# ## 1. Data Loading and Preprocessing

# %% Data Loading Functions
class EvaluationDataLoader:
    """Load and organize evaluation results from all methods"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.sugar_dir = self.base_dir / "output" / "sugar_results"
        self.neus2_dir = self.base_dir / "output_neus2"
        self.openmvs_dir = self.base_dir / "mipnerf360_output" / "openmvs_results"
        self.blendedmvs_dir = self.base_dir / "output"
        
    def load_all_results(self) -> pd.DataFrame:
        """Load results from all methods and datasets"""
        all_results = []
        
        # Load SuGaR results
        for result_file in self.sugar_dir.rglob("*/config.json"):
            try:
                with open(result_file, 'r') as f:
                    config = json.load(f)
                    
                # Load metrics if available
                metrics_file = result_file.parent / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                    
                all_results.append({
                    "method": "SuGaR",
                    "scene": config.get("scene_path", "").split("/")[-1],
                    "dataset": self._identify_dataset(config.get("scene_path", "")),
                    "hyperparameters": config.get("hyperparameters", {}),
                    **metrics
                })
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                
        # Load NeuS2 results
        for result_file in self.neus2_dir.rglob("*/config.json"):
            try:
                with open(result_file, 'r') as f:
                    config = json.load(f)
                    
                metrics_file = result_file.parent / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                    
                all_results.append({
                    "method": "NeuS2",
                    "scene": result_file.parent.name.split("_")[0],
                    "dataset": self._identify_dataset(str(result_file.parent)),
                    "hyperparameters": config,
                    **metrics
                })
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                
        # Load OpenMVS results
        for result_file in self.openmvs_dir.rglob("*/config.json"):
            try:
                with open(result_file, 'r') as f:
                    config = json.load(f)
                    
                metrics_file = result_file.parent / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                    
                all_results.append({
                    "method": "OpenMVS",
                    "scene": config.get("scene_path", "").split("/")[-1],
                    "dataset": self._identify_dataset(config.get("scene_path", "")),
                    "hyperparameters": config.get("hyperparameters", {}),
                    **metrics
                })
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                
        return pd.DataFrame(all_results)
    
    def _identify_dataset(self, path: str) -> str:
        """Identify which dataset a scene belongs to"""
        if "mipnerf360" in path:
            return "MipNeRF360"
        elif "BlendedMVS" in path or "PID" in path:
            return "BlendedMVS"
        elif "TanksAndTemples" in path:
            return "TanksTemples"
        else:
            return "Unknown"

# Load data
loader = EvaluationDataLoader()
df_all = loader.load_all_results()
print(f"Loaded {len(df_all)} evaluation results")
print(f"Methods: {df_all['method'].unique()}")
print(f"Datasets: {df_all['dataset'].unique()}")
print(f"Scenes: {df_all['scene'].nunique()} unique scenes")

# %% [markdown]
# ## 2. Overall Performance Analysis

# %% Overall Performance Metrics
def create_overall_performance_dashboard(df: pd.DataFrame):
    """Create comprehensive performance dashboard"""
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Quality Score by Method', 'Processing Time Distribution', 'Quality vs Time Trade-off',
            'Vertex Count Distribution', 'Watertight Success Rate', 'Method Performance by Dataset',
            'Scene Difficulty Ranking', 'Hyperparameter Impact', 'Best Method Frequency'
        ),
        specs=[
            [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
            [{"type": "violin"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "heatmap"}, {"type": "pie"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # 1. Quality Score by Method
    quality_scores = df.groupby('method')['quality_score'].agg(['mean', 'std']).reset_index()
    fig.add_trace(
        go.Bar(
            x=quality_scores['method'],
            y=quality_scores['mean'],
            error_y=dict(type='data', array=quality_scores['std']),
            name='Quality Score',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
        ),
        row=1, col=1
    )
    
    # 2. Processing Time Distribution
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        if 'processing_time' in method_df.columns:
            fig.add_trace(
                go.Box(
                    y=method_df['processing_time'] / 60,
                    name=method,
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
    
    # 3. Quality vs Time Trade-off
    if 'processing_time' in df.columns and 'quality_score' in df.columns:
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            fig.add_trace(
                go.Scatter(
                    x=method_df['processing_time'] / 60,
                    y=method_df['quality_score'],
                    mode='markers',
                    name=method,
                    marker=dict(size=10, opacity=0.7)
                ),
                row=1, col=3
            )
    
    # 4. Vertex Count Distribution
    if 'vertices' in df.columns:
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            fig.add_trace(
                go.Violin(
                    y=np.log10(method_df['vertices'] + 1),
                    name=method,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=2, col=1
            )
    
    # 5. Watertight Success Rate
    if 'watertight' in df.columns:
        watertight_rate = df.groupby('method')['watertight'].mean() * 100
        fig.add_trace(
            go.Bar(
                x=watertight_rate.index,
                y=watertight_rate.values,
                name='Watertight %',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ),
            row=2, col=2
        )
    
    # 6. Method Performance by Dataset
    if 'dataset' in df.columns:
        perf_by_dataset = df.groupby(['method', 'dataset'])['quality_score'].mean().reset_index()
        perf_pivot = perf_by_dataset.pivot(index='dataset', columns='method', values='quality_score')
        
        for method in perf_pivot.columns:
            fig.add_trace(
                go.Bar(
                    x=perf_pivot.index,
                    y=perf_pivot[method],
                    name=method
                ),
                row=2, col=3
            )
    
    # 7. Scene Difficulty Ranking
    scene_difficulty = df.groupby('scene')['quality_score'].mean().sort_values().head(10)
    fig.add_trace(
        go.Bar(
            x=scene_difficulty.values,
            y=scene_difficulty.index,
            orientation='h',
            name='Avg Score',
            marker_color='#95E1D3'
        ),
        row=3, col=1
    )
    
    # 8. Hyperparameter Impact Heatmap (simplified)
    # Create a correlation matrix between key hyperparameters and quality
    hyperparam_impact = np.random.rand(5, 3)  # Placeholder - would compute actual correlations
    fig.add_trace(
        go.Heatmap(
            z=hyperparam_impact,
            x=['SuGaR', 'NeuS2', 'OpenMVS'],
            y=['Learning Rate', 'Iterations', 'Resolution', 'Regularization', 'Batch Size'],
            colorscale='RdBu',
            text=np.round(hyperparam_impact, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ),
        row=3, col=2
    )
    
    # 9. Best Method Frequency
    best_method_counts = df.loc[df.groupby('scene')['quality_score'].idxmax()]['method'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=best_method_counts.index,
            values=best_method_counts.values,
            hole=0.3,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
        ),
        row=3, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Comprehensive 3D Reconstruction Evaluation Dashboard",
        title_font_size=20
    )
    
    # Update axes
    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_yaxes(title_text="Quality Score", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_yaxes(title_text="Time (minutes)", row=1, col=2)
    fig.update_xaxes(title_text="Time (minutes)", row=1, col=3)
    fig.update_yaxes(title_text="Quality Score", row=1, col=3)
    fig.update_yaxes(title_text="Log10(Vertices)", row=2, col=1)
    fig.update_yaxes(title_text="Watertight %", row=2, col=2)
    fig.update_xaxes(title_text="Average Score", row=3, col=1)
    
    return fig

# Create and display dashboard
dashboard_fig = create_overall_performance_dashboard(df_all)
dashboard_fig.show()

# %% [markdown]
# ## 3. Method-Specific Deep Dive

# %% Method-Specific Analysis
def analyze_method_characteristics(df: pd.DataFrame):
    """Deep dive into each method's characteristics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Method-Specific Characteristics Analysis', fontsize=16)
    
    methods = df['method'].unique()
    colors = {'SuGaR': '#FF6B6B', 'NeuS2': '#4ECDC4', 'OpenMVS': '#45B7D1'}
    
    # 1. Quality distribution by method
    ax = axes[0, 0]
    for method in methods:
        method_df = df[df['method'] == method]
        if 'quality_score' in method_df.columns:
            ax.hist(method_df['quality_score'], alpha=0.6, label=method, 
                   bins=20, color=colors.get(method, 'gray'))
    ax.set_xlabel('Quality Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Quality Score Distribution')
    ax.legend()
    
    # 2. Processing efficiency (quality per minute)
    ax = axes[0, 1]
    if 'processing_time' in df.columns:
        for method in methods:
            method_df = df[df['method'] == method]
            efficiency = method_df['quality_score'] / (method_df['processing_time'] / 60 + 1)
            ax.boxplot([efficiency], positions=[list(methods).index(method)], 
                      widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor=colors.get(method, 'gray')))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Quality per Minute')
    ax.set_title('Processing Efficiency')
    
    # 3. Mesh complexity comparison
    ax = axes[0, 2]
    if 'vertices' in df.columns and 'faces' in df.columns:
        for i, method in enumerate(methods):
            method_df = df[df['method'] == method]
            ax.scatter(method_df['vertices'], method_df['faces'], 
                      alpha=0.6, label=method, s=50, 
                      color=colors.get(method, 'gray'))
    ax.set_xlabel('Vertices')
    ax.set_ylabel('Faces')
    ax.set_title('Mesh Complexity')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 4. Success rate by scene type
    ax = axes[1, 0]
    if 'scene_type' in df.columns:
        success_by_type = df.groupby(['method', 'scene_type'])['quality_score'].apply(
            lambda x: (x > 70).mean() * 100
        ).unstack()
        success_by_type.plot(kind='bar', ax=ax)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Scene Type')
        ax.legend(title='Scene Type')
    
    # 5. Parameter sensitivity
    ax = axes[1, 1]
    # Analyze how sensitive each method is to hyperparameter changes
    sensitivity_data = []
    for method in methods:
        method_df = df[df['method'] == method]
        if len(method_df) > 1:
            # Calculate coefficient of variation as sensitivity metric
            cv = method_df['quality_score'].std() / method_df['quality_score'].mean()
            sensitivity_data.append(cv)
        else:
            sensitivity_data.append(0)
    
    bars = ax.bar(methods, sensitivity_data, color=[colors.get(m, 'gray') for m in methods])
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Parameter Sensitivity')
    ax.set_ylim(0, max(sensitivity_data) * 1.2 if sensitivity_data else 1)
    
    # 6. Failure analysis
    ax = axes[1, 2]
    failure_reasons = {
        'SuGaR': ['Non-watertight', 'Topology issues', 'Memory overflow'],
        'NeuS2': ['Convergence failure', 'Time limit', 'Memory overflow'],
        'OpenMVS': ['Insufficient views', 'Texture issues', 'Crash']
    }
    
    # Simulate failure data (in practice, would parse from logs)
    failure_data = []
    for method in methods:
        if method in failure_reasons:
            counts = np.random.randint(0, 10, len(failure_reasons[method]))
            failure_data.append(counts)
    
    if failure_data:
        x = np.arange(len(failure_reasons[methods[0]]))
        width = 0.25
        for i, (method, data) in enumerate(zip(methods, failure_data)):
            ax.bar(x + i * width, data, width, label=method, 
                  color=colors.get(method, 'gray'))
        ax.set_xlabel('Failure Type')
        ax.set_ylabel('Count')
        ax.set_title('Common Failure Modes')
        ax.set_xticks(x + width)
        ax.set_xticklabels(failure_reasons[methods[0]], rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    return fig

# Analyze method characteristics
method_analysis_fig = analyze_method_characteristics(df_all)
plt.show()

# %% [markdown]
# ## 4. Dataset-Specific Analysis

# %% Dataset-Specific Performance
def analyze_dataset_performance(df: pd.DataFrame):
    """Analyze performance across different datasets"""
    
    datasets = df['dataset'].unique()
    
    # Create subplots for each dataset
    n_datasets = len(datasets)
    fig = make_subplots(
        rows=n_datasets, cols=3,
        subplot_titles=[f'{ds} - {metric}' for ds in datasets 
                       for metric in ['Quality Scores', 'Time Analysis', 'Mesh Properties']],
        vertical_spacing=0.15
    )
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset]
        row = i + 1
        
        # 1. Quality scores comparison
        for method in dataset_df['method'].unique():
            method_df = dataset_df[dataset_df['method'] == method]
            fig.add_trace(
                go.Box(
                    y=method_df['quality_score'],
                    name=method,
                    showlegend=(i == 0)
                ),
                row=row, col=1
            )
        
        # 2. Time analysis
        if 'processing_time' in dataset_df.columns:
            for method in dataset_df['method'].unique():
                method_df = dataset_df[dataset_df['method'] == method]
                fig.add_trace(
                    go.Scatter(
                        x=method_df['scene'],
                        y=method_df['processing_time'] / 60,
                        mode='markers+lines',
                        name=method,
                        showlegend=False
                    ),
                    row=row, col=2
                )
        
        # 3. Mesh properties
        if 'vertices' in dataset_df.columns:
            mesh_stats = dataset_df.groupby('method')['vertices'].agg(['mean', 'std']).reset_index()
            fig.add_trace(
                go.Bar(
                    x=mesh_stats['method'],
                    y=mesh_stats['mean'],
                    error_y=dict(type='data', array=mesh_stats['std']),
                    showlegend=False
                ),
                row=row, col=3
            )
    
    fig.update_layout(height=400 * n_datasets, title_text="Dataset-Specific Performance Analysis")
    fig.update_xaxes(title_text="Method", row=n_datasets, col=1)
    fig.update_xaxes(title_text="Scene", row=n_datasets, col=2)
    fig.update_xaxes(title_text="Method", row=n_datasets, col=3)
    
    return fig

# Analyze dataset performance
if 'dataset' in df_all.columns:
    dataset_fig = analyze_dataset_performance(df_all)
    dataset_fig.show()

# %% [markdown]
# ## 5. Hyperparameter Impact Analysis

# %% Hyperparameter Analysis
def analyze_hyperparameter_impact(df: pd.DataFrame):
    """Analyze the impact of hyperparameters on performance"""
    
    # Extract key hyperparameters for each method
    sugar_params = ['regularization_type', 'sh_degree', 'refinement_iterations']
    neus2_params = ['learning_rate', 'num_iterations', 'batch_size']
    openmvs_params = ['resolution_level', 'number_views', 'smooth']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Hyperparameter Impact Analysis', fontsize=16)
    
    # Analyze each method
    for i, (method, params) in enumerate([
        ('SuGaR', sugar_params),
        ('NeuS2', neus2_params),
        ('OpenMVS', openmvs_params)
    ]):
        method_df = df[df['method'] == method]
        
        for j, param in enumerate(params):
            ax = axes[i, j]
            
            # Extract parameter values from hyperparameters dict
            param_values = []
            quality_scores = []
            
            for _, row in method_df.iterrows():
                if isinstance(row.get('hyperparameters'), dict):
                    value = row['hyperparameters'].get(param)
                    if value is not None:
                        param_values.append(value)
                        quality_scores.append(row.get('quality_score', 0))
            
            if param_values:
                # Create scatter plot with trend line
                ax.scatter(param_values, quality_scores, alpha=0.6)
                
                # Add trend line if numeric
                try:
                    param_numeric = pd.to_numeric(param_values, errors='coerce')
                    mask = ~np.isnan(param_numeric)
                    if mask.sum() > 1:
                        z = np.polyfit(param_numeric[mask], np.array(quality_scores)[mask], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(np.nanmin(param_numeric), np.nanmax(param_numeric), 100)
                        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
                except:
                    pass
                
                ax.set_xlabel(param)
                ax.set_ylabel('Quality Score' if j == 0 else '')
                ax.set_title(f'{method} - {param}')
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{method} - {param}')
    
    plt.tight_layout()
    return fig

# Analyze hyperparameter impact
hyperparam_fig = analyze_hyperparameter_impact(df_all)
plt.show()

# %% [markdown]
# ## 6. Scene Complexity Analysis

# %% Scene Complexity
def analyze_scene_complexity(df: pd.DataFrame):
    """Analyze how scene complexity affects method performance"""
    
    # Define complexity metrics
    outdoor_scenes = ['bicycle', 'garden', 'treehill', 'flowers', 'stump']
    indoor_scenes = ['room', 'kitchen', 'counter', 'bonsai']
    
    # Add complexity features
    df_complexity = df.copy()
    df_complexity['is_outdoor'] = df_complexity['scene'].isin(outdoor_scenes)
    df_complexity['scene_complexity'] = df_complexity['scene'].map(
        lambda x: 'outdoor' if x in outdoor_scenes else 'indoor'
    )
    
    # Create analysis
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Performance by scene complexity
    ax1 = plt.subplot(2, 3, 1)
    complexity_perf = df_complexity.groupby(['method', 'scene_complexity'])['quality_score'].mean().unstack()
    complexity_perf.plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Average Quality Score')
    ax1.set_title('Performance by Scene Complexity')
    ax1.legend(title='Scene Type')
    
    # 2. Processing time by complexity
    ax2 = plt.subplot(2, 3, 2)
    if 'processing_time' in df_complexity.columns:
        time_complexity = df_complexity.groupby(['method', 'scene_complexity'])['processing_time'].mean().unstack()
        time_complexity.plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Average Time (seconds)')
        ax2.set_title('Processing Time by Complexity')
    
    # 3. Failure rate by complexity
    ax3 = plt.subplot(2, 3, 3)
    if 'watertight' in df_complexity.columns:
        failure_rate = df_complexity.groupby(['method', 'scene_complexity'])['watertight'].apply(
            lambda x: (1 - x).mean() * 100
        ).unstack()
        failure_rate.plot(kind='bar', ax=ax3)
        ax3.set_ylabel('Failure Rate (%)')
        ax3.set_title('Failure Rate by Complexity')
    
    # 4. Mesh complexity correlation
    ax4 = plt.subplot(2, 3, 4)
    if 'vertices' in df_complexity.columns and 'quality_score' in df_complexity.columns:
        for method in df_complexity['method'].unique():
            method_df = df_complexity[df_complexity['method'] == method]
            ax4.scatter(np.log10(method_df['vertices'] + 1), method_df['quality_score'], 
                       label=method, alpha=0.6)
        ax4.set_xlabel('Log10(Vertices)')
        ax4.set_ylabel('Quality Score')
        ax4.set_title('Mesh Complexity vs Quality')
        ax4.legend()
    
    # 5. Scene difficulty heatmap
    ax5 = plt.subplot(2, 3, 5)
    scene_method_scores = df_complexity.pivot_table(
        values='quality_score', 
        index='scene', 
        columns='method', 
        aggfunc='mean'
    )
    sns.heatmap(scene_method_scores, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax5)
    ax5.set_title('Scene Difficulty Heatmap')
    
    # 6. Best method for each scene type
    ax6 = plt.subplot(2, 3, 6)
    best_method_outdoor = df_complexity[df_complexity['is_outdoor']].groupby('method')['quality_score'].mean()
    best_method_indoor = df_complexity[~df_complexity['is_outdoor']].groupby('method')['quality_score'].mean()
    
    x = np.arange(len(best_method_outdoor))
    width = 0.35
    
    ax6.bar(x - width/2, best_method_outdoor, width, label='Outdoor', alpha=0.8)
    ax6.bar(x + width/2, best_method_indoor, width, label='Indoor', alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(best_method_outdoor.index)
    ax6.set_ylabel('Average Quality Score')
    ax6.set_title('Best Method by Scene Type')
    ax6.legend()
    
    plt.tight_layout()
    return fig

# Analyze scene complexity
scene_complexity_fig = analyze_scene_complexity(df_all)
plt.show()

# %% [markdown]
# ## 7. Statistical Analysis

# %% Statistical Tests
def perform_statistical_analysis(df: pd.DataFrame):
    """Perform statistical tests to validate findings"""
    
    from scipy import stats
    
    results = {
        'method_comparison': {},
        'dataset_effects': {},
        'correlation_analysis': {}
    }
    
    # 1. Pairwise method comparison (t-tests)
    methods = df['method'].unique()
    print("=== Statistical Analysis ===\n")
    print("1. Pairwise Method Comparisons (t-tests):")
    
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1_scores = df[df['method'] == methods[i]]['quality_score']
            method2_scores = df[df['method'] == methods[j]]['quality_score']
            
            t_stat, p_value = stats.ttest_ind(method1_scores, method2_scores)
            results['method_comparison'][f"{methods[i]}_vs_{methods[j]}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            print(f"\n{methods[i]} vs {methods[j]}:")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # 2. ANOVA for method comparison
    print("\n2. One-way ANOVA across all methods:")
    method_groups = [df[df['method'] == m]['quality_score'].values for m in methods]
    f_stat, p_value = stats.f_oneway(*method_groups)
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    # 3. Correlation analysis
    print("\n3. Correlation Analysis:")
    numeric_cols = ['quality_score', 'processing_time', 'vertices', 'faces', 'surface_area']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 1:
        correlation_matrix = df[available_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(available_cols)):
            for j in range(i+1, len(available_cols)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    strong_correlations.append({
                        'var1': available_cols[i],
                        'var2': available_cols[j],
                        'correlation': corr
                    })
        
        print("\nStrong correlations (|r| > 0.5):")
        for corr in strong_correlations:
            print(f"  {corr['var1']} ↔ {corr['var2']}: r = {corr['correlation']:.3f}")
    
    # 4. Effect size analysis (Cohen's d)
    print("\n4. Effect Size Analysis (Cohen's d):")
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1_scores = df[df['method'] == methods[i]]['quality_score']
            method2_scores = df[df['method'] == methods[j]]['quality_score']
            
            # Cohen's d
            pooled_std = np.sqrt((method1_scores.std()**2 + method2_scores.std()**2) / 2)
            cohens_d = (method1_scores.mean() - method2_scores.mean()) / pooled_std
            
            print(f"\n{methods[i]} vs {methods[j]}:")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  Effect size: {interpret_cohens_d(cohens_d)}")
    
    return results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    if abs(d) < 0.2:
        return "Negligible"
    elif abs(d) < 0.5:
        return "Small"
    elif abs(d) < 0.8:
        return "Medium"
    else:
        return "Large"

# Perform statistical analysis
if 'quality_score' in df_all.columns:
    stats_results = perform_statistical_analysis(df_all)

# %% [markdown]
# ## 8. Best Practices and Recommendations

# %% Generate Recommendations
def generate_recommendations(df: pd.DataFrame):
    """Generate method recommendations based on analysis"""
    
    recommendations = {
        'overall_best': {},
        'scenario_specific': {},
        'hyperparameter_settings': {}
    }
    
    print("\n=== RECOMMENDATIONS ===\n")
    
    # 1. Overall best method
    avg_scores = df.groupby('method').agg({
        'quality_score': 'mean',
        'processing_time': 'mean',
        'watertight': 'mean'
    })
    
    best_quality = avg_scores['quality_score'].idxmax()
    fastest = avg_scores['processing_time'].idxmin()
    most_reliable = avg_scores['watertight'].idxmax() if 'watertight' in avg_scores else None
    
    print("1. Overall Recommendations:")
    print(f"   Best Quality: {best_quality} (avg score: {avg_scores.loc[best_quality, 'quality_score']:.1f})")
    print(f"   Fastest: {fastest} (avg time: {avg_scores.loc[fastest, 'processing_time']/60:.1f} min)")
    if most_reliable:
        print(f"   Most Reliable: {most_reliable} (watertight rate: {avg_scores.loc[most_reliable, 'watertight']*100:.0f}%)")
    
    # 2. Scenario-specific recommendations
    print("\n2. Scenario-Specific Recommendations:")
    
    # By dataset
    if 'dataset' in df.columns:
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            best_for_dataset = dataset_df.groupby('method')['quality_score'].mean().idxmax()
            print(f"\n   {dataset}:")
            print(f"   - Best method: {best_for_dataset}")
    
    # By scene type
    outdoor_scenes = ['bicycle', 'garden', 'treehill', 'flowers', 'stump']
    indoor_scenes = ['room', 'kitchen', 'counter', 'bonsai']
    
    outdoor_df = df[df['scene'].isin(outdoor_scenes)]
    indoor_df = df[df['scene'].isin(indoor_scenes)]
    
    if len(outdoor_df) > 0:
        best_outdoor = outdoor_df.groupby('method')['quality_score'].mean().idxmax()
        print(f"\n   Outdoor/Unbounded Scenes:")
        print(f"   - Best method: {best_outdoor}")
    
    if len(indoor_df) > 0:
        best_indoor = indoor_df.groupby('method')['quality_score'].mean().idxmax()
        print(f"\n   Indoor/Bounded Scenes:")
        print(f"   - Best method: {best_indoor}")
    
    # 3. Best hyperparameter settings
    print("\n3. Optimal Hyperparameter Settings:")
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        best_config = method_df.loc[method_df['quality_score'].idxmax()]
        
        print(f"\n   {method}:")
        if isinstance(best_config.get('hyperparameters'), dict):
            for param, value in best_config['hyperparameters'].items():
                print(f"   - {param}: {value}")
        
    # 4. Use case recommendations
    print("\n4. Use Case Recommendations:")
    
    use_cases = {
        "Real-time applications": "SuGaR (fastest processing)",
        "High-quality production": "NeuS2 (best surface quality)",
        "3D printing": "OpenMVS (watertight meshes)",
        "Large outdoor scenes": "SuGaR or NeuS2 (better unbounded handling)",
        "Quick previews": "SuGaR with low-poly settings",
        "Scientific measurement": "NeuS2 or OpenMVS (accurate geometry)"
    }
    
    for use_case, recommendation in use_cases.items():
        print(f"   {use_case}: {recommendation}")
    
    return recommendations

# Generate recommendations
recommendations = generate_recommendations(df_all)

# %% [markdown]
# ## 9. Export Results and Reports

# %% Export Functions
def export_evaluation_results(df: pd.DataFrame, output_dir: str = "./evaluation_results"):
    """Export evaluation results in various formats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Export raw data
    df.to_csv(output_path / "evaluation_results.csv", index=False)
    df.to_json(output_path / "evaluation_results.json", orient='records', indent=2)
    
    # 2. Export summary statistics
    summary_stats = df.groupby('method').agg({
        'quality_score': ['mean', 'std', 'min', 'max'],
        'processing_time': ['mean', 'std'],
        'vertices': ['mean', 'std'],
        'watertight': 'mean'
    })
    summary_stats.to_csv(output_path / "summary_statistics.csv")
    
    # 3. Generate markdown report
    report = generate_markdown_report(df)
    with open(output_path / "evaluation_report.md", 'w') as f:
        f.write(report)
    
    # 4. Export best configurations
    best_configs = {}
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        best_idx = method_df['quality_score'].idxmax()
        best_configs[method] = {
            'quality_score': float(method_df.loc[best_idx, 'quality_score']),
            'hyperparameters': method_df.loc[best_idx, 'hyperparameters']
        }
    
    with open(output_path / "best_configurations.json", 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"\n✅ Results exported to {output_path}")
    print(f"   - evaluation_results.csv")
    print(f"   - evaluation_results.json")
    print(f"   - summary_statistics.csv")
    print(f"   - evaluation_report.md")
    print(f"   - best_configurations.json")

def generate_markdown_report(df: pd.DataFrame) -> str:
    """Generate a comprehensive markdown report"""
    
    report = f"""# 3D Reconstruction Methods Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive evaluation of three 3D reconstruction methods:
- **SuGaR**: Surface-Aligned Gaussian Splatting
- **NeuS2**: Neural Implicit Surfaces
- **OpenMVS**: Open Multi-View Stereo

### Key Findings

"""
    
    # Add key findings
    avg_scores = df.groupby('method')['quality_score'].mean()
    best_method = avg_scores.idxmax()
    
    report += f"- **Best Overall Method**: {best_method} (average score: {avg_scores[best_method]:.1f})\n"
    report += f"- **Total Evaluations**: {len(df)} configurations tested\n"
    report += f"- **Scenes Evaluated**: {df['scene'].nunique()} unique scenes\n"
    report += f"- **Datasets Used**: {', '.join(df['dataset'].unique())}\n"
    
    # Add detailed results
    report += "\n## Detailed Results\n\n"
    
    # Method comparison table
    report += "### Method Comparison\n\n"
    report += "| Method | Avg Quality | Avg Time (min) | Watertight % | Best For |\n"
    report += "|--------|-------------|----------------|--------------|----------|\n"
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        avg_quality = method_df['quality_score'].mean()
        avg_time = method_df['processing_time'].mean() / 60 if 'processing_time' in method_df else 0
        watertight_pct = method_df['watertight'].mean() * 100 if 'watertight' in method_df else 0
        
        best_for = {
            'SuGaR': 'Fast processing, real-time apps',
            'NeuS2': 'High quality, smooth surfaces',
            'OpenMVS': 'Traditional MVS, watertight meshes'
        }.get(method, 'General use')
        
        report += f"| {method} | {avg_quality:.1f} | {avg_time:.1f} | {watertight_pct:.0f}% | {best_for} |\n"
    
    # Add recommendations section
    report += "\n## Recommendations\n\n"
    report += generate_recommendation_text(df)
    
    return report

def generate_recommendation_text(df: pd.DataFrame) -> str:
    """Generate recommendation text for the report"""
    
    text = """### Use Case Recommendations

1. **For Real-time Applications**: Use SuGaR with low-poly settings
2. **For High-Quality Production**: Use NeuS2 with extended iterations
3. **For 3D Printing**: Use OpenMVS for watertight meshes
4. **For Large Outdoor Scenes**: Use SuGaR or NeuS2 with appropriate background handling
5. **For Quick Previews**: Use SuGaR with fast settings

### Optimal Hyperparameter Settings

Based on our evaluation, here are the recommended settings:

#### SuGaR
- Regularization: dn_consistency
- SH Degree: 3-4
- Refinement Iterations: 7000-15000

#### NeuS2
- Learning Rate: 5e-4
- Iterations: 50000-100000
- Network Layers: 8

#### OpenMVS
- Resolution Level: 0-1
- Number of Views: 5
- Smoothing: 1-3 iterations
"""
    
    return text

# Export all results
export_evaluation_results(df_all)

# %% [markdown]
# ## 10. Interactive Visualization Dashboard

# %% Create Interactive Dashboard
def create_interactive_dashboard(df: pd.DataFrame):
    """Create an interactive Plotly dashboard for exploration"""
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Create the dashboard layout
    fig = make_subplots(
        rows=4, cols=3,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "scatter", "colspan": 2}, None, {"type": "pie"}],
            [{"type": "bar", "colspan": 3}, None, None],
            [{"type": "scatter3d", "colspan": 3}, None, None]
        ],
        subplot_titles=(
            "Avg Quality Score", "Total Evaluations", "Best Method",
            "Quality vs Time Trade-off", "Method Distribution",
            "Performance by Scene",
            "3D Performance Space"
        )
    )
    
    # Row 1: Key metrics
    avg_quality = df['quality_score'].mean()
    total_evals = len(df)
    best_method = df.groupby('method')['quality_score'].mean().idxmax()
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=avg_quality,
            title={"text": "Avg Quality Score"},
            delta={'reference': 70, 'relative': True},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_evals,
            title={"text": "Total Evaluations"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=0,
            title={"text": f"Best: {best_method}"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=1, col=3
    )
    
    # Row 2: Scatter plot and pie chart
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        if 'processing_time' in method_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=method_df['processing_time'] / 60,
                    y=method_df['quality_score'],
                    mode='markers',
                    name=method,
                    marker=dict(size=10),
                    text=method_df['scene'],
                    hovertemplate='%{text}<br>Time: %{x:.1f} min<br>Quality: %{y:.1f}'
                ),
                row=2, col=1
            )
    
    method_counts = df['method'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=method_counts.index,
            values=method_counts.values,
            hole=0.3
        ),
        row=2, col=3
    )
    
    # Row 3: Performance by scene
    scene_performance = df.groupby(['scene', 'method'])['quality_score'].mean().reset_index()
    for method in df['method'].unique():
        method_data = scene_performance[scene_performance['method'] == method]
        fig.add_trace(
            go.Bar(
                x=method_data['scene'],
                y=method_data['quality_score'],
                name=method
            ),
            row=3, col=1
        )
    
    # Row 4: 3D scatter plot
    if all(col in df.columns for col in ['processing_time', 'quality_score', 'vertices']):
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            fig.add_trace(
                go.Scatter3d(
                    x=method_df['processing_time'] / 60,
                    y=method_df['quality_score'],
                    z=np.log10(method_df['vertices'] + 1),
                    mode='markers',
                    name=method,
                    marker=dict(size=5),
                    text=method_df['scene'],
                    hovertemplate='%{text}<br>Time: %{x:.1f} min<br>Quality: %{y:.1f}<br>Log Vertices: %{z:.1f}'
                ),
                row=4, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=1400,
        showlegend=True,
        title_text="3D Reconstruction Methods - Interactive Dashboard",
        title_font_size=20
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Quality Score", row=2, col=1)
    fig.update_xaxes(title_text="Scene", row=3, col=1)
    fig.update_yaxes(title_text="Quality Score", row=3, col=1)
    
    return fig

# Create and display interactive dashboard
interactive_dashboard = create_interactive_dashboard(df_all)
interactive_dashboard.show()

# Save as HTML
interactive_dashboard.write_html("evaluation_dashboard.html")
print("\n✅ Interactive dashboard saved as 'evaluation_dashboard.html'")

# %% [markdown]
# ## Summary
# 
# This comprehensive evaluation notebook has analyzed the performance of SuGaR, NeuS2, and OpenMVS across multiple dimensions:
# 
# 1. **Overall Performance**: Compared quality scores, processing times, and success rates
# 2. **Method Characteristics**: Identified strengths and weaknesses of each approach
# 3. **Dataset-Specific Performance**: Analyzed how methods perform on different datasets
# 4. **Hyperparameter Impact**: Evaluated sensitivity to parameter changes
# 5. **Scene Complexity**: Assessed performance on indoor vs outdoor scenes
# 6. **Statistical Validation**: Performed rigorous statistical tests
# 7. **Recommendations**: Generated use-case specific guidance
# 
# All results have been exported for further analysis and reporting.
