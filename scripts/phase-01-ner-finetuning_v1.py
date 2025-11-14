# %% [markdown]
# ### 0. Project Configuration & Setup

# %% [markdown]
# #### 0.1 Import Libraries

# %%
# ============================================================================
# 0.1 IMPORT LIBRARIES
# ============================================================================

# Standard Libraries
import os
import sys
import json
import random
import warnings
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bars
from tqdm.auto import tqdm

# Machine Learning & NLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Transformers & Datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset, DatasetDict

# Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score as seqeval_f1_score,
    precision_score as seqeval_precision_score,
    recall_score as seqeval_recall_score
)

# Weights & Biases for experiment tracking
import wandb

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Suppress warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

print("\n✓ All libraries imported successfully!")

# %% [markdown]
# #### 0.2 Configuration Constants

# %%
# ============================================================================
# 0.2 CONFIGURATION CONSTANTS
# ============================================================================

# Project Information
PROJECT_NAME = "Medical_Entity_Recognition_Linking"

# Timestamp for experiment tracking
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"MERL_{TIMESTAMP}"

# Dataset Configuration
DATASET_NAME = "ktgiahieu/maccrobat2018_2020"
DATASET_SPLIT_SIZES = {
    'train': 340,
    'validation': 30,
    'test': 30
}

# Model Configuration
MODEL_CONFIGS = {
    'pubmedbert': {
        'name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        'display_name': 'PubMedBERT'
    },
    'biobert': {
        'name': 'dmis-lab/biobert-v1.1',
        'display_name': 'BioBERT'
    },
    'bioformer': {
        'name': 'bioformers/bioformer-16L',
        'display_name': 'Bioformer-16L'
    }
}

# Primary model for baseline
PRIMARY_MODEL = 'pubmedbert'
PRIMARY_MODEL_NAME = MODEL_CONFIGS[PRIMARY_MODEL]['name']

# Training Hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    'fp16': torch.cuda.is_available(),  # Use mixed precision if GPU available
    'logging_steps': 50,
    'eval_steps': 100,
    'save_steps': 100,
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'greater_is_better': True
}

# Entity Type Tiers (from EDA)
ENTITY_TIERS = {
    'frequent': [
        'Age', 'Diagnostic_procedure', 'Sign_symptom', 'Lab_value',
        'Biological_structure', 'Detailed_description', 'Date',
        'Disease_disorder', 'History', 'Therapeutic_procedure',
        'Medication', 'Dosage', 'Duration', 'Clinical_event',
        'Nonbiological_location'
    ],
    'medium': [
        'Family_history', 'Coreference', 'Sex', 'Distance', 'Other_entity',
        'Area', 'Volume', 'Time', 'Frequency', 'Activity', 'Other_event',
        'Personal_background', 'Administration'
    ],
    'rare': [
        'Subject', 'Occupation', 'Outcome', 'Shape', 'Severity',
        'Qualitative_concept', 'Quantitative_concept', 'Texture',
        'Color', 'Height', 'Weight', 'Biological_attribute', 'Mass'
    ]
}

# High-value entities for medical coding (production focus)
HIGH_VALUE_ENTITIES = [
    'Disease_disorder',
    'Therapeutic_procedure',
    'Medication',
    'Sign_symptom',
    'Diagnostic_procedure'
]

# Confidence Thresholds to Test
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

# Output Directories
OUTPUT_DIR = f"./outputs/{EXPERIMENT_NAME}"
MODEL_SAVE_DIR = f"{OUTPUT_DIR}/models"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Random Seeds for Reproducibility
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# Display Configuration
print("="*80)
print(f"PROJECT: {PROJECT_NAME}")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"TIMESTAMP: {TIMESTAMP}")
print("="*80)
print(f"\nPrimary Model: {MODEL_CONFIGS[PRIMARY_MODEL]['display_name']}")
print(f"Model Path: {PRIMARY_MODEL_NAME}")
print(f"\nDataset: {DATASET_NAME}")
print(f"Train: {DATASET_SPLIT_SIZES['train']} | "
      f"Val: {DATASET_SPLIT_SIZES['validation']} | "
      f"Test: {DATASET_SPLIT_SIZES['test']}")
print(f"\nTraining Config:")
for key, value in TRAINING_CONFIG.items():
    print(f"  {key}: {value}")
print(f"\nOutput Directory: {OUTPUT_DIR}")
print("="*80)
print("\n✓ Configuration loaded successfully!")

# %% [markdown]
# #### 0.3 WandB Initialization

# %%
# ============================================================================
# 0.3 WANDB INITIALIZATION
# ============================================================================

# Method 1: Using Kaggle Secrets (if available)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
    os.environ['WANDB_API_KEY'] = wandb_api_key
    print("✓ WandB API key loaded from Kaggle Secrets")
except:
    print("⚠ Kaggle Secrets not available. WandB will prompt for login.")
    print("  You can manually set: os.environ['WANDB_API_KEY'] = 'your_key_here'")

# Initialize WandB
try:
    # Login to WandB
    wandb.login()
    
    # Initialize WandB project
    wandb.init(
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        config={
            "model_name": PRIMARY_MODEL_NAME,
            "dataset": DATASET_NAME,
            "timestamp": TIMESTAMP,
            **TRAINING_CONFIG,
            "entity_tiers": ENTITY_TIERS,
            "high_value_entities": HIGH_VALUE_ENTITIES,
            "random_seed": RANDOM_SEED
        },
        tags=[PRIMARY_MODEL],
        notes=f"Baseline NER model training on MACCROBAT dataset. "
              f"Focus: production-readiness metrics and error analysis."
    )
    
    print("="*80)
    print(f"✓ WandB initialized successfully!")
    print(f"  Project: {PROJECT_NAME}")
    print(f"  Run Name: {EXPERIMENT_NAME}")
    print(f"  Dashboard: {wandb.run.get_url()}")
    print("="*80)
    
    WANDB_ENABLED = True
    
except Exception as e:
    print(f"⚠ WandB initialization failed: {e}")
    print("  Continuing without WandB logging...")
    WANDB_ENABLED = False

# %% [markdown]
# #### 0.4 Helper Functions

# %%
# ============================================================================
# 0.4 HELPER FUNCTIONS
# ============================================================================

def log_to_wandb(data: Dict, step: Optional[int] = None, commit: bool = True):
    """
    Safely log data to WandB if enabled
    
    Args:
        data: Dictionary of metrics to log
        step: Optional step number
        commit: Whether to commit the log immediately
    """
    if WANDB_ENABLED:
        try:
            wandb.log(data, step=step, commit=commit)
        except Exception as e:
            print(f"Warning: Failed to log to WandB: {e}")


def save_results(data: Dict, filename: str, directory: str = RESULTS_DIR):
    """
    Save results to JSON file
    
    Args:
        data: Dictionary to save
        filename: Name of the file (with .json extension)
        directory: Directory to save to
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Results saved to: {filepath}")
    
    # Also log to WandB as artifact
    if WANDB_ENABLED:
        try:
            artifact = wandb.Artifact(
                name=filename.replace('.json', ''),
                type='results'
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to log artifact to WandB: {e}")


def save_plot(fig, filename: str, directory: str = PLOTS_DIR, dpi: int = 300):
    """
    Save matplotlib figure and log to WandB
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (with extension)
        directory: Directory to save to
        dpi: Resolution for saved figure
    """
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Plot saved to: {filepath}")
    
    # Log to WandB
    if WANDB_ENABLED:
        try:
            wandb.log({filename.replace('.png', ''): wandb.Image(filepath)})
        except Exception as e:
            print(f"Warning: Failed to log plot to WandB: {e}")
    
    plt.close(fig)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")


def create_summary_table(results_dict: Dict) -> pd.DataFrame:
    """
    Create a formatted summary table from results dictionary
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
    
    Returns:
        Pandas DataFrame with formatted results
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df


print("="*80)
print("✓ Helper functions defined successfully!")
print("\nAvailable functions:")
print("  - log_to_wandb(): Log metrics to WandB")
print("  - save_results(): Save results to JSON")
print("  - save_plot(): Save and log matplotlib figures")
print("  - format_time(): Convert seconds to readable format")
print("  - print_gpu_memory(): Check GPU usage")
print("  - create_summary_table(): Format results as DataFrame")
print("="*80)

# %% [markdown]
# ---

# %% [markdown]
# ### 1. Environment Setup & Data Preparation

# %% [markdown]
# #### 1.1 Load MACCROBAT Dataset

# %% [markdown]
# #### 1.2 Dataset Structure Analysis

# %% [markdown]
# #### 1.3 Tag Mapping Creation (tag2id, id2tag)

# %% [markdown]
# #### 1.4 Entity Type Tier Classification

# %% [markdown]
# #### 1.5 Dataset Splitting (Train/Val/Test)

# %% [markdown]
# #### 1.6 Tokenization & Data Collator

# %% [markdown]
# #### 1.7 Evaluation Metrics Setup

# %% [markdown]
# ---

# %% [markdown]
# ### 2. Baseline Model: PubMedBERT Fine-tuning

# %% [markdown]
# #### 2.1 Model & Tokenizer Initialization

# %% [markdown]
# #### 2.2 Training Configuration

# %% [markdown]
# #### 2.3 Training Loop with WandB Logging

# %% [markdown]
# #### 2.4 Model Evaluation on Test Set

# %% [markdown]
# #### 2.5 Save Model Artifacts to WandB

# %% [markdown]
# ---

# %% [markdown]
# ### 3. Additional Model Training

# %% [markdown]
# #### 3.1 BioBERT Fine-tuning

# %% [markdown]
# #### 3.2 Alternative Model Training (Bioformer/ClinicalBERT)

# %% [markdown]
# #### 3.3 Model Comparison Matrix

# %% [markdown]
# #### 3.4 Best Model Selection Criteria

# %% [markdown]
# ---

# %% [markdown]
# ### 4. Per-Entity-Type Performance Analysis

# %% [markdown]
# #### 4.1 Extract Predictions by Entity Type

# %% [markdown]
# #### 4.2 Calculate Metrics per Entity Type

# %% [markdown]
# #### 4.3 Tier-Based Analysis (Frequent/Medium/Rare)

# %% [markdown]
# #### 4.4 Identify Systematic Weaknesses

# %% [markdown]
# #### 4.5 Visualizations

# %% [markdown]
# ##### 4.5.1 F1 Score by Entity Type (Bar Chart)

# %% [markdown]
# ##### 4.5.2 Entity Frequency vs F1 Score (Scatter Plot)

# %% [markdown]
# ##### 4.5.3 Confusion Matrix Heatmap

# %% [markdown]
# ---

# %% [markdown]
# ### 5. Error Analysis Framework

# %% [markdown]
# #### 5.1 Boundary Error Detection

# %% [markdown]
# ##### 5.1.1 Partial Match Analysis

# %% [markdown]
# ##### 5.1.2 Exact Match vs Partial Match Rate

# %% [markdown]
# #### 5.2 Entity Type Confusion Analysis

# %% [markdown]
# ##### 5.2.1 Build Confusion Matrix

# %% [markdown]
# ##### 5.2.2 Common Confusion Patterns

# %% [markdown]
# #### 5.3 Multi-token Entity Analysis

# %% [markdown]
# ##### 5.3.1 Single-token vs Multi-token Performance

# %% [markdown]
# ##### 5.3.2 Entity Length Impact on F1

# %% [markdown]
# #### 5.4 Error Examples Collection

# %% [markdown]
# ##### 5.4.1 Sample Mispredictions

# %% [markdown]
# ##### 5.4.2 Error Categorization

# %% [markdown]
# ##### 5.4.3 Pattern Documentation

# %% [markdown]
# ---

# %% [markdown]
# ### 6. Confidence Score Analysis

# %% [markdown]
# #### 6.1 Extract Prediction Confidence Scores

# %% [markdown]
# #### 6.2 Confidence Distribution Analysis

# %% [markdown]
# ##### 6.2.1 Correct vs Incorrect Predictions

# %% [markdown]
# ##### 6.2.2 Average Confidence by Prediction Type (TP/FP/FN)

# %% [markdown]
# #### 6.3 Threshold Analysis for Production

# %% [markdown]
# ##### 6.3.1 Test Multiple Thresholds [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

# %% [markdown]
# ##### 6.3.2 Precision/Coverage/Recall at Each Threshold

# %% [markdown]
# #### 6.4 Precision-Coverage Tradeoff Visualization

# %% [markdown]
# #### 6.5 Production Threshold Recommendation

# %% [markdown]
# ---

# %% [markdown]
# ### 7. Inference Speed & Scalability Testing

# %% [markdown]
# #### 7.1 Measure Inference Latency

# %% [markdown]
# ##### 7.1.1 Per Document (Batch Size = 1)

# %% [markdown]
# ##### 7.1.2 Per Batch (Batch Size = 8, 16, 32)

# %% [markdown]
# #### 7.2 Memory Profiling

# %% [markdown]
# ##### 7.2.1 GPU Memory During Training

# %% [markdown]
# ##### 7.2.2 GPU Memory During Inference

# %% [markdown]
# ##### 7.2.3 CPU Inference Feasibility

# %% [markdown]
# #### 7.3 Throughput Estimation

# %% [markdown]
# ##### 7.3.1 Documents per Second

# %% [markdown]
# ##### 7.3.2 Large-scale Processing Estimates

# %% [markdown]
# #### 7.4 Scalability Considerations

# %% [markdown]
# ---

# %% [markdown]
# ### 8. Production-Readiness Metrics

# %% [markdown]
# #### 8.1 Clinical Coding Simulation

# %% [markdown]
# ##### 8.1.1 Auto-code vs Manual Review Decision

# %% [markdown]
# ##### 8.1.2 Automation Rate at Different Thresholds

# %% [markdown]
# #### 8.2 High-Value Entity Analysis

# %% [markdown]
# ##### 8.2.1 F1 on Critical Entities (Disease_disorder, Therapeutic_procedure, Medication)

# %% [markdown]
# ##### 8.2.2 Precision on High-Value Entities at 0.85 Confidence

# %% [markdown]
# #### 8.3 False Positive Impact Analysis

# %% [markdown]
# ##### 8.3.1 FP Rate by Entity Type

# %% [markdown]
# ##### 8.3.2 High-Risk Entity Identification

# %% [markdown]
# #### 8.4 Production Readiness Report Generation

# %% [markdown]
# ---

# %% [markdown]
# ### 9. Model Selection & Final Documentation

# %% [markdown]
# #### 9.1 Best Model Selection

# %% [markdown]
# ##### 9.1.1 Selection Criteria & Tradeoffs

# %% [markdown]
# ##### 9.1.2 Justification Documentation

# %% [markdown]
# #### 9.2 Phase 1 Summary

# %% [markdown]
# ##### 9.2.1 Training Results Comparison Table

# %% [markdown]
# ##### 9.2.2 Key Visualizations Compilation

# %% [markdown]
# ##### 9.2.3 Error Analysis Summary

# %% [markdown]
# ##### 9.2.4 Production Metrics Summary

# %% [markdown]
# #### 9.3 Model Artifacts Export

# %% [markdown]
# ##### 9.3.1 Save Best Checkpoint

# %% [markdown]
# ##### 9.3.2 Save Tokenizer & Mappings

# %% [markdown]
# ##### 9.3.3 Save Evaluation Predictions

# %% [markdown]
# #### 9.4 Phase 1 Insights Document

# %% [markdown]
# ##### 9.4.1 Key Findings

# %% [markdown]
# ##### 9.4.2 Surprising Results

# %% [markdown]
# ##### 9.4.3 Recommendations for Phase 2

# %% [markdown]
# ---

# %% [markdown]
# ### 10. Appendix

# %% [markdown]
# #### 10.1 Utility Functions Reference

# %% [markdown]
# #### 10.2 Hyperparameter Log

# %% [markdown]
# #### 10.3 Raw Results Tables

# %% [markdown]
# #### 10.4 Additional Visualizations


