#!/usr/bin/env python3
"""
PCA Preprocessing Script for Isoform Prediction Project
Preprocesses full dataset and saves PCA-transformed features
"""

import os
import pickle
import gc

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = '/work3/s193518/scIsoPred/data/'


def load_data():
    """Load gene and transcript expression data."""
    print("Loading gene expression data...")
    genes_file = os.path.join(DATA_PATH, 'bulk_processed_genes.h5ad')
    genes = sc.read_h5ad(genes_file)
    print(f"Genes loaded: {genes.shape}")
    
    print("Loading transcript expression data...")
    transcripts_file = os.path.join(DATA_PATH, 'bulk_processed_transcripts.h5ad')
    transcripts = sc.read_h5ad(transcripts_file)
    print(f"Transcripts loaded: {transcripts.shape}")
    
    return genes, transcripts


def align_samples(genes, transcripts):
    """Ensure gene and transcript data have matching sample order."""
    if np.array_equal(genes.obs_names, transcripts.obs_names):
        print("Samples already aligned.")
        return genes, transcripts
    
    print("Aligning samples between gene and transcript data...")
    common_samples = genes.obs_names.intersection(transcripts.obs_names)
    genes = genes[common_samples, :]
    transcripts = transcripts[common_samples, :]
    print(f"Aligned samples: {len(common_samples)}")
    
    return genes, transcripts


def densify(adata):
    """Convert sparse matrix to dense numpy array."""
    matrix = adata.X
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def preprocess_inputs(X_raw):
    """
    Preprocess gene expression: log1p transform followed by z-score normalization.
    This matches the preprocessing in dataloaders/preprocess.py
    """
    print("Applying log1p transformation...")
    X_log = np.log1p(X_raw)
    
    print("Applying z-score normalization per gene...")
    mean = X_log.mean(axis=0, keepdims=True)
    std = X_log.std(axis=0, keepdims=True) + 1e-6
    X_normalized = (X_log - mean) / std
    
    return X_normalized, mean, std


def apply_pca(X, n_components):
    """Apply PCA dimensionality reduction."""
    print(f"Applying PCA with {n_components} components...")
    
    max_components = min(X.shape[0], X.shape[1]) - 1
    if n_components > max_components:
        print(f"Reducing n_components from {n_components} to {max_components}")
        n_components = max_components
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA complete: {X_pca.shape}")
    print(f"Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    
    return X_pca, pca, explained_var


def prepare_targets(Y_raw):
    """
    Prepare target isoform expression: log1p transform.
    This matches the preprocessing in dataloaders/preprocess.py
    """
    print("Applying log1p transformation to targets...")
    return np.log1p(Y_raw)


def save_outputs(output_dir, X_normalized, X_pca_dict, Y, metadata):
    """Save all preprocessed data and models."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving outputs to {output_dir}...")
    
    np.save(os.path.join(output_dir, 'X_normalized.npy'), X_normalized)
    print(f"  X_normalized.npy: {X_normalized.shape}")
    
    np.save(os.path.join(output_dir, 'Y.npy'), Y)
    print(f"  Y.npy: {Y.shape}")
    
    for n_comp, data in X_pca_dict.items():
        filename = f'X_pca_{n_comp}.npy'
        np.save(os.path.join(output_dir, filename), data['X_pca'])
        print(f"  {filename}: {data['X_pca'].shape} (explained var: {data['explained_var']:.4f})")
    
    with open(os.path.join(output_dir, 'preprocessing_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print("  preprocessing_metadata.pkl: saved")


def main():
    print("=" * 60)
    print("PCA PREPROCESSING FOR ISOFORM PREDICTION")
    print("=" * 60)
    
    output_dir = 'preprocessed_data'
    pca_components_list = [100, 500, 1000, 2000]
    
    # Load data
    genes, transcripts = load_data()
    genes, transcripts = align_samples(genes, transcripts)
    
    # Extract matrices
    print("\nExtracting dense matrices...")
    X_raw = densify(genes)
    Y_raw = densify(transcripts)
    print(f"X shape: {X_raw.shape}")
    print(f"Y shape: {Y_raw.shape}")
    
    # Store gene and transcript names
    gene_names = genes.var_names.tolist()
    transcript_names = transcripts.var_names.tolist()
    sample_names = genes.obs_names.tolist()
    
    # Free memory
    del genes, transcripts
    gc.collect()
    
    # Preprocess inputs
    print("\nPreprocessing inputs...")
    X_normalized, norm_mean, norm_std = preprocess_inputs(X_raw)
    del X_raw
    gc.collect()
    
    # Prepare targets
    print("\nPreparing targets...")
    Y = prepare_targets(Y_raw)
    del Y_raw
    gc.collect()
    
    # Apply PCA with different component counts
    print("\nApplying PCA transformations...")
    X_pca_dict = {}
    for n_comp in pca_components_list:
        X_pca, pca_model, explained_var = apply_pca(X_normalized, n_comp)
        X_pca_dict[n_comp] = {
            'X_pca': X_pca,
            'pca_model': pca_model,
            'explained_var': explained_var
        }
    
    # Prepare metadata
    metadata = {
        'n_samples': X_normalized.shape[0],
        'n_genes': X_normalized.shape[1],
        'n_transcripts': Y.shape[1],
        'gene_names': gene_names,
        'transcript_names': transcript_names,
        'sample_names': sample_names,
        'normalization_mean': norm_mean,
        'normalization_std': norm_std,
        'pca_models': {n: {'model': data['pca_model'], 'explained_var': data['explained_var']} 
                       for n, data in X_pca_dict.items()}
    }
    
    # Save outputs
    print("\n" + "-" * 60)
    save_outputs(output_dir, X_normalized, X_pca_dict, Y, metadata)
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nDataset: {metadata['n_samples']} samples")
    print(f"Input features: {metadata['n_genes']} genes")
    print(f"Output features: {metadata['n_transcripts']} transcripts")
    print(f"\nFiles saved to: {output_dir}/")
    print(f"  - X_normalized.npy (log1p + z-score)")
    print(f"  - Y.npy (log1p transformed)")
    for n_comp in pca_components_list:
        if n_comp in X_pca_dict:
            print(f"  - X_pca_{n_comp}.npy")
    print(f"  - preprocessing_metadata.pkl")


if __name__ == "__main__":
    main()