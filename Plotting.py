import pickle
import sys
import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import re

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import streamlit as st

def plot_histogram(predicted_time_list, cell_types):
    fig, ax = plt.subplots()

    if len(predicted_time_list) == 1:
        ax.hist(predicted_time_list[0], bins=24, range=(0, 24), alpha=0.5, label="Predicted TOD ("+cell_types[0]+')', edgecolor='black')
    else:
        for i in range(len(predicted_time_list)):
            predict_time = predicted_time_list[i]
            cell_type = cell_types[i]
            ax.hist(predict_time, bins=24, range=(0, 24), alpha=0.5, label="Predicted TOD ("+cell_type+')', edgecolor='black')

    # ax.set_xlabel("Time (h)")
    # ax.set_ylabel("Count")
    # ax.set_title("Circadian Distribution")
    ax.set_xlabel("Time of Day (Hour)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Time of Day (TOD)")
    
    ax.set_xticks(range(0, 25, 1))  
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    return fig

def plot_UMAP(test_adata):
    min_cells=3,
    min_genes=200,
    n_top_genes=1000,
    n_pcs=50,
    random_state=42

    with st.status("🔍 Calculating UMAP...", expanded=True) as status:
        st.write("⏳ Filtering anndata ...")
        sc.pp.filter_cells(test_adata, min_genes=min_genes)
        sc.pp.filter_genes(test_adata, min_cells=min_cells)
        st.write("⏳ Normalization and data scaling ...")
        sc.pp.normalize_total(test_adata, target_sum=1e4)
        sc.pp.log1p(test_adata)
        sc.pp.highly_variable_genes(test_adata, n_top_genes=2000)

        test_adata = test_adata[:, test_adata.var.highly_variable]
        sc.pp.scale(test_adata, max_value=10)
        st.write("⏳ Running PCA ...")
        sc.tl.pca(test_adata, svd_solver='arpack', n_comps=50, random_state=random_state)
        sc.pp.neighbors(test_adata, n_pcs=50)
        st.write("⏳ Running UMAP ...")
        sc.tl.umap(test_adata, random_state=random_state)
        status.update(label="✅ UMAP calculation complete!", state="complete", expanded=False)

    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(8, 6))

    sc.pl.umap(test_adata, color='Predict time',show=False,ax=fig.gca())

    plt.tight_layout()
    
    return fig, test_adata