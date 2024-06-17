import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from typing import Union

def reduce_dimensionality(features: np.ndarray, dimension_reduction: str = "umap") -> np.ndarray:
    """
    Reduce the dimensionality of features using techniques such as UMAP or t-SNE.

    Args:
        features (np.ndarray): The feature array to reduce. Shape (n_samples, n_features).
        dimension_reduction (str): The dimensionality reduction method to use ("umap" or "tsne").

    Returns:
        np.ndarray: The reduced features. Shape (n_samples, 2).
    """
    if not isinstance(features, np.ndarray):
        raise ValueError("Input features should be a numpy.ndarray")

    if dimension_reduction == "umap":
        reducer = UMAP(n_components=2)
    elif dimension_reduction == "tsne":
        reducer = TSNE(n_components=2)
    elif dimension_reduction == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unsupported dimension reduction method: {dimension_reduction}")

    # Perform dimensionality reduction
    reduced_features = reducer.fit_transform(features)
    return reduced_features

def plot_reduced_features(reduced_features: np.ndarray, labels: np.ndarray, title: str = "Feature Visualization", save_flag=False) -> None:
    """
    Plot the reduced features.

    Args:
        reduced_features (np.ndarray): The reduced features to plot. Shape (n_samples, 2).
        title (str): The title of the plot.

    Returns:
        None
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(10, 8))
    for label, color in zip(unique_labels, colors):
        indices = labels == label
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], 
                    color=color, label=f'Label {label}', alpha=0.6)
    # plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='Spectral', s=5)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # plt.colorbar()
    plt.legend()
    if save_flag:
        plt.savefig('outputs/feature_visualization.png')
    plt.show()
