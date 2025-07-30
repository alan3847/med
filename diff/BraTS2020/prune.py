import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import SpectralClustering

# Assuming this is the import path for the Diff-UNet model
from train import DiffUNet

def feature_group_clustering(model, layer_name, num_clusters):
    """
    Perform feature group clustering on the specified layer to reduce redundancy.
    :param model: The model to prune.
    :param layer_name: The name of the layer to prune.
    :param num_clusters: The number of clusters to form.
    :return: Clustered weights.
    """
    # Retrieve the weights of the specified layer
    layer = getattr(model, layer_name)
    weight = layer.weight.data.cpu().numpy()
    
    # Reshape the weights into a 2D matrix
    weight_matrix = weight.reshape(weight.shape[0], -1)
    
    # Apply spectral clustering
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
    cluster_labels = clustering.fit_predict(weight_matrix)
    
    # Compute the weights for each cluster center
    clustered_weights = np.zeros_like(weight_matrix)
    for cluster_idx in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        clustered_weights[cluster_indices] = np.mean(weight_matrix[cluster_indices], axis=0)
    
    # Reshape the clustered weights back to the original shape
    clustered_weights = clustered_weights.reshape(weight.shape)
    
    return torch.tensor(clustered_weights, dtype=torch.float32).to(layer.weight.device)

def spectral_variance_rebalancing(model, layer_name, clustered_weights):
    """
    Perform spectral variance rebalancing on the clustered weights.
    :param model: The model to rebalance.
    :param layer_name: The name of the layer to rebalance.
    :param clustered_weights: The clustered weights.
    :return: Rebalanced weights.
    """
    # Retrieve the weights and normalization layer
    layer = getattr(model, layer_name)
    norm_layer = getattr(model, f"{layer_name}_norm")
    
    # Compute the output variance for each cluster center
    weight = layer.weight.data
    norm_factor = norm_layer.weight.data
    cluster_centers = clustered_weights
    
    # Assume the variance of each cluster center is 1
    variance = 1.0
    rebalanced_weights = cluster_centers * (variance / torch.norm(cluster_centers, dim=1, keepdim=True))
    
    return rebalanced_weights

def prune_model(model, layer_names, num_clusters_list):
    """
    Prune the model using feature group clustering and spectral variance rebalancing.
    :param model: The model to prune.
    :param layer_names: List of layer names to prune.
    :param num_clusters_list: List of the number of clusters for each layer.
    :return: The pruned model.
    """
    for layer_name, num_clusters in zip(layer_names, num_clusters_list):
        clustered_weights = feature_group_clustering(model, layer_name, num_clusters)
        rebalanced_weights = spectral_variance_rebalancing(model, layer_name, clustered_weights)
        
        # Update the model weights
        layer = getattr(model, layer_name)
        layer.weight.data = rebalanced_weights
    
    return model

if __name__ == "__main__":
    # Load the model
    model = DiffUNet()
    
    # Define the layers to prune and the number of clusters for each layer
    layer_names = ["encoder_block1", "encoder_block2", "decoder_block1", "decoder_block2"]
    num_clusters_list = [16, 32, 16, 32]
    
    # Prune the model
    pruned_model = prune_model(model, layer_names, num_clusters_list)
    
    # Save the pruned model
    torch.save(pruned_model.state_dict(), "pruned_diff_unet.pth")
    print("Pruning completed and the pruned model is saved to pruned_diff_unet.pth")