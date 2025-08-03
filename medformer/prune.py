import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import get_model  # 假设模型定义在 model 文件夹中
from dataset_conversion import get_dataset  # 假设数据集处理代码在 dataset_conversion 文件夹中
from utils import get_parser  # 假设工具函数在 utils 文件夹中

class FeatureGroupClusteringPruner:
    def __init__(self, model, device, sigma=1.0):
        self.model = model
        self.device = device
        self.sigma = sigma
        self.cluster_centers = None

    def spectral_clustering(self, weight_matrix, num_clusters):
        # Compute similarity matrix using Gaussian kernel
        similarity_matrix = torch.exp(-torch.cdist(weight_matrix, weight_matrix) ** 2 / (2 * self.sigma ** 2))
        
        # Compute graph Laplacian matrix
        degree_matrix = torch.diag(similarity_matrix.sum(dim=1))
        laplacian_matrix = degree_matrix - similarity_matrix
        
        # Perform spectral clustering
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
        cluster_centers = eigenvectors[:, :num_clusters]
        
        return cluster_centers

    def cluster_weights(self, layer_index, num_clusters):
        weight_matrix = self.model.swinViT.layers[layer_index].blocks[0].attn.qkv.weight
        self.cluster_centers = self.spectral_clustering(weight_matrix, num_clusters)

    def apply_pruning(self):
        # Apply clustering to each layer
        for layer_index in range(len(self.model.swinViT.layers)):
            self.cluster_weights(layer_index, num_clusters=16)  # Example: 16 clusters per layer

        # Update model weights with cluster centers
        for layer_index in range(len(self.model.swinViT.layers)):
            weight_matrix = self.model.swinViT.layers[layer_index].blocks[0].attn.qkv.weight
            updated_weights = torch.matmul(self.cluster_centers, weight_matrix)
            self.model.swinViT.layers[layer_index].blocks[0].attn.qkv.weight.data = updated_weights

    def spectral_variance_rebalancing(self):
        # Compute intra-cluster spectral correlation coefficient
        for t in range(self.cluster_centers.shape[1]):
            cluster_indices = torch.where(self.cluster_centers[:, t] == 1)[0]
            if len(cluster_indices) > 1:
                lambda_t = torch.mean(torch.corrcoef(self.cluster_centers[cluster_indices, :]))
            else:
                lambda_t = 0
            # Adjust normalization scaling factor
            self.model.swinViT.layers[layer_index].blocks[0].attn.qkv.norm.weight.data *= torch.sqrt(torch.tensor(len(cluster_indices)) / (len(cluster_indices) + (len(cluster_indices) ** 2 - len(cluster_indices)) * lambda_t))

    def prune_model(self):
        self.apply_pruning()
        self.spectral_variance_rebalancing()