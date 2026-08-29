"""
Module 2: Cross-Asset Hypergraph Encoder

Map cross-asset dependencies using hypergraph structure.
Input: X ∈ R^(N×T×F)
Output: Z ∈ R^(N×T×D)

Hypergraph: G = (V, E) with incidence matrix H ∈ {0,1}^(N×K)
Two-step exchange: Stocks → Sectors → Stocks
Degree normalization prevents large sectors from dominating.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HypergraphEncoder(nn.Module):
    """Hypergraph-based cross-asset encoder."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_sectors: int = 16,
        num_assets: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize hypergraph encoder.
        
        Args:
            input_dim: Input feature dimension (F)
            output_dim: Output embedding dimension (D)
            num_sectors: Number of latent sectors (K)
            num_assets: Number of assets (N)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_sectors = num_sectors
        self.num_assets = num_assets
        
        # Incidence matrix H ∈ {0,1}^(N×K): learnable sector assignments
        self.register_buffer(
            "H",
            torch.zeros(num_assets, num_sectors)
        )
        self._init_incidence_matrix()
        
        # Stock-to-sector encoder
        self.stock_to_sector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Sector aggregation & transformation
        self.sector_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Sector-to-stock decoder
        self.sector_to_stock = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Final projection
        self.output_projection = nn.Linear(output_dim, output_dim)
        
    def _init_incidence_matrix(self):
        """Initialize H with approximately equal sector assignments."""
        assignments = np.arange(self.num_assets) % self.num_sectors
        for n in range(self.num_assets):
            self.H[n, assignments[n]] = 1.0
    
    def forward(self, X: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: two-step hypergraph exchange.
        
        Args:
            X: (batch, N, T, F) or (N, T, F) - stock features
            validity_mask: (batch, N, T) or (N, T) - binary mask for valid data
            
        Returns:
            Z: (batch, N, T, D) or (N, T, D) - context-aware embeddings
        """
        is_3d = X.dim() == 3
        if is_3d:
            X = X.unsqueeze(0)
            validity_mask = validity_mask.unsqueeze(0)
            
        batch_size, N, T, F = X.shape
        device = X.device
        
        # Reshape for processing: (batch * N * T, F) → (batch * N * T, D)
        X_flat = X.reshape(batch_size * N * T, F)
        
        # Step 1: Stocks send to sectors
        stock_embeddings = self.stock_to_sector(X_flat)  # (batch * N * T, D)
        stock_embeddings = stock_embeddings.view(batch_size, N, T, self.output_dim)
        
        # Reshape: (batch, N, T, D) → (batch, T, N, D)
        stock_emb_t = stock_embeddings.permute(0, 2, 1, 3)
        
        # Aggregate stocks to sectors with degree normalization
        # H: (N, K)
        # batched einsum: (batch, T, N, D) @ (N, K) = (batch, T, K, D)
        sector_emb = torch.einsum('btnd,nk->btkd', stock_emb_t, self.H)  # (batch, T, K, D)
        
        # Degree normalization: d_k = sum_n H[n,k]
        sector_degrees = self.H.sum(dim=0)  # (K,)
        sector_degrees = torch.clamp(sector_degrees, min=1e-8)
        
        # Normalize by degree
        sector_emb = sector_emb / sector_degrees.view(1, 1, self.num_sectors, 1)
        
        # Transform sectors
        sector_emb_flat = sector_emb.reshape(batch_size * T * self.num_sectors, self.output_dim)
        sector_emb_transformed = self.sector_transform(sector_emb_flat)  # (batch*T*K, D)
        sector_emb = sector_emb_transformed.view(batch_size, T, self.num_sectors, self.output_dim)
        
        # Step 2: Sectors send back to stocks
        # sector_emb: (batch, T, K, D), H: (N, K)
        # (batch, T, K, D) @ (K, N).T = (batch, T, N, D)
        stock_context = torch.einsum('btkd,nk->btnd', sector_emb, self.H)  # (batch, T, N, D)
        
        # Decode sector context back to stocks
        stock_context_flat = stock_context.reshape(batch_size * T * N, self.output_dim)
        updated_embeddings = self.sector_to_stock(stock_context_flat)  # (batch*T*N, D)
        updated_embeddings = updated_embeddings.view(batch_size, T, N, self.output_dim)
        
        # Residual connection + output projection
        Z = updated_embeddings.permute(0, 2, 1, 3)  # Back to (batch, N, T, D)
        Z = Z + stock_embeddings  # Residual
        Z = self.output_projection(Z)  # Final projection
        
        # Apply validity mask: zero out invalid positions
        mask = validity_mask.unsqueeze(-1)  # (batch, N, T, 1)
        Z = Z * mask
        
        if is_3d:
            Z = Z.squeeze(0)
            
        return Z


class SectorAssignmentLearner(nn.Module):
    """Learn soft sector assignments instead of fixed H."""
    
    def __init__(
        self,
        num_assets: int,
        num_sectors: int,
        asset_features_dim: int,
    ):
        """
        Learn soft sector assignments.
        
        Args:
            num_assets: Number of assets
            num_sectors: Number of sectors
            asset_features_dim: Dimension of asset features for assignment
        """
        super().__init__()
        self.num_assets = num_assets
        self.num_sectors = num_sectors
        
        # Learnable sector embeddings
        self.sector_embeddings = nn.Parameter(
            torch.randn(num_sectors, asset_features_dim) / np.sqrt(asset_features_dim)
        )
        
        # Asset-to-sector projection
        self.asset_projection = nn.Linear(asset_features_dim, asset_features_dim)
    
    def forward(self, asset_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft sector assignments.
        
        Args:
            asset_features: (N, D) - asset embedding features
            
        Returns:
            H_soft: (N, K) - soft assignment probabilities
            H_hard: (N, K) - hard assignment (for visualization)
        """
        # Project assets
        projected = self.asset_projection(asset_features)  # (N, D)
        
        # Compute assignment probabilities
        # (N, D) @ (D, K) = (N, K)
        logits = torch.matmul(projected, self.sector_embeddings.T)
        H_soft = F.softmax(logits, dim=1)  # (N, K)
        
        # Hard assignment (argmax)
        H_hard = F.one_hot(H_soft.argmax(dim=1), num_classes=self.num_sectors).float()
        
        return H_soft, H_hard


class MultiLayerHypergraphEncoder(nn.Module):
    """Stack multiple hypergraph layers for deeper encoding."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_sectors: int = 16,
        num_assets: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-layer encoder.
        
        Args:
            input_dim: Initial feature dimension
            output_dim: Final embedding dimension
            num_sectors: Number of sectors per layer
            num_assets: Number of assets
            num_layers: Number of hypergraph layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        
        # First layer
        self.layers = nn.ModuleList()
        self.layers.append(
            HypergraphEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                num_sectors=num_sectors,
                num_assets=num_assets,
                dropout=dropout,
            )
        )
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(
                HypergraphEncoder(
                    input_dim=output_dim,
                    output_dim=output_dim,
                    num_sectors=num_sectors,
                    num_assets=num_assets,
                    dropout=dropout,
                )
            )
    
    def forward(self, X: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multiple hypergraph layers.
        
        Args:
            X: (N, T, F) - stock features
            validity_mask: (N, T) - validity mask
            
        Returns:
            Z: (N, T, D) - encoded embeddings
        """
        Z = X
        for layer in self.layers:
            Z = layer(Z, validity_mask)
        
        return Z


class DependencyAnalyzer:
    """Analyze cross-asset dependencies from encoder."""
    
    @staticmethod
    def compute_dependency_matrix(
        Z: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute asset-asset dependency matrix from embeddings.
        
        Args:
            Z: (N, T, D) - embeddings
            validity_mask: (N, T) - validity mask
            
        Returns:
            dep_matrix: (N, N) - dependency correlation matrix
        """
        N, T, D = Z.shape
        
        # Mask invalid timesteps
        mask = validity_mask.unsqueeze(-1)  # (N, T, 1)
        Z_masked = Z * mask
        
        # Compute pairwise similarity per timestep
        # (N, T, D) → average over time → (N, D)
        Z_avg = Z_masked.sum(dim=1) / validity_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Normalize
        Z_avg = F.normalize(Z_avg, p=2, dim=1)
        
        # Compute similarity: (N, D) @ (D, N) = (N, N)
        dep_matrix = torch.mm(Z_avg, Z_avg.T)
        
        return dep_matrix
    
    @staticmethod
    def get_sector_assignments(H: torch.Tensor) -> dict:
        """Extract sector assignments from incidence matrix."""
        N, K = H.shape
        sectors = {}
        
        for k in range(K):
            members = (H[:, k] > 0).nonzero(as_tuple=True)[0].tolist()
            if members:
                sectors[f"sector_{k}"] = members
        
        return sectors
