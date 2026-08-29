"""
Module 4: Decision Transformer Module (Policy)

Sequence modeling via transformers conditioned on regime embeddings.
Input: Enriched sequences τ = (R̂t, Z̃t, at) where Z̃ = Z + E_regime
Output: H - latent decision vector

Design: Shared Decision Transformer with continuous regime conditioning
(Avoids parameter explosion and data starvation of 4 separate transformers)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:, :x.size(1), :]


class DecisionTransformer(nn.Module):
    """
    Decision Transformer conditioned on regime embeddings.
    
    Processes sequences of (Return-to-Go, State, Action) tokens.
    Regime embeddings Z̃ = Z + E_regime act as conditioning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        """
        Initialize Decision Transformer.
        
        Args:
            state_dim: Dimension of state embeddings (D from Module 2/3)
            action_dim: Dimension of action space
            embedding_dim: Transformer embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: Feedforward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Input projections
        self.return_to_go_projection = nn.Linear(1, embedding_dim)
        self.state_projection = nn.Linear(state_dim, embedding_dim)
        self.action_projection = nn.Linear(action_dim, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(
        self,
        return_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Decision Transformer.
        
        Args:
            return_to_go: (batch, seq_len, 1) - cumulative future returns
            states: (batch, seq_len, D) - regime-conditioned state embeddings
            actions: (batch, seq_len, action_dim) - previous actions (optional for training)
            
        Returns:
            latent_decision: (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = states.shape
        
        # Project inputs
        rtg_embed = self.return_to_go_projection(return_to_go)  # (batch, seq_len, E)
        state_embed = self.state_projection(states)  # (batch, seq_len, E)
        
        if actions is not None:
            action_embed = self.action_projection(actions)  # (batch, seq_len, E)
        else:
            action_embed = torch.zeros_like(state_embed)
        
        # Interleave tokens: RTG, State, Action, RTG, State, Action, ...
        # Stack all: (batch, seq_len, 3, E)
        tokens = torch.stack([rtg_embed, state_embed, action_embed], dim=2)
        # Flatten: (batch, seq_len*3, E)
        tokens = tokens.view(batch_size, seq_len * 3, self.embedding_dim)
        
        # Add positional encoding
        tokens = self.pos_encoding(tokens)
        
        # Apply transformer
        transformer_output = self.transformer(tokens)  # (batch, seq_len*3, E)
        
        # Extract state outputs (every 3rd token starting at index 1)
        latent_output = transformer_output[:, 1::3, :]  # (batch, seq_len, E)
        
        # Project to output
        latent_decision = self.output_projection(latent_output)  # (batch, seq_len, E)
        
        return latent_decision


class RegimeConditionedTransformer(nn.Module):
    """
    Wrapper around Decision Transformer with explicit regime conditioning.
    
    Enriches state embeddings with regime information before passing to DT.
    """
    
    def __init__(
        self,
        state_dim: int,
        regime_embedding_dim: int,
        action_dim: int,
        transformer_embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize regime-conditioned transformer.
        
        Args:
            state_dim: State dimension (Z)
            regime_embedding_dim: Regime embedding dimension (E_regime)
            action_dim: Action dimension
            transformer_embedding_dim: Transformer embedding dimension
            num_layers: Transformer layers
            num_heads: Attention heads
            ffn_dim: FFN dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.state_dim = state_dim
        self.regime_embedding_dim = regime_embedding_dim
        
        # Combine state and regime embedding
        self.state_regime_combiner = nn.Sequential(
            nn.Linear(state_dim + regime_embedding_dim, state_dim + regime_embedding_dim),
            nn.LayerNorm(state_dim + regime_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Decision transformer
        self.decision_transformer = DecisionTransformer(
            state_dim=state_dim + regime_embedding_dim,
            action_dim=action_dim,
            embedding_dim=transformer_embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        return_to_go: torch.Tensor,
        states: torch.Tensor,
        regime_embeddings: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with explicit regime conditioning.
        
        Args:
            return_to_go: (batch, seq_len, 1)
            states: (batch, seq_len, D) - Z embeddings
            regime_embeddings: (batch, seq_len, D) - E_regime from Module 3
            actions: (batch, seq_len, action_dim) - optional
            
        Returns:
            latent_decision: (batch, seq_len, embedding_dim)
        """
        # Combine state and regime: Z̃ = Z + E_regime
        combined = torch.cat([states, regime_embeddings], dim=-1)
        enriched_states = self.state_regime_combiner(combined)
        
        # Pass through decision transformer
        latent_decision = self.decision_transformer(
            return_to_go=return_to_go,
            states=enriched_states,
            actions=actions,
        )
        
        return latent_decision


class TrajectoryEncoder(nn.Module):
    """Encode full trajectories into decision representations."""
    
    def __init__(
        self,
        state_dim: int,
        regime_embedding_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        """Initialize trajectory encoder."""
        super().__init__()
        self.transformer = RegimeConditionedTransformer(
            state_dim=state_dim,
            regime_embedding_dim=regime_embedding_dim,
            action_dim=action_dim,
            transformer_embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    
    def encode_trajectory(
        self,
        return_to_go: torch.Tensor,  # (batch, seq_len, 1)
        states: torch.Tensor,  # (batch, seq_len, D)
        regime_embeddings: torch.Tensor,  # (batch, seq_len, D)
        actions: Optional[torch.Tensor] = None,  # (batch, seq_len, action_dim)
    ) -> torch.Tensor:
        """Encode trajectory into latent decision vectors."""
        return self.transformer(
            return_to_go=return_to_go,
            states=states,
            regime_embeddings=regime_embeddings,
            actions=actions,
        )
    
    @staticmethod
    def construct_return_to_go(
        future_returns: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Construct return-to-go from future returns.
        
        Return-to-go at time t = sum of discounted future returns from t to T.
        
        Args:
            future_returns: (batch, seq_len) - per-timestep returns
            
        Returns:
            return_to_go: (batch, seq_len, 1)
        """
        batch_size, seq_len = future_returns.shape
        device = future_returns.device
        
        return_to_go = torch.zeros(batch_size, seq_len, 1, device=device)
        
        # Compute cumulative discounted returns (gamma=0.99)
        gamma = 0.99
        for t in range(seq_len - 1, -1, -1):
            if t == seq_len - 1:
                return_to_go[:, t, 0] = future_returns[:, t]
            else:
                return_to_go[:, t, 0] = future_returns[:, t] + gamma * return_to_go[:, t+1, 0]
        
        return return_to_go


class ActionPolicyHead(nn.Module):
    """Policy head that outputs actions from latent decisions."""
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Initialize policy head.
        
        Args:
            latent_dim: Latent decision vector dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, latent_decision: torch.Tensor) -> torch.Tensor:
        """
        Compute actions from latent decisions.
        
        Args:
            latent_decision: (batch, seq_len, latent_dim)
            
        Returns:
            actions: (batch, seq_len, action_dim)
        """
        return self.policy_head(latent_decision)
