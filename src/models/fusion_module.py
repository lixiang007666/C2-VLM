"""
Fusion module for contextual multimodal alignment in C2-VLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class FusionModule(nn.Module):
    """
    Contextual fusion module that aligns vision and text representations
    with cross-modal attention and contrastive learning.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        projection_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
        use_cross_attention: bool = True,
    ):
        """
        Initialize fusion module.
        
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            hidden_dim: Hidden dimension for fusion
            projection_dim: Dimension of final projections
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for contrastive learning
            use_cross_attention: Whether to use cross-modal attention
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_attention_heads = num_attention_heads
        self.use_cross_attention = use_cross_attention
        
        # Input projections to common hidden dimension
        self.vision_input_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_input_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-modal attention layers
        if use_cross_attention:
            self.vision_to_text_attention = CrossModalAttention(
                hidden_dim, num_attention_heads, dropout
            )
            self.text_to_vision_attention = CrossModalAttention(
                hidden_dim, num_attention_heads, dropout
            )
        
        # Contextual fusion layers
        self.fusion_layers = nn.ModuleList([
            FusionLayer(hidden_dim, num_attention_heads, dropout)
            for _ in range(2)
        ])
        
        # Output projections for contrastive learning
        self.vision_output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.text_output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        
        # Layer normalization
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of fusion module.
        
        Args:
            vision_features: Vision features [B, vision_dim]
            text_features: Text features [B, text_dim]  
            text_attention_mask: Text attention mask [B, seq_len]
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        batch_size = vision_features.size(0)
        
        # Project to common hidden dimension
        vision_hidden = self.vision_input_proj(vision_features)  # [B, hidden_dim]
        text_hidden = self.text_input_proj(text_features)  # [B, hidden_dim]
        
        # Add batch dimension for attention computation
        vision_hidden = vision_hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        text_hidden = text_hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        
        attention_weights = {}
        
        # Cross-modal attention
        if self.use_cross_attention:
            # Vision attending to text
            vision_attended, v2t_weights = self.vision_to_text_attention(
                vision_hidden, text_hidden, text_hidden
            )
            attention_weights['vision_to_text'] = v2t_weights
            
            # Text attending to vision  
            text_attended, t2v_weights = self.text_to_vision_attention(
                text_hidden, vision_hidden, vision_hidden
            )
            attention_weights['text_to_vision'] = t2v_weights
            
            # Update features with attended representations
            vision_hidden = vision_hidden + vision_attended
            text_hidden = text_hidden + text_attended
        
        # Apply fusion layers
        for i, fusion_layer in enumerate(self.fusion_layers):
            vision_hidden, text_hidden, fusion_weights = fusion_layer(
                vision_hidden, text_hidden
            )
            attention_weights[f'fusion_layer_{i}'] = fusion_weights
        
        # Remove sequence dimension
        vision_fused = vision_hidden.squeeze(1)  # [B, hidden_dim]
        text_fused = text_hidden.squeeze(1)  # [B, hidden_dim]
        
        # Apply layer normalization
        vision_fused = self.vision_norm(vision_fused)
        text_fused = self.text_norm(text_fused)
        
        # Create fused representation by concatenation
        fused_features = torch.cat([vision_fused, text_fused], dim=1)  # [B, 2*hidden_dim]
        
        # Project to final embedding space and normalize
        vision_embeddings = F.normalize(self.vision_output_proj(vision_fused), dim=1)
        text_embeddings = F.normalize(self.text_output_proj(text_fused), dim=1)
        
        # Compute contrastive logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * vision_embeddings @ text_embeddings.t()
        logits_per_text = logit_scale * text_embeddings @ vision_embeddings.t()
        
        return {
            'fused_features': fused_features,
            'vision_fused': vision_fused,
            'text_fused': text_fused,
            'vision_embeddings': vision_embeddings,
            'text_embeddings': text_embeddings,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'attention_weights': attention_weights,
            'logit_scale': logit_scale,
        }


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [B, seq_len, hidden_dim]
            key: Key tensor [B, seq_len, hidden_dim]
            value: Value tensor [B, seq_len, hidden_dim]
            
        Returns:
            Attended features and attention weights
        """
        attended, weights = self.attention(query, key, value)
        attended = self.norm(attended + self.dropout(attended))
        return attended, weights


class FusionLayer(nn.Module):
    """Contextual fusion layer with self-attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention for joint vision-text representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.vision_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        vision_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_features: Vision features [B, 1, hidden_dim]
            text_features: Text features [B, 1, hidden_dim]
            
        Returns:
            Updated vision features, text features, and attention weights
        """
        # Concatenate for joint attention
        joint_features = torch.cat([vision_features, text_features], dim=1)  # [B, 2, hidden_dim]
        
        # Self-attention over joint representation
        attended_joint, attention_weights = self.self_attention(
            joint_features, joint_features, joint_features
        )
        attended_joint = self.norm1(attended_joint + self.dropout(attended_joint))
        
        # Split back to vision and text
        attended_vision = attended_joint[:, :1]  # [B, 1, hidden_dim]
        attended_text = attended_joint[:, 1:]  # [B, 1, hidden_dim]
        
        # Apply feed-forward networks with residual connections
        vision_out = attended_vision + self.dropout(self.vision_ffn(attended_vision))
        vision_out = self.norm2(vision_out)
        
        text_out = attended_text + self.dropout(self.text_ffn(attended_text))
        text_out = self.norm3(text_out)
        
        return vision_out, text_out, attention_weights