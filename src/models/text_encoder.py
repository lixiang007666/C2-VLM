"""
Text encoder implementation for C2-VLM.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig


class TextEncoder(nn.Module):
    """
    Text encoder using pre-trained language models.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 77,
        embed_dim: int = 768,
        freeze_backbone: bool = False,
        use_pooler_output: bool = False,
        pooling_strategy: str = "cls",  # "cls", "mean", "max"
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: Name of the pre-trained language model
            max_length: Maximum sequence length
            embed_dim: Output embedding dimension
            freeze_backbone: Whether to freeze backbone parameters
            use_pooler_output: Whether to use model's pooler output
            pooling_strategy: Strategy for pooling token embeddings
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.use_pooler_output = use_pooler_output
        self.pooling_strategy = pooling_strategy
        
        # Load pre-trained model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.config.hidden_size
        
        # Projection layer to desired embedding dimension
        if self.hidden_size != embed_dim:
            self.projection = nn.Linear(self.hidden_size, embed_dim)
        else:
            self.projection = nn.Identity()
            
        self.output_dim = embed_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of text encoder.
        
        Args:
            input_ids: Tokenized text [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Text features [B, embed_dim]
        """
        # Get model outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract features based on strategy
        if self.use_pooler_output and hasattr(outputs, 'pooler_output'):
            # Use model's pooler output if available
            features = outputs.pooler_output  # [B, hidden_size]
        else:
            # Use custom pooling strategy
            last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_size]
            features = self._pool_features(last_hidden_state, attention_mask)
        
        # Project to desired dimension
        features = self.projection(features)  # [B, embed_dim]
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        return features
    
    def _pool_features(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool token-level features to sentence-level representation.
        
        Args:
            hidden_states: Token hidden states [B, seq_len, hidden_size]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Pooled features [B, hidden_size]
        """
        if self.pooling_strategy == "cls":
            # Use CLS token (first token)
            return hidden_states[:, 0]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Mask padded tokens
                hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                # Compute mean over valid tokens
                sum_hidden = hidden_states.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True)
                return sum_hidden / sum_mask.clamp(min=1)
            else:
                return hidden_states.mean(dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                # Set padded positions to very negative values
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, 
                    float('-inf')
                )
            return hidden_states.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text strings to features.
        
        Args:
            texts: List of text strings
            
        Returns:
            Text features [B, embed_dim]
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward pass
        with torch.no_grad():
            features = self.forward(input_ids, attention_mask)
        
        return features
    
    def get_token_embeddings(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get token-level embeddings for detailed analysis.
        
        Args:
            input_ids: Tokenized text [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Token embeddings [B, seq_len, embed_dim]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        token_features = outputs.last_hidden_state  # [B, seq_len, hidden_size]
        
        # Project token features
        token_features = self.projection(token_features)  # [B, seq_len, embed_dim]
        token_features = self.layer_norm(token_features)
        
        return token_features


class RobertaTextEncoder(TextEncoder):
    """RoBERTa-based text encoder."""
    
    def __init__(self, model_name: str = "roberta-base", **kwargs):
        kwargs['model_name'] = model_name
        super().__init__(**kwargs)


class DistilBertTextEncoder(TextEncoder):
    """DistilBERT-based text encoder for efficiency."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", **kwargs):
        kwargs['model_name'] = model_name
        super().__init__(**kwargs)