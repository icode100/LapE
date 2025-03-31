# imports 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
from utils import *
from dataloader import *

class LaplaceIntersection(nn.Module):
    def __init__(self, dim):
        super(LaplaceIntersection, self).__init__()
        self.dim = dim
        # Linear layers for computing attention over mean (mu)
        self.layer_mu1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_mu2 = nn.Linear(self.dim, self.dim)
        # Linear layers for computing attention over scale (b)
        self.layer_b1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_b2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_mu1.weight)
        nn.init.xavier_uniform_(self.layer_mu2.weight)
        nn.init.xavier_uniform_(self.layer_b1.weight)
        nn.init.xavier_uniform_(self.layer_b2.weight)

    def forward(self, mu_embeddings, b_embeddings):
        """
        Implements the intersection (AND) operation on Laplace embeddings.
        
        Inputs:
          - mu_embeddings: Tensor of shape (num_conj, batch_size, dim) representing means.
          - b_embeddings: Tensor of shape (num_conj, batch_size, dim) representing scale parameters.
          
        Outputs:
          - mu_out: Mean of the resulting intersection embedding.
          - b_out: Scale (uncertainty) of the resulting intersection embedding.
        """
        # Concatenate mean and scale embeddings along last dimension
        all_embeddings = torch.cat([mu_embeddings, b_embeddings], dim=-1)

        # Compute attention for mu (mean)
        layer_mu = F.relu(self.layer_mu1(all_embeddings))  
        attention_mu = F.softmax(self.layer_mu2(layer_mu), dim=0)  

        # Compute attention for b (scale)
        layer_b = F.relu(self.layer_b1(all_embeddings))  
        attention_b = F.softmax(self.layer_b2(layer_b), dim=0)  

        # Compute new mean using weighted sum (weighted by uncertainty)
        mu_out = torch.sum(attention_mu * mu_embeddings, dim=0)
        # Compute new scale using harmonic mean-like aggregation
        b_out = torch.sum(attention_b * b_embeddings, dim=0)  

        # Clamping the scale to avoid extreme values
        b_out = torch.clamp(b_out, min=1e-4, max=1.0)

        return mu_out, b_out

class LaplaceUnion(nn.Module):
    def __init__(self, dim, projection_regularizer, drop):
        super(LaplaceUnion, self).__init__()
        self.dim = dim
        # Layers for mean (mu)
        self.layer_mu1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_mu2 = nn.Linear(self.dim, self.dim // 2)
        self.layer_mu3 = nn.Linear(self.dim // 2, self.dim)
        # Layers for scale (b)
        self.layer_b1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_b2 = nn.Linear(self.dim, self.dim // 2)
        self.layer_b3 = nn.Linear(self.dim // 2, self.dim)

        self.projection_regularizer = projection_regularizer
        self.drop = nn.Dropout(p=drop)

        nn.init.xavier_uniform_(self.layer_mu1.weight)
        nn.init.xavier_uniform_(self.layer_mu2.weight)
        nn.init.xavier_uniform_(self.layer_mu3.weight)
        nn.init.xavier_uniform_(self.layer_b1.weight)
        nn.init.xavier_uniform_(self.layer_b2.weight)
        nn.init.xavier_uniform_(self.layer_b3.weight)

    def forward(self, mu_embeddings, b_embeddings):
        """
        Implements the union (OR) operation on Laplace embeddings.
        Inputs:
          - mu_embeddings: Tensor of shape (num_disj, batch_size, dim) for means.
          - b_embeddings: Tensor of shape (num_disj, batch_size, dim) for scale parameters.
          
        Outputs:
          - mu_out: Mean of the resulting union embedding.
          - b_out: Scale (uncertainty) of the resulting union embedding.
        """
        # Concatenate means and scales along the last dimension
        all_embeddings = torch.cat([mu_embeddings, b_embeddings], dim=-1)
        
        # Compute attention for mu (mean)
        layer_mu = F.relu(self.layer_mu1(all_embeddings))  
        layer_mu = F.relu(self.layer_mu2(layer_mu))
        attention_mu = F.softmax(self.drop(self.layer_mu3(layer_mu)), dim=0)  

        # Compute attention for b (scale)
        layer_b = F.relu(self.layer_b1(all_embeddings))  
        layer_b = F.relu(self.layer_b2(layer_b))
        attention_b = F.softmax(self.drop(self.layer_b3(layer_b)), dim=0)  

        # Compute new mean and scale
        mu_out = torch.sum(attention_mu * mu_embeddings, dim=0)  # Average of means
        b_out = torch.sum(b_embeddings, dim=0)  # Sum of uncertainties

        # Clamping the scale to avoid extreme values
        b_out = torch.clamp(b_out, min=1e-4, max=1.0)

        return mu_out, b_out

class LaplaceProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(LaplaceProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Neural network for processing the mean (mu)
        self.layer_mu1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  
        self.layer_mu0 = nn.Linear(self.hidden_dim, self.entity_dim)  # Final layer
        
        # Neural network for processing the scale (b)
        self.layer_b1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  
        self.layer_b0 = nn.Linear(self.hidden_dim, self.entity_dim)  

        # Additional layers for deeper networks
        for nl in range(2, num_layers + 1):
            setattr(self, f"layer_mu{nl}", nn.Linear(self.hidden_dim, self.hidden_dim))
            setattr(self, f"layer_b{nl}", nn.Linear(self.hidden_dim, self.hidden_dim))

        # Xavier Initialization
        for nl in range(1, num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer_mu{nl}").weight)
            nn.init.xavier_uniform_(getattr(self, f"layer_b{nl}").weight)

        self.projection_regularizer = projection_regularizer

    def forward(self, mu_embedding, b_embedding, mu_embedding_r, b_embedding_r):
        """
        Implements the projection operation: moving from one entity to another via a relation.
        
        Inputs:
          - mu_embedding: Mean of entity embeddings.
          - b_embedding: Scale (uncertainty) of entity embeddings.
          - mu_embedding_r: Mean of relation embeddings.
          - b_embedding_r: Scale (uncertainty) of relation embeddings.
        
        Outputs:
          - mu_out: Projected mean embedding.
          - b_out: Projected scale (uncertainty).
        """
        # Concatenate entity and relation embeddings
        x_mu = torch.cat([mu_embedding, mu_embedding_r], dim=-1)
        x_b = torch.cat([b_embedding, b_embedding_r], dim=-1)

        # Pass through deep network for mu (mean)
        for nl in range(1, self.num_layers + 1):
            x_mu = F.relu(getattr(self, f"layer_mu{nl}")(x_mu))
        mu_out = self.layer_mu0(x_mu)
        mu_out = self.projection_regularizer(mu_out)

        # Pass through deep network for b (scale)
        for nl in range(1, self.num_layers + 1):
            x_b = F.relu(getattr(self, f"layer_b{nl}")(x_b))
        b_out = self.layer_b0(x_b)
        b_out = self.projection_regularizer(b_out)

        # Enforce positivity constraint on scale (uncertainty)
        b_out = torch.clamp(b_out, min=1e-4, max=1.0)

        return mu_out, b_out

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)