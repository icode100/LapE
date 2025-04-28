from models import *
from dataloader import *
from utils import *
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

class KGReasoningLapE(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, gamma_mode=None, drop=0.):
        super(KGReasoningLapE, self).__init__()

        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.is_u = False
        self.use_cuda = use_cuda
        self.batch_entity_range = (
            torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda()
            if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)
        )
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        # Each entity embedding is represented as a Laplace distribution:
        # first half is μ (mean) and second half is b (scale/uncertainty)
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2))
        self.entity_regularizer = Regularizer(1, 0.15, 1e9)
        self.projection_regularizer = Regularizer(1, 0.15, 1e9)

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )

        # Standard relation embeddings remain for additional relation information
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )

        # For relations, we now maintain separate parameters for μ and b
        self.mu_relation = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.mu_relation,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )
        self.b_relation = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.b_relation,
            a=-3 * self.embedding_range.item(),
            b=3 * self.embedding_range.item()
        )

        self.modulus = nn.Parameter(torch.Tensor([1 * self.embedding_range.item()]), requires_grad=True)

        # gamma_mode returns (hidden_dim, num_layers)
        hidden_dim, num_layers = gamma_mode
        # Use our Laplace-based modules for logical operations:
        self.center_net = LaplaceIntersection(self.entity_dim)
        self.projection_net = LaplaceProjection(self.entity_dim,
                                                 self.relation_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)
        self.union_net = LaplaceUnion(self.entity_dim, self.projection_regularizer, drop)
    def sample_laplace(self, mu, b):
        # u ~ Uniform(-0.5, 0.5)
        u = torch.rand_like(mu) - 0.5
        # z ~ Laplace(0, 1)
        z = -torch.sign(u) * torch.log(1 - 2 * torch.abs(u) + 1e-12)
        # reparameterized sample
        return mu + b * z
    def embed_query_lape(self, queries, query_structure, idx):
        '''
        Iteratively embeds a batch of queries with the same structure using Laplace-based embeddings.
        Each entity is represented as a Laplace distribution: (mu, b).
        '''
        # Special case handling (if needed)
        if query_structure == ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)):
            aa = 1  # (dummy code, as in original)
        
        # Determine if the current query structure is purely relational (only 'r' and 'n')
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break

        if all_relation_flag:
            # Base case: query structure starts with an entity
            if query_structure[0] == 'e':
                ent_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                # Split the entity embedding into mean (mu) and scale (b)
                mu_embedding, b_embedding = torch.chunk(ent_embedding, 2, dim=-1)
                mu_embedding = self.sample_laplace(mu_embedding, b_embedding)
                idx += 1
            else:
                mu_embedding, b_embedding, idx = self.embed_query_lape(queries, query_structure[0], idx)
            
            # Process each relation (or negation) in the current branch
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                # Negation: flip the sign of the mean and increase the scale
                # Here we add a constant (0.07) to represent increased uncertainty
                    mu_embedding = -mu_embedding
                    b_embedding = b_embedding + 0.07
                else:
                # For relation traversal, use the LapE relation embeddings:
                # Use self.mu_relation and self.b_relation instead of self.alpha_embedding and self.beta_embedding
                    mu_r_embedding = torch.index_select(self.mu_relation, dim=0, index=queries[:, idx])
                    b_r_embedding = torch.index_select(self.b_relation, dim=0, index=queries[:, idx])
                    # Apply projection operation
                    mu_embedding, b_embedding = self.projection_net(mu_embedding, b_embedding,
                                                                 mu_r_embedding, b_r_embedding)
                idx += 1

        else:
        # If not all relations, then we are dealing with a multi-branch (e.g., union or intersection) query.
            if self.is_u:
                mu_embedding_list = []
                b_embedding_list = []
                for i in range(len(query_structure)):
                    mu_emb, b_emb, idx = self.embed_query_lape(queries, query_structure[i], idx)
                    mu_embedding_list.append(mu_emb)
                    b_embedding_list.append(b_emb)
                mu_embedding, b_embedding = self.union_net(torch.stack(mu_embedding_list),
                                                        torch.stack(b_embedding_list))
            else:
                mu_embedding_list = []
                b_embedding_list = []
                for i in range(len(query_structure)):
                    mu_emb, b_emb, idx = self.embed_query_lape(queries, query_structure[i], idx)
                    mu_embedding_list.append(mu_emb)
                    b_embedding_list.append(b_emb)
                mu_embedding, b_embedding = self.center_net(torch.stack(mu_embedding_list),
                                                         torch.stack(b_embedding_list))

        return mu_embedding, b_embedding, idx

    def cal_logit_lape(self, entity_embedding, query_dist):
        """
        Compute the logit for a query based on Laplace-distributed embeddings.
        
        Args:
          entity_embedding: Tensor of shape (..., 2 * dim), where the first half is μ (mean)
                            and the second half is b (scale).
          query_dist: A tuple (query_mu, query_b) representing the Laplace distribution
                      of the query. distance computed using Wasserstein-1 distance between Laplace distributions.
        
        Returns:
          logit: The computed logit value, where higher values indicate higher similarity.
        """
        # Split the entity embedding into mean (mu) and scale (b)
        mu_embedding, b_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        # Unpack the query distribution (assumed to be in the same format: (mu, b))
        query_mu, query_b = query_dist
        query_mu = self.sample_laplace(query_mu, query_b)
        # Compute the Wasserstein-1 distance between the Laplace distributions:
        # Compute elementwise absolute differences for both μ and b, then sum over the embedding dimensions.
        distance = torch.abs(mu_embedding - query_mu) + torch.abs(b_embedding - query_b)
        distance = torch.sum(distance, dim=-1)
        
        # Compute the logit by subtracting the distance from the margin parameter gamma.
        logit = self.gamma - distance
    
        return logit

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_mu_embeddings, all_b_embeddings = [], [], []
        all_union_idxs, all_union_mu_embeddings, all_union_b_embeddings = [], [], []
    
        # Loop over each query structure in the batch.
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                self.is_u = True
                # For union queries, transform the query and then embed it.
                mu_embedding, b_embedding, _ = \
                    self.embed_query_lape(self.transform_union_query(batch_queries_dict[query_structure],
                                                                      query_structure),
                                           self.transform_union_structure(query_structure),
                                           0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_mu_embeddings.append(mu_embedding)
                all_union_b_embeddings.append(b_embedding)
            else:
                self.is_u = False
                mu_embedding, b_embedding, _ = self.embed_query_lape(batch_queries_dict[query_structure],
                                                                     query_structure,
                                                                     0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_mu_embeddings.append(mu_embedding)
                all_b_embeddings.append(b_embedding)
    
        # Form the Laplace distributions for non-union queries as (mu, b) tuples.
        if len(all_mu_embeddings) > 0:
            all_mu_embeddings = torch.cat(all_mu_embeddings, dim=0).unsqueeze(1)
            all_b_embeddings = torch.cat(all_b_embeddings, dim=0).unsqueeze(1)
            all_dists = (all_mu_embeddings, all_b_embeddings)
        # For union queries.
        if len(all_union_mu_embeddings) > 0:
            all_union_mu_embeddings = torch.cat(all_union_mu_embeddings, dim=0).unsqueeze(1)
            all_union_b_embeddings = torch.cat(all_union_b_embeddings, dim=0).unsqueeze(1)
            all_union_dists = (all_union_mu_embeddings, all_union_b_embeddings)
            
        if subsampling_weight is not None:
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]
    
        # Process positive samples.
        if positive_sample is not None:
            if len(all_mu_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_lape(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
    
            if len(all_union_mu_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1))
                positive_union_logit = self.cal_logit_lape(positive_embedding, all_union_dists)
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None
    
        # Process negative samples.
        if negative_sample is not None:
            if len(all_mu_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1))
                    .view(batch_size, negative_size, -1)
                )
                negative_logit = self.cal_logit_lape(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
    
            if len(all_union_mu_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1))
                    .view(batch_size, negative_size, -1)
                )
                negative_union_logit = self.cal_logit_lape(negative_embedding, all_union_dists)
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None
    
        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs
    def transform_union_query(self, queries, query_structure):
        # For union queries, the transformation remains the same
        # regardless of whether we use Gamma or Laplace embeddings.
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([queries[:, :4], queries[:, 5:6]], dim=1)
        return queries
    
    def transform_union_structure(self, query_structure):
        # The union structure mapping is identical for LapE.
        if self.query_name_dict[query_structure] == '2u-DNF':
            return (('e', ('r',)), ('e', ('r',)))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ((('e', ('r',)), ('e', ('r',))), ('r',))

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
    
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
    
        # Call the LapE model's forward function, which returns positive and negative logits computed via Laplace embeddings.
        positive_logit, negative_logit, subsampling_weight, _ = model(
            positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict
        )
    
        # Compute the ranking loss using logsigmoid. Note that a higher logit means a better match.
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
    
        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log
    
    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()
        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)
    
        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(
                    test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()
    
                # Call our LapE model's forward function (which now returns logits computed using Laplace embeddings)
                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:  # reuse batch_entity_range if possible
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range)
                else:
                    if args.cuda:
                        ranking = ranking.scatter_(1, argsort,
                                                    torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).cuda())
                    else:
                        ranking = ranking.scatter_(1, argsort,
                                                    torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1))
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers
                        
                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
    
                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })
    
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                step += 1
    
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])
    
        return metrics
    
    