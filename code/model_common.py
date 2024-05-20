import torch
import numpy                as np
import torch.nn             as nn
import torch.nn.functional  as F

from utils                  import *
from dataloader             import *
from tqdm                   import tqdm
from torch.utils.data       import DataLoader


class KGFIT(nn.Module):
    def __init__(self, base_model, nentity, nrelation, hidden_dim, gamma, 
                    double_entity_embedding=False, double_relation_embedding=False, triple_relation_embedding=False,
                    entity_text_embeddings=None, cluster_embeddings=None, 
                    rho=0.4, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5, 
                    zeta_1=0.2, zeta_2=0.3, zeta_3=0.5, distance_metric='cosine', 
                    inter_cluster_constraint="true", intra_cluster_constraint="true", hier_dist_constraint="true", text_dist_constraint="true",
                    hake_p=0.5, hake_m=0.5, kwargs={},
                    ):
        
        super(KGFIT, self).__init__()
        self.model_name = base_model
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.distance_metric = distance_metric
        self.inter_cluster = inter_cluster_constraint
        self.intra_cluster = intra_cluster_constraint
        self.hier_constraint = hier_dist_constraint
        self.text_dist_constraint = text_dist_constraint
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        if double_relation_embedding:
            self.relation_dim = hidden_dim*2
        elif triple_relation_embedding:
            self.relation_dim = hidden_dim*3
        else:
            self.relation_dim = hidden_dim
        
        # Initialize relation embeddings (Equation 7)
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        print(f"Size of random relation_embedding: {self.relation_embedding.size()}")
        
        # Initialize randomly initialized component of entity embeddings
        self.entity_embedding_init = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding_init, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        print(f"Size of random entity_embedding_init: {self.entity_embedding_init.size()}")
        
        ent_text_emb, ent_desc_emb      = torch.chunk(entity_text_embeddings, 2, dim=1)
        clus_text_emb, clus_desc_emb    = torch.chunk(cluster_embeddings, 2, dim=1)
        
        # concatenate ent_text_emb[:self.entity_dim/2] and ent_desc_emb[:self.entity_dim/2], size: (nentity, self.entity_dim)
        self.entity_text_embeddings = torch.cat([ent_text_emb[:, :self.entity_dim//2], ent_desc_emb[:, :self.entity_dim//2]], dim=1)
        self.entity_text_embeddings.requires_grad = False
        print(f"Size of entity_text_embeddings: {self.entity_text_embeddings.size()}")
        # concatenate clus_text_emb[:self.entity_dim/2] and clus_desc_emb[:self.entity_dim/2], size: (nentity, self.entity_dim)
        self.cluster_embeddings     = torch.cat([clus_text_emb[:, :self.entity_dim//2], clus_desc_emb[:, :self.entity_dim//2]], dim=1)
        self.cluster_embeddings.requires_grad = False
        print(f"Size of cluster_embeddings: {self.cluster_embeddings.size()}")
        
        if base_model == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        if base_model not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'HAKE']:
            raise ValueError('model %s not supported' % base_model)
            
        if base_model == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if base_model == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        # Hyperparameters
        self.rho = rho              # Hyperparameter controlling the influence of the randomly initialized component in the embedding
        
        self.lambda_1 = lambda_1    # Hyperparameter controlling the influence of the inter-level cluster separation
        self.lambda_2 = lambda_2    # Hyperparameter controlling the influence of the hierarchical distance maintenance
        self.lambda_3 = lambda_3    # Hyperparameter controlling the influence of the cluster cohesion
        
        self.zeta_1 = zeta_1        # Hyperparameter controlling the influence of the entire hierarchical constraint
        self.zeta_2 = zeta_2        # Hyperparameter controlling the influence of the text embedding deviation
        self.zeta_3 = zeta_3        # Hyperparameter controlling the influence of the link prediction score
        
        # model specific parameters
        # HAKE
        self.phase_weight = nn.Parameter(torch.Tensor([[hake_p * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[hake_m]]))

    @staticmethod
    def get_masked_embeddings(indices, embeddings, dim_size):
        """
        Retrieves and applies a mask to embeddings based on provided indices.
        
        Args:
            indices (torch.Tensor): Tensor of indices with possible -1 indicating invalid entries.
            embeddings (torch.nn.Parameter): Embeddings from which to select.
            dim_size (tuple): The desired dimension sizes of the output tensor.
        
        Returns:
            torch.Tensor: Masked and selected embeddings based on valid indices.
        """
        valid_mask = indices != -1
        # Initialize tensor to hold the masked embeddings
        masked_embeddings = torch.zeros(*dim_size, dtype=embeddings.dtype, device=embeddings.device)
        # Apply mask to filter valid indices
        valid_indices = indices[valid_mask]
        selected_embeddings = torch.index_select(embeddings, dim=0, index=valid_indices)
        # Place selected embeddings back into the appropriate locations
        masked_embeddings.view(-1, embeddings.shape[1])[valid_mask.view(-1)] = selected_embeddings
        return masked_embeddings
    
    def get_entity_embedding(self):
        """
        Retrieve the embedding for the given entity ID.
        """
        # size: (nentity, self.entity_dim)
        return self.rho * self.entity_embedding_init + (1 - self.rho) * self.entity_text_embeddings
    

    def forward(self, sample, self_cluster_ids, neighbor_clusters_ids, parent_ids, mode='single'):
        if mode == 'single':
            # positive relation embeddings,     size: (batch_size, 1, hidden_dim)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            # positive head embeddings,         size: (batch_size, 1, hidden_dim)
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=sample[:, 0]).unsqueeze(1)
            # positive tail embeddings,         size: (batch_size, 1, hidden_dim)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=sample[:, 2]).unsqueeze(1)
            # positive head text embeddings,    size: (batch_size, 1, hidden_dim)
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 0]).unsqueeze(1)
            # positive tail text embeddings,    size: (batch_size, 1, hidden_dim)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 2]).unsqueeze(1)
            
            
            # Combine entity embeddings with text embeddings and randomly initialized component, size: (batch_size, 1, hidden_dim)
            head_combined           =   self.rho * head_init + (1 - self.rho) * head_text
            tail_combined           =   self.rho * tail_init + (1 - self.rho) * tail_text
            
            # KGE Score (positive),               (lower -> better),     size: (batch_size, 1)
            link_pred_score         =   self.score_func(head_combined, relation, tail_combined, mode)
            
            return link_pred_score
            
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            # assert torch.all(head_part < self.nentity), "head_part contains out-of-bounds indices"
            # assert torch.all(tail_part < self.nentity), "tail_part contains out-of-bounds indices"
            # assert torch.all(neighbor_clusters_ids < len(self.cluster_embeddings)), "neighbor_clusters_ids contains out-of-bounds indices"
            # assert torch.all(parent_ids < len(self.cluster_embeddings)), "parent_ids contains out-of-bounds indices"
            
            # positive relation embeddings,     size: (batch_size, 1, hidden_dim)
            relation  = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            # positive tail embeddings,         size: (batch_size, 1, hidden_dim)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            # negative head embeddings,         size: (batch_size, negative_sample_size, hidden_dim)
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            # positive tail text embeddings,    size: (batch_size, 1, hidden_dim)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            # negative head text embeddings,    size: (batch_size, negative_sample_size, hidden_dim)
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            if self.intra_cluster == "true":
                # positive tail cluster embeddings, size: (batch_size, 1, hidden_dim)
                cluster_emb = torch.index_select(self.cluster_embeddings, dim=0, index=self_cluster_ids).unsqueeze(1)
            if self.inter_cluster == "true":
                # positive other cluster embeddings, size: (batch_size, max_num_neighbor_clusters, hidden_dim)
                neighbor_cluster_emb = self.get_masked_embeddings(
                    neighbor_clusters_ids, self.cluster_embeddings,
                    (neighbor_clusters_ids.size(0), neighbor_clusters_ids.size(1), self.entity_dim)
                )
            if self.hier_constraint == "true":
                # positive parent embeddings, size: (batch_size, max_parent_num, hidden_dim)
                parent_emb = self.get_masked_embeddings(
                    parent_ids, self.cluster_embeddings,
                    (parent_ids.size(0), parent_ids.size(1), self.entity_dim)
                )
            
            # positive tail embeddings,         size: (batch_size, 1, hidden_dim)
            tail_combined           =   self.rho * tail_init + (1 - self.rho) * tail_text
            # # negative head embeddings,         size: (batch_size, negative_sample_size, hidden_dim)
            head_combined           =   self.rho * head_init + (1 - self.rho) * head_text
            
            if self.text_dist_constraint == "true":
                # Text Embedding Deviation,         (lower -> better),      size: (batch_size, 1)
                text_dist               =   self.distance(tail_combined, tail_text  )
            else:
                text_dist = torch.zeros((batch_size, 1), dtype=tail_combined.dtype, device=tail_combined.device)

            if self.intra_cluster == "true":
                # Cluster Cohesion,                 (lower -> better),      size: (batch_size, 1)
                self_cluster_dist       =   self.distance(tail_combined, cluster_emb)
            else:
                self_cluster_dist = torch.zeros((batch_size, 1), dtype=tail_combined.dtype, device=tail_combined.device)
            
            if self.inter_cluster == "true":
                # Inter-level Cluster Separation,   (higher -> better),     size: (batch_size, neibor_cluster_size)
                neighbor_cluster_dist   =   self.distance(tail_combined, neighbor_cluster_emb)
            else:
                neighbor_cluster_dist = torch.zeros((batch_size, 1), dtype=tail_combined.dtype, device=tail_combined.device)
                
            if self.hier_constraint == "true": 
                #Hierarchical Distance Maintenance, (higher -> better),     size: (batch_size, max_parent_num)
                hier_dist = 0
                for i in range(len(parent_emb)-1):
                    parent_embedding, parent_parent_embedding = parent_emb[i], parent_emb[i+1]
                    hier_dist           +=  self.distance(tail_combined, parent_parent_embedding) - self.distance(tail_combined, parent_embedding)
            else:
                hier_dist = torch.zeros((batch_size, 1), dtype=tail_combined.dtype, device=tail_combined.device)
                    
            # KGE Score (negative heads),       (lower -> better),      size: (batch_size, negative_sample_size)
            link_pred_score         =   self.score_func(head_combined, relation, tail_combined, mode)
            
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # assert torch.all(head_part < self.nentity), "head_part contains out-of-bounds indices"
            # assert torch.all(tail_part < self.nentity), "tail_part contains out-of-bounds indices"
            # assert torch.all(neighbor_clusters_ids < len(self.cluster_embeddings)), "neighbor_clusters_ids contains out-of-bounds indices"
            # assert torch.all(parent_ids < len(self.cluster_embeddings)), "parent_ids contains out-of-bounds indices"
            
            # positive relation embeddings,     size: (batch_size, 1, hidden_dim)
            relation  = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            # positive head embeddings,         size: (batch_size, 1, hidden_dim)
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=head_part[:, 0]).unsqueeze(1)
            # negative tail embeddings,         size: (batch_size, negative_sample_size, hidden_dim)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            # positive head text embeddings,    size: (batch_size, 1, hidden_dim)
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part[:, 0]).unsqueeze(1)
            # negative tail text embeddings,    size: (batch_size, negative_sample_size, hidden_dim)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            if self.intra_cluster == "true":
                # positive head cluster embeddings, size: (batch_size, 1, hidden_dim)
                cluster_emb = torch.index_select(self.cluster_embeddings, dim=0, index=self_cluster_ids).unsqueeze(1)
            
            if self.inter_cluster == "true":
                # positive other cluster embeddings, size: (batch_size, max_num_neighbor_clusters, hidden_dim)
                neighbor_cluster_emb = self.get_masked_embeddings(
                    neighbor_clusters_ids, self.cluster_embeddings,
                    (neighbor_clusters_ids.size(0), neighbor_clusters_ids.size(1), self.entity_dim)
                )
            if self.hier_constraint == "true":
                # positive parent embeddings, size: (batch_size, max_parent_num, hidden_dim)
                parent_emb = self.get_masked_embeddings(
                    parent_ids, self.cluster_embeddings,
                    (parent_ids.size(0), parent_ids.size(1), self.entity_dim)
                )
                
            # positive head embeddings,        size: (batch_size, 1, hidden_dim)
            head_combined = self.rho * head_init + (1 - self.rho) * head_text 
            # negative tail embeddings,       size: (batch_size, negative_sample_size, hidden_dim)
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text 
            
            if self.text_dist_constraint == "true":
                # Text Embedding Deviation,         (lower -> better),      size: (batch_size, 1)
                text_dist               =   self.distance(head_combined, head_text  )
            else:
                text_dist = torch.zeros((batch_size, 1), dtype=head_combined.dtype, device=head_combined.device)
            
            if self.intra_cluster == "true":
                # Cluster Cohesion,                 (lower -> better),      size: (batch_size, 1)
                self_cluster_dist       =   self.distance(head_combined, cluster_emb)
            else:
                self_cluster_dist = torch.zeros((batch_size, 1), dtype=head_combined.dtype, device=head_combined.device)
            
            if self.inter_cluster == "true":
                # Inter-level Cluster Separation,   (higher -> better),     size: (batch_size, neibor_cluster_size)
                neighbor_cluster_dist   =   self.distance(head_combined, neighbor_cluster_emb)
            else:
                neighbor_cluster_dist = torch.zeros((batch_size, 1), dtype=head_combined.dtype, device=head_combined.device)
            
            if self.hier_constraint == "true":
                #Hierarchical Distance Maintenance, (higher -> better),     size: (batch_size, max_parent_num)
                hier_dist = 0
                for i in range(len(parent_emb)-1):
                    parent_embedding, parent_parent_embedding = parent_emb[i], parent_emb[i+1]
                    hier_dist           +=  self.distance(head_combined, parent_parent_embedding) - self.distance(head_combined, parent_embedding)
            else:
                hier_dist = torch.zeros((batch_size, 1), dtype=head_combined.dtype, device=head_combined.device)
            
            # KGE Score (negative tails),       (lower -> better),      size: (batch_size, negative_sample_size)
            link_pred_score         =   self.score_func(head_combined, relation, tail_combined, mode)
            
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        
        return text_dist, self_cluster_dist, neighbor_cluster_dist, hier_dist, link_pred_score 

    def rotate_distance(self, embeddings1, embeddings2):
        pi = 3.14159262358979323846
        
        phase1, mod1 = torch.chunk(embeddings1, 2, dim=-1)
        phase2, mod2 = torch.chunk(embeddings2, 2, dim=-1)
        
        phase1 = phase1 / (self.embedding_range.item() / pi)
        phase2 = phase2 / (self.embedding_range.item() / pi)
        
        phase_diff = torch.abs(torch.sin((phase1 - phase2) / 2))
        # return torch.mean(phase_diff, dim=-1)
        return torch.sum(phase_diff, dim=-1)

    def distance(self, embeddings1, embeddings2):
        """
        Compute the distance between two sets of embeddings.
        """
        if self.distance_metric == 'euclidean':
            return torch.norm(embeddings1 - embeddings2, p=2, dim=-1)
        elif self.distance_metric == 'manhattan':
            return torch.norm(embeddings1 - embeddings2, p=1, dim=-1)
        elif self.distance_metric == 'cosine':
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
            cosine_similarity = torch.sum(embeddings1_norm * embeddings2_norm, dim=-1)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance
        elif self.distance_metric == 'rotate':
            return self.rotate_distance(embeddings1, embeddings2)
        elif self.distance_metric == 'pi':
            pi = 3.14159262358979323846
            phase1 = embeddings1 / (self.embedding_range.item() / pi)
            phase2 = embeddings2 / (self.embedding_range.item() / pi)
            distance = torch.abs(torch.sin((phase1 - phase2) / 2))
            return 1 - torch.mean(distance, dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def score_func(self, head, relation, tail, mode='single'):
        """
        Compute the score for the given triple (head, relation, tail).
        """
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'HAKE': self.HAKE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def TransE(self, head, relation, tail, mode):
        """
        Compute the score using the TransE model.
        """
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        """
        Compute the score using the DistMult model.
        """
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        """
        Compute the score using the ComplEx model.
        """
        head_re, head_im = torch.chunk(head, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            score = head_re * re_score + head_im * im_score
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            score = re_score * tail_re + im_score * tail_im

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        """
        Compute the score using the RotatE model.
        """
        pi = 3.14159265358979323846
        
        head_re, head_im = torch.chunk(head, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        relation_re = torch.cos(phase_relation)
        relation_im = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            re_score = re_score - head_re
            im_score = im_score - head_im
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            re_score = re_score - tail_re
            im_score = im_score - tail_im

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        """
        Compute the score using the pRotatE model.
        """
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    def HAKE(self, head, rel, tail, mode):
        """
        Compute the score using the HAKE model.
        """
        pi = 3.14159262358979323846
        
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
    
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, cluster_id_head, cluster_id_tail, \
            neighbor_clusters_ids_head, neighbor_clusters_ids_tail, parent_ids_head, parent_ids_tail, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            cluster_id_head = cluster_id_head.cuda()
            cluster_id_tail = cluster_id_tail.cuda()
            neighbor_clusters_ids_head = neighbor_clusters_ids_head.cuda()
            neighbor_clusters_ids_tail = neighbor_clusters_ids_tail.cuda()
            parent_ids_head = parent_ids_head.cuda()
            parent_ids_tail = parent_ids_tail.cuda()
        
        ## Negative Samples
        if mode == 'head-batch':
            self_cluster_ids = cluster_id_tail
            neighbor_clusters_ids = neighbor_clusters_ids_tail
            parent_ids = parent_ids_tail
            
        elif mode == 'tail-batch':
            self_cluster_ids = cluster_id_head
            neighbor_clusters_ids = neighbor_clusters_ids_head
            parent_ids = parent_ids_head
            

        text_dist_n, self_cluster_dist_n, neighbor_cluster_dist_n, hier_dist_n, negative_score = \
            model((positive_sample, negative_sample), self_cluster_ids, neighbor_clusters_ids, parent_ids, mode=mode)
            
        neighbor_cluster_dist_mean_n = neighbor_cluster_dist_n.mean(dim=1, keepdim=True)
        hier_dist_mean_n = hier_dist_n.mean(dim=1, keepdim=True)


        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)


        ## Positive Sample
        self_cluster_ids = (cluster_id_head, cluster_id_tail)
        neighbor_clusters_ids = (neighbor_clusters_ids_head, neighbor_clusters_ids_tail)
        parent_ids = (parent_ids_head, parent_ids_tail)
        
        positive_score = \
            model(positive_sample, self_cluster_ids, neighbor_clusters_ids, parent_ids, mode='single')
            

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        ## Loss function
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        entity_embedding = model.get_entity_embedding()
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss = model.zeta_3 * loss \
            + model.zeta_1 * (model.lambda_1 * (self_cluster_dist_n) \
                                - model.lambda_2 * (neighbor_cluster_dist_mean_n) \
                                - model.lambda_3 * (hier_dist_mean_n)) \
            + model.zeta_2 * (text_dist_n)
        
        loss = loss.mean()        
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        
        loss_details = {
            'text_dist_n': text_dist_n.mean().item(),
            'self_cluster_dist_n': self_cluster_dist_n.mean().item(),
            'neighbor_cluster_dist_n': neighbor_cluster_dist_n.mean().item(),
            'hier_dist_n': hier_dist_n.mean().item(),
        }
        
        log.update(loss_details)

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, entity_info, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
    
        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                entity_info,
                'head-batch',
                rerank=True if args.rerank == "true" else False
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                entity_info,
                'tail-batch',
                rerank=True if args.rerank == "true" else False
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, cluster_id_head, cluster_id_tail, \
                    neighbor_clusters_ids_head, neighbor_clusters_ids_tail, parent_ids_head, \
                        parent_ids_tail, mode in tqdm(test_dataset):
                            
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                        cluster_id_head = cluster_id_head.cuda()
                        cluster_id_tail = cluster_id_tail.cuda()
                        neighbor_clusters_ids_head = neighbor_clusters_ids_head.cuda()
                        neighbor_clusters_ids_tail = neighbor_clusters_ids_tail.cuda()
                        parent_ids_head = parent_ids_head.cuda()
                        parent_ids_tail = parent_ids_tail.cuda()

                    batch_size = positive_sample.size(0)
                    
                    ## Negative Samples
                    if mode == 'head-batch':
                        self_cluster_ids = cluster_id_tail
                        neighbor_clusters_ids = neighbor_clusters_ids_tail
                        parent_ids = parent_ids_tail
                        
                    elif mode == 'tail-batch':
                        self_cluster_ids = cluster_id_head
                        neighbor_clusters_ids = neighbor_clusters_ids_head
                        parent_ids = parent_ids_head

                    text_dist, self_cluster_dist, neighbor_cluster_dist, hier_dist, link_pred_score = model((positive_sample, negative_sample), self_cluster_ids, neighbor_clusters_ids, parent_ids, mode)
                   
                    if args.fuse_score == "true":
                        # Compute the cosine similarity
                        if mode == 'head-batch':
                            # Compute cosine similarity between (tail - relation) and all entities
                            tail = torch.index_select(model.get_entity_embedding(), dim=0, index=positive_sample[:, 2])
                            relation = torch.index_select(model.relation_embedding, dim=0, index=positive_sample[:, 1])
                            tail_relation = tail[:, None, :] - relation[:, None, :]  # shape: (batch_size, 1, embedding_dim)
                            cosine_sim = F.cosine_similarity(tail_relation, model.get_entity_embedding()[None, :, :], dim=-1)  # shape: (batch_size, num_entities)
                        elif mode == 'tail-batch':
                            # Compute cosine similarity between (head + relation) and all entities
                            head = torch.index_select(model.get_entity_embedding(), dim=0, index=positive_sample[:, 0])
                            relation = torch.index_select(model.relation_embedding, dim=0, index=positive_sample[:, 1])
                            head_relation = head[:, None, :] + relation[:, None, :]  # shape: (batch_size, 1, embedding_dim)
                            cosine_sim = F.cosine_similarity(head_relation, model.get_entity_embedding()[None, :, :], dim=-1)  # shape: (batch_size, num_entities)
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        
                        link_pred_score += filter_bias
                        cosine_sim += filter_bias

                        # Normalize the score and cosine similarity tensors
                        normalized_score = F.normalize(link_pred_score, p=2, dim=-1)
                        normalized_cosine_sim = F.normalize(cosine_sim, p=2, dim=-1)

                        # Compute ranks for normalized scores
                        link_pred_rank = torch.argsort(normalized_score, dim=1, descending=True).argsort(dim=1)
                        cosine_sim_rank = torch.argsort(normalized_cosine_sim, dim=1, descending=True).argsort(dim=1)

                        # Convert ranks to reciprocal ranks
                        link_pred_reciprocal_rank = 1.0 / (link_pred_rank.float() + 1)
                        cosine_sim_reciprocal_rank = 1.0 / (cosine_sim_rank.float() + 1)

                        # Combine the reciprocal ranks
                        combined_reciprocal_rank = 9.0 * link_pred_reciprocal_rank + 1.0 * cosine_sim_reciprocal_rank

                        # Sort based on the combined reciprocal rank
                        argsort = torch.argsort(combined_reciprocal_rank, dim=1, descending=True)
                        
                    else:
                        score = link_pred_score + filter_bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@5': 1.0 if ranking <= 5 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics