import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch.nn import functional as F, Parameter
import torch.nn             as nn

class KGFIT_TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, nentity, nrelation,
                 entity_text_embeddings=None, cluster_embeddings=None,
                 rho=0.4, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5, 
                 zeta_1=0.3, zeta_2=0.2, zeta_3=0.5, distance_metric='cosine',**kwargs):
        super(KGFIT_TuckER, self).__init__()
        self.epsilon = 2.0
        
        self.rho = rho
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.zeta_1 = zeta_1
        self.zeta_2 = zeta_2
        self.zeta_3 = zeta_3
        
        self.distance_metric = distance_metric
        
        self.entity_dim = d1
        
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
        
        
        self.E = torch.nn.Embedding(nentity, d1, padding_idx=0)
        self.R = torch.nn.Embedding(nrelation, d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        
        self.E.weight.data = self.rho * self.E.weight.data + (1 - self.rho) * self.entity_text_embeddings
        
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
    
    def distance(self, embeddings1, embeddings2):
        """
        Compute the distance between two sets of embeddings.
        """
        if embeddings1.size() != embeddings2.size():
            embeddings1 = embeddings1.unsqueeze(1)
            
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
            return self.rotate_distance(embeddings1, embeddings2, embeddings2)
        elif self.distance_metric == 'pi':
            pi = 3.14159262358979323846
            phase1 = embeddings1 / (self.embedding_range.item() / pi)
            phase2 = embeddings2 / (self.embedding_range.item() / pi)
            distance = torch.abs(torch.sin((phase1 - phase2) / 2))
            return 1 - torch.mean(distance, dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def forward(self, e1_idx, r_idx, self_cluster_ids, neighbor_clusters_ids, parent_ids):
        e1_text = torch.index_select(self.entity_text_embeddings, 0, e1_idx.view(-1)).unsqueeze(1)
        cluster_emb = torch.index_select(self.cluster_embeddings, dim=0, index=self_cluster_ids).unsqueeze(1)
        # positive other cluster embeddings, size: (batch_size, max_num_neighbor_clusters, hidden_dim)
        neighbor_cluster_emb = self.get_masked_embeddings(
            neighbor_clusters_ids, self.cluster_embeddings,
            (neighbor_clusters_ids.size(0), neighbor_clusters_ids.size(1), self.entity_dim)
        )
        # positive parent embeddings, size: (batch_size, max_parent_num, hidden_dim)
        parent_emb = self.get_masked_embeddings(
            parent_ids, self.cluster_embeddings,
            (parent_ids.size(0), parent_ids.size(1), self.entity_dim)
        )
        
        e1 = self.E(e1_idx)
        
        text_dist = self.distance(e1, e1_text)
        self_cluster_dist = self.distance(e1, cluster_emb)
        neighbor_cluster_dist = self.distance(e1, neighbor_cluster_emb)
        hier_dist = 0
        for i in range(len(parent_emb)-1):
            parent_embedding, parent_parent_embedding = parent_emb[i], parent_emb[i+1]
            hier_dist           +=  self.distance(e1, parent_parent_embedding) - self.distance(e1, parent_embedding)
        
        
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred, text_dist, self_cluster_dist, neighbor_cluster_dist, hier_dist
    

class KGFIT_ConvE(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(KGFIT_ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 16
        self.emb_dim2 = d1 // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        # self.fc = torch.nn.Linear(5760, int(d1))
        if int(d1) == 128:
            self.fc = torch.nn.Linear(5760, d1)
        elif int(d1) == 256:
            self.fc = torch.nn.Linear(13440, d1)
        elif int(d1) == 512:
            self.fc = torch.nn.Linear(28800, d1)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1_idx, r_idx):
        # attention
        e1_embedded = self.emb_e(e1_idx)

        e1_embedded= e1_embedded.view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)

        # e1_embedded= e1_embedded.view(-1, 1, 10, 20)
        # rel_embedded = self.emb_rel(r_idx).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred