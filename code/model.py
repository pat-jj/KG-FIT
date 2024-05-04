import torch
import numpy                as np
import torch.nn             as nn
import torch.nn.functional  as F

from utils                  import *
from dataloader             import *
from tqdm                   import tqdm
from torch.utils.data       import DataLoader
from sklearn.metrics        import average_precision_score


class KGFIT(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False,
                 entity_hierarchy=None, entity_text_embeddings=None, rho=0.5, alpha=0.5, beta=0.5, gamma_1=1.0, gamma_2=1.0, lambda_1=1.0, lambda_2=1.0):
        super(KGFIT, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        # Initialize relation embeddings (Equation 7)
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # Initialize randomly initialized component of entity embeddings
        self.entity_embedding_init = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding_init, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        self.entity_hierarchy = entity_hierarchy
        self.entity_text_embeddings = entity_text_embeddings
        self.rho = rho              # Hyperparameter controlling the influence of the randomly initialized component in the embedding
        self.alpha = alpha          # Weight for the hierarchical constraint
        self.beta = beta            # Weight for the text embedding deviation constraint
        self.gamma_1 = gamma_1      # Weight for the parent-child constraint
        self.gamma_2 = gamma_2      # Weight for the link prediction constraint
        self.lambda_1 = lambda_1    # Weight for the inter-cluster separation
        self.lambda_2 = lambda_2    # Weight for the parent-child relationship preservation
        
    def forward(self, sample, mode='single'):
        if mode == 'single':
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=sample[:, 0]).unsqueeze(1)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=sample[:, 2]).unsqueeze(1)
            
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 0]).unsqueeze(1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 2]).unsqueeze(1)
            
            # Combine entity embeddings with text embeddings and randomly initialized component 
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            
            head_cluster = self.get_cluster_embedding(sample[:, 0], head_combined)
            tail_cluster = self.get_cluster_embedding(sample[:, 2], tail_combined)
            
            head_parent_cluster = self.get_parent_cluster_embedding(sample[:, 0], head_combined)
            tail_parent_cluster = self.get_parent_cluster_embedding(sample[:, 2], tail_combined)
            
            head_text_dist = self.distance(head_combined, head_text)
            tail_text_dist = self.distance(tail_combined, tail_text)
            
            # Compute intra-cluster distances
            intra_cluster_dists = []
            for cluster_embedding in torch.cat([head_cluster, tail_cluster], dim=0):
                cluster_entities = self.entity_hierarchy[cluster_embedding]
                cluster_entity_embeddings = torch.index_select(torch.cat([head_combined, tail_combined], dim=0), dim=0, index=torch.tensor(list(cluster_entities)))
                intra_cluster_dist = torch.mean(self.distance(cluster_entity_embeddings, cluster_embedding))
                intra_cluster_dists.append(intra_cluster_dist)
            intra_cluster_loss = torch.mean(torch.stack(intra_cluster_dists))
            
            # Compute inter-cluster distances
            inter_cluster_dists = []
            for i, cluster_embedding in enumerate(torch.cat([head_cluster, tail_cluster], dim=0)):
                other_cluster_embeddings = torch.cat([head_cluster[:i], head_cluster[i+1:], tail_cluster[:i], tail_cluster[i+1:]], dim=0)
                inter_cluster_dist = torch.min(self.distance(cluster_embedding, other_cluster_embeddings))
                inter_cluster_dists.append(inter_cluster_dist)
            inter_cluster_loss = torch.mean(torch.stack(inter_cluster_dists))
            
            # Compute parent-child distances
            parent_child_dists = []
            for cluster_embedding, parent_cluster_embedding in zip(torch.cat([head_cluster, tail_cluster], dim=0), torch.cat([head_parent_cluster, tail_parent_cluster], dim=0)):
                parent_child_dist = self.distance(cluster_embedding, parent_cluster_embedding)
                parent_child_dists.append(parent_child_dist)
            parent_child_loss = torch.mean(torch.stack(parent_child_dists))
            
            hierarchical_loss = intra_cluster_loss - self.lambda_1 * inter_cluster_loss + self.lambda_2 * parent_child_loss
            
            # Apply constraints 
            score = - self.alpha * hierarchical_loss  # Hierarchical constraint 
            score = score - self.beta * (head_text_dist + tail_text_dist)  # Text embedding deviation constraint 
            
            # Link prediction constraint (refined)
            true_triple_score = self.score_func(head_combined, relation, tail_combined)
            negative_tails = self.sample_negative_entities(sample[:, 2])
            negative_tail_scores = self.score_func(head_combined, relation, negative_tails)
            
            score = score - (self.gamma_2 * (true_triple_score - torch.mean(negative_tail_scores, dim=1)))
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            
            head_cluster = self.get_cluster_embedding(head_part.view(-1), head_combined).view(batch_size, negative_sample_size, -1)
            tail_cluster = self.get_cluster_embedding(tail_part[:, 2], tail_combined)
            
            head_parent_cluster = self.get_parent_cluster_embedding(head_part.view(-1), head_combined).view(batch_size, negative_sample_size, -1)
            tail_parent_cluster = self.get_parent_cluster_embedding(tail_part[:, 2], tail_combined)
            
            head_text_dist = self.distance(head_combined, head_text)
            tail_text_dist = self.distance(tail_combined, tail_text)
            
            # Compute intra-cluster distances
            intra_cluster_dists = []
            for cluster_embedding in torch.cat([head_cluster.view(-1, head_cluster.size(-1)), tail_cluster], dim=0):
                cluster_entities = self.entity_hierarchy[cluster_embedding]
                cluster_entity_embeddings = torch.index_select(torch.cat([head_combined.view(-1, head_combined.size(-1)), tail_combined], dim=0), dim=0, index=torch.tensor(list(cluster_entities)))
                intra_cluster_dist = torch.mean(self.distance(cluster_entity_embeddings, cluster_embedding))
                intra_cluster_dists.append(intra_cluster_dist)
            intra_cluster_loss = torch.mean(torch.stack(intra_cluster_dists))
            
            # Compute inter-cluster distances
            inter_cluster_dists = []
            for i, cluster_embedding in enumerate(torch.cat([head_cluster.view(-1, head_cluster.size(-1)), tail_cluster], dim=0)):
                other_cluster_embeddings = torch.cat([head_cluster.view(-1, head_cluster.size(-1))[:i], head_cluster.view(-1, head_cluster.size(-1))[i+1:], tail_cluster[:i], tail_cluster[i+1:]], dim=0)
                inter_cluster_dist = torch.min(self.distance(cluster_embedding, other_cluster_embeddings))
                inter_cluster_dists.append(inter_cluster_dist)
            inter_cluster_loss = torch.mean(torch.stack(inter_cluster_dists))
            
            # Compute parent-child distances
            parent_child_dists = []
            for cluster_embedding, parent_cluster_embedding in zip(torch.cat([head_cluster.view(-1, head_cluster.size(-1)), tail_cluster], dim=0), torch.cat([head_parent_cluster.view(-1, head_parent_cluster.size(-1)), tail_parent_cluster], dim=0)):
                parent_child_dist = self.distance(cluster_embedding, parent_cluster_embedding)
                parent_child_dists.append(parent_child_dist)
            parent_child_loss = torch.mean(torch.stack(parent_child_dists))
            
            hierarchical_loss = intra_cluster_loss - self.lambda_1 * inter_cluster_loss + self.lambda_2 * parent_child_loss
            
            score = - self.alpha * hierarchical_loss
            score = score - self.beta * (head_text_dist + tail_text_dist)
            
            true_triple_score = self.score_func(head_combined, relation, tail_combined)
            negative_heads = self.sample_negative_entities(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            negative_head_scores = self.score_func(negative_heads, relation, tail_combined)
            
            score = score - (self.gamma_2 * (true_triple_score - torch.mean(negative_head_scores, dim=1)))
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=head_part[:, 0]).unsqueeze(1)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part[:, 0]).unsqueeze(1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            
            head_cluster = self.get_cluster_embedding(head_part[:, 0], head_combined)
            tail_cluster = self.get_cluster_embedding(tail_part.view(-1), tail_combined).view(batch_size, negative_sample_size, -1)
            
            head_parent_cluster = self.get_parent_cluster_embedding(head_part[:, 0], head_combined)
            tail_parent_cluster = self.get_parent_cluster_embedding(tail_part.view(-1), tail_combined).view(batch_size, negative_sample_size, -1)
            
            head_text_dist = self.distance(head_combined, head_text)
            tail_text_dist = self.distance(tail_combined, tail_text)
            
            # Compute intra-cluster distances
            intra_cluster_dists = []
            for cluster_embedding in torch.cat([head_cluster, tail_cluster.view(-1, tail_cluster.size(-1))], dim=0):
                cluster_entities = self.entity_hierarchy[cluster_embedding]
                cluster_entity_embeddings = torch.index_select(torch.cat([head_combined, tail_combined.view(-1, tail_combined.size(-1))], dim=0), dim=0, index=torch.tensor(list(cluster_entities)))
                intra_cluster_dist = torch.mean(self.distance(cluster_entity_embeddings, cluster_embedding))
                intra_cluster_dists.append(intra_cluster_dist)
            intra_cluster_loss = torch.mean(torch.stack(intra_cluster_dists))
            
            # Compute inter-cluster distances
            inter_cluster_dists = []
            for i, cluster_embedding in enumerate(torch.cat([head_cluster, tail_cluster.view(-1, tail_cluster.size(-1))], dim=0)):
                other_cluster_embeddings = torch.cat([head_cluster[:i], head_cluster[i+1:], tail_cluster.view(-1, tail_cluster.size(-1))[:i], tail_cluster.view(-1, tail_cluster.size(-1))[i+1:]], dim=0)
                inter_cluster_dist = torch.min(self.distance(cluster_embedding, other_cluster_embeddings))
                inter_cluster_dists.append(inter_cluster_dist)
                inter_cluster_loss = torch.mean(torch.stack(inter_cluster_dists))

            # Compute parent-child distances
            parent_child_dists = []
            for cluster_embedding, parent_cluster_embedding in zip(torch.cat([head_cluster, tail_cluster.view(-1, tail_cluster.size(-1))], dim=0), torch.cat([head_parent_cluster, tail_parent_cluster.view(-1, tail_parent_cluster.size(-1))], dim=0)):
                parent_child_dist = self.distance(cluster_embedding, parent_cluster_embedding)
                parent_child_dists.append(parent_child_dist)
            parent_child_loss = torch.mean(torch.stack(parent_child_dists))
            
            hierarchical_loss = intra_cluster_loss - self.lambda_1 * inter_cluster_loss + self.lambda_2 * parent_child_loss
            
            score = - self.alpha * hierarchical_loss
            score = score - self.beta * (head_text_dist + tail_text_dist)
            
            true_triple_score = self.score_func(head_combined, relation, tail_combined)
            negative_tails = self.sample_negative_entities(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            negative_tail_scores = self.score_func(head_combined, relation, negative_tails)
            
            score = score - (self.gamma_2 * (true_triple_score - torch.mean(negative_tail_scores, dim=1)))
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        return score

    def get_cluster_embedding(self, entities, entity_embeddings):
        """
        Compute the cluster center embeddings for the given entities based on the provided entity embeddings.
        """
        if isinstance(entities, torch.Tensor):
            entities = entities.tolist()
        
        cluster_embeddings = []
        for entity in entities:
            cluster = self.entity_hierarchy[entity]
            cluster_entities = list(cluster)
            cluster_entity_indices = torch.tensor([entities.index(e) for e in cluster_entities])
            cluster_entity_embeddings = torch.index_select(entity_embeddings, dim=0, index=cluster_entity_indices)
            cluster_embedding = torch.mean(cluster_entity_embeddings, dim=0)
            cluster_embeddings.append(cluster_embedding)
        
        cluster_embeddings = torch.stack(cluster_embeddings, dim=0)
        return cluster_embeddings

    def get_parent_cluster_embedding(self, entities, entity_embeddings):
        """
        Compute the parent cluster embeddings for the given entities based on the provided entity embeddings.
        """
        if isinstance(entities, torch.Tensor):
            entities = entities.tolist()
        
        parent_cluster_embeddings = []
        for entity in entities:
            cluster = self.entity_hierarchy[entity]
            parent_cluster = self.entity_hierarchy[cluster]
            parent_cluster_entities = list(parent_cluster)
            parent_cluster_entity_indices = torch.tensor([entities.index(e) for e in parent_cluster_entities])
            parent_cluster_entity_embeddings = torch.index_select(entity_embeddings, dim=0, index=parent_cluster_entity_indices)
            parent_cluster_embedding = torch.mean(parent_cluster_entity_embeddings, dim=0)
            parent_cluster_embeddings.append(parent_cluster_embedding)
        
        parent_cluster_embeddings = torch.stack(parent_cluster_embeddings, dim=0)
        return parent_cluster_embeddings

    def sample_negative_entities(self, entities, num_samples=10):
        """
        Sample negative entities for the given entities.
        """
        if isinstance(entities, torch.Tensor):
            entities = entities.tolist()
        
        batch_size = len(entities)
        negative_entities = []
        for i in range(batch_size):
            entity_samples = []
            while len(entity_samples) < num_samples:
                negative_entity = torch.randint(0, self.nentity, (1,))
                if negative_entity != entities[i]:
                    entity_samples.append(negative_entity)
            negative_entities.append(torch.cat(entity_samples, dim=0))
        
        negative_entities = torch.stack(negative_entities, dim=0)
        return negative_entities

    def distance(self, embeddings1, embeddings2, metric='cosine'):
        """
        Compute the distance between two sets of embeddings.
        """
        if metric == 'euclidean':
            return torch.norm(embeddings1 - embeddings2, p=2, dim=-1)
        elif metric == 'cosine':
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
            cosine_similarity = torch.sum(embeddings1_norm * embeddings2_norm, dim=-1)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance

    def score_func(self, head, relation, tail, mode='single'):
        """
        Compute the score for the given triple (head, relation, tail).
        """
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
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
    
    
    
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample, mode='single')

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
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
                    'tail-batch'
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
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

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