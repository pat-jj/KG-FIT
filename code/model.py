import torch
import torch.nn as nn
import torch.nn.functional as F

class LAMAKE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False,
                 entity_hierarchy=None, entity_text_embeddings=None, alpha=0.5, beta=0.5, gamma_1=1.0, gamma_2=1.0):
        super(LAMAKE, self).__init__()
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
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
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
        self.alpha = alpha
        self.beta = beta
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
    def forward(self, sample, mode='single'):
        if mode == 'single':
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
            
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 0]).unsqueeze(1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 2]).unsqueeze(1)
            
            head_cluster = self.get_cluster_embedding(sample[:, 0])
            tail_cluster = self.get_cluster_embedding(sample[:, 2])
            
            head_parent_cluster = self.get_parent_cluster_embedding(sample[:, 0])
            tail_parent_cluster = self.get_parent_cluster_embedding(sample[:, 2])
            
            head_combined = self.alpha * head + (1 - self.alpha) * head_text
            tail_combined = self.alpha * tail + (1 - self.alpha) * tail_text
            
            head_text_dist = self.distance(head_combined, head_text)
            tail_text_dist = self.distance(tail_combined, tail_text)
            
            head_cluster_dist = self.distance(head_combined, head_cluster)
            tail_cluster_dist = self.distance(tail_combined, tail_cluster)
            
            head_parent_cluster_dist = self.distance(head_cluster, head_parent_cluster)
            tail_parent_cluster_dist = self.distance(tail_cluster, tail_parent_cluster)
            
            score = self.score_func(head_combined, relation, tail_combined)
            
            pred_tail = self.predict_tail(head_combined, relation)
            true_tail_dist = self.distance(pred_tail, tail_combined + tail_text)
            
            negative_tails = self.sample_negative_entities(sample[:, 2])
            negative_tail_dists = self.distance(pred_tail, negative_tails + torch.index_select(self.entity_text_embeddings, dim=0, index=negative_tails.view(-1)).view(negative_tails.size()))
            
            # Apply constraints
            score = score - self.beta * (head_cluster_dist + tail_cluster_dist)  # Hierarchical constraint
            score = score - self.gamma_1 * (head_text_dist + tail_text_dist)  # Text embedding deviation constraint
            score = score - self.gamma_2 * (head_parent_cluster_dist + tail_parent_cluster_dist)  # Parent-child constraint
            score = score - (true_tail_dist - torch.mean(negative_tail_dists))  # Link prediction constraint
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            
            head_cluster = self.get_cluster_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_cluster = self.get_cluster_embedding(tail_part[:, 2])
            
            head_parent_cluster = self.get_parent_cluster_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_parent_cluster = self.get_parent_cluster_embedding(tail_part[:, 2])
            
            head_combined = self.alpha * head + (1 - self.alpha) * head_text
            tail_combined = self.alpha * tail + (1 - self.alpha) * tail_text
            
            head_text_dist = self.distance(head_combined, head_text)
            tail_text_dist = self.distance(tail_combined, tail_text)
            
            head_cluster_dist = self.distance(head_combined, head_cluster)
            tail_cluster_dist = self.distance(tail_combined, tail_cluster)
            
            head_parent_cluster_dist = self.distance(head_cluster, head_parent_cluster)
            tail_parent_cluster_dist = self.distance(tail_cluster, tail_parent_cluster)
            
            score = self.score_func(head_combined, relation, tail_combined, mode)
            
            pred_head = self.predict_head(relation, tail_combined)
            true_head_dist = self.distance(pred_head, head_combined + head_text)
            
            negative_heads = self.sample_negative_entities(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            negative_head_dists = self.distance(pred_head, negative_heads + torch.index_select(self.entity_text_embeddings, dim=0, index=negative_heads.view(-1)).view(negative_heads.size()))
            
            # Apply constraints
            score = score - self.beta * (head_cluster_dist + tail_cluster_dist)  # Hierarchical constraint
            score = score - self.gamma_1 * (head_text_dist + tail_text_dist)  # Text embedding deviation constraint
            score = score - self.gamma_2 * (head_parent_cluster_dist + tail_parent_cluster_dist)  # Parent-child constraint
            score = score - (true_head_dist - torch.mean(negative_head_dists, dim=1))  # Link prediction constraint
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part[:, 0]).unsqueeze(1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_cluster = self.get_cluster_embedding(head_part[:, 0])
            tail_cluster = self.get_cluster_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_parent_cluster = self.get_parent_cluster_embedding(head_part[:, 0])
            tail_parent_cluster = self.get_parent_cluster_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_combined = self.alpha * head + (1 - self.alpha) * head_text
            tail_combined = self.alpha * tail + (1 - self.alpha) * tail_text
            
            head_text_dist = self.distance(head_combined, head_text)
            tail_text_dist = self.distance(tail_combined, tail_text)
            
            head_cluster_dist = self.distance(head_combined, head_cluster)
            tail_cluster_dist = self.distance(tail_combined, tail_cluster)
            
            head_parent_cluster_dist = self.distance(head_cluster, head_parent_cluster)
            tail_parent_cluster_dist = self.distance(tail_cluster, tail_parent_cluster)
            
            score = self.score_func(head_combined, relation, tail_combined, mode)
            
            pred_tail = self.predict_tail(head_combined, relation)
            true_tail_dist = self.distance(pred_tail, tail_combined + tail_text)
            
            negative_tails = self.sample_negative_entities(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            negative_tail_dists = self.distance(pred_tail, negative_tails + torch.index_select(self.entity_text_embeddings, dim=0, index=negative_tails.view(-1)).view(negative_tails.size()))
            
            # Apply constraints
            score = score - self.beta * (head_cluster_dist + tail_cluster_dist)  # Hierarchical constraint
            score = score - self.gamma_1 * (head_text_dist + tail_text_dist)  # Text embedding deviation constraint
            score = score - self.gamma_2 * (head_parent_cluster_dist + tail_parent_cluster_dist)  # Parent-child constraint
            score = score - (true_tail_dist - torch.mean(negative_tail_dists, dim=1))  # Link prediction constraint
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        return score
    
    def get_cluster_embedding(self, entities):
        if isinstance(entities, torch.Tensor):
            entities = entities.tolist()
        
        cluster_embeddings = []
        for entity in entities:
            cluster = self.entity_hierarchy[entity]
            cluster_embedding = torch.mean(torch.index_select(self.entity_embedding, dim=0, index=torch.tensor(list(cluster))), dim=0)
            cluster_embeddings.append(cluster_embedding)
        
        cluster_embeddings = torch.stack(cluster_embeddings, dim=0)
        return cluster_embeddings
    
    def get_parent_cluster_embedding(self, entities):
        if isinstance(entities, torch.Tensor):
            entities = entities.tolist()
        
        parent_cluster_embeddings = []
        for entity in entities:
            cluster = self.entity_hierarchy[entity]
            parent_cluster = self.entity_hierarchy[cluster]
            parent_cluster_embedding = torch.mean(torch.index_select(self.entity_embedding, dim=0, index=torch.tensor(list(parent_cluster))), dim=0)
            parent_cluster_embeddings.append(parent_cluster_embedding)
        
        parent_cluster_embeddings = torch.stack(parent_cluster_embeddings, dim=0)
        return parent_cluster_embeddings
    
    def predict_tail(self, head, relation):
        model_func = {
            'TransE': self.TransE_predict_tail,
            'DistMult': self.DistMult_predict_tail,
            'ComplEx': self.ComplEx_predict_tail,
            'RotatE': self.RotatE_predict_tail,
            'pRotatE': self.pRotatE_predict_tail
        }
        
        if self.model_name in model_func:
            pred_tail = model_func[self.model_name](head, relation)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return pred_tail
    
    def predict_head(self, relation, tail):
        model_func = {
            'TransE': self.TransE_predict_head,
            'DistMult': self.DistMult_predict_head,
            'ComplEx': self.ComplEx_predict_head,
            'RotatE': self.RotatE_predict_head,
            'pRotatE': self.pRotatE_predict_head
        }
        
        if self.model_name in model_func:
            pred_head = model_func[self.model_name](relation, tail)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return pred_head
    
    def TransE_predict_tail(self, head, relation):
        return head + relation
    
    def TransE_predict_head(self, relation, tail):
        return tail - relation
    
    def DistMult_predict_tail(self, head, relation):
        return head * relation
    
    def DistMult_predict_head(self, relation, tail):
        return tail * relation

    def ComplEx_predict_tail(self, head, relation):
        head_re, head_im = torch.chunk(head, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        return torch.cat([head_re * relation_re - head_im * relation_im,
                        head_re * relation_im + head_im * relation_re], dim=2)

    def ComplEx_predict_head(self, relation, tail):
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        return torch.cat([tail_re * relation_re + tail_im * relation_im,
                        tail_re * relation_im - tail_im * relation_re], dim=2)

    def RotatE_predict_tail(self, head, relation):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(relation, 2, dim=2)

        re_score = re_head * re_tail - im_head * im_tail
        im_score = re_head * im_tail + im_head * re_tail

        return torch.cat([re_score, im_score], dim=2)

    def RotatE_predict_head(self, relation, tail):
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_head, im_head = torch.chunk(relation, 2, dim=2)

        re_score = re_tail * re_head + im_tail * im_head
        im_score = re_tail * im_head - im_tail * re_head

        return torch.cat([re_score, im_score], dim=2)

    def pRotatE_predict_tail(self, head, relation):
        phase_head = head / (self.embedding_range.item() / self.pi)
        phase_relation = relation / (self.embedding_range.item() / self.pi)
        
        re_score = torch.cos(phase_head + phase_relation)
        im_score = torch.sin(phase_head + phase_relation)
        
        return torch.cat([re_score, im_score], dim=2)

    def pRotatE_predict_head(self, relation, tail):
        phase_tail = tail / (self.embedding_range.item() / self.pi)
        phase_relation = relation / (self.embedding_range.item() / self.pi)
        
        re_score = torch.cos(phase_tail - phase_relation)
        im_score = torch.sin(phase_tail - phase_relation)
        
        return torch.cat([re_score, im_score], dim=2)

    def sample_negative_entities(self, entities, num_samples=10):
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
        negative_entities = torch.index_select(self.entity_embedding, dim=0, index=negative_entities.view(-1)).view(batch_size, num_samples, -1)
        
        return negative_entities

    def distance(self, embeddings1, embeddings2):
        return torch.norm(embeddings1 - embeddings2, p=2, dim=-1)

    def score_func(self, head, relation, tail, mode='single'):
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
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
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
    
    
def train_lamake(model, optimizer, train_iterator, args):
    model.train()
    optimizer.zero_grad()
    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

    if args.cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
        
    negative_score = model((positive_sample, negative_sample), mode=mode)

    if args.negative_adversarial_sampling:
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        
    positive_score = model(positive_sample)
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

    if args.uni_weight:
        positive_sample_loss = -positive_score.mean()
        negative_sample_loss = -negative_score.mean()
    else:
        positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        
    loss = (positive_sample_loss + negative_sample_loss) / 2

    if args.regularization != 0.0:
        regularization = args.regularization * (
            model.entity_embedding.norm(p=3) ** 3 +
            model.relation_embedding.norm(p=3).norm(p=3) ** 3
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


def main(args):
    
    # Load the entity hierarchy and text embeddings
    entity_hierarchy = load_entity_hierarchy(args.hierarchy_path)
    entity_text_embeddings = load_entity_text_embeddings(args.text_embedding_path)

    # Create the LAMAKE model
    lamake_model = LAMAKE(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        entity_hierarchy=entity_hierarchy,
        entity_text_embeddings=entity_text_embeddings,
        alpha=args.alpha,
        beta=args.beta,
        gamma_1=args.gamma_1,
        gamma_2=args.gamma_2
    )

    # ...

    # Training loop
    for step in range(init_step, args.max_steps):
        log = train_lamake(lamake_model, optimizer, train_iterator, args)
        
        # ...
        
    # ...