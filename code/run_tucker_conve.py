from load_data_tucker_conve import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model_tucker_conve import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
from tqdm import tqdm
from utils import read_entity_initial_embedding, read_cluster_embeddings, read_entity_info
from dataloader import TrainDataset
from torch.utils.data import DataLoader
import wandb
    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.1, model='tucker', dataset='infer'):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.model = model
        self.cuda = cuda
        self.dataset = dataset
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    
    def get_entity_info_data(self, entity_info, entity_id):
        cluster_id_head, neighbor_clusters_ids_head, \
            parent_ids_head, _, _, _ = entity_info[entity_id]
        
        return cluster_id_head, neighbor_clusters_ids_head, parent_ids_head
    
    def process_e1_info(self, e1_info):
        cluster_id = torch.tensor([_[0] for _ in e1_info])
        neighbor_clusters_ids = [torch.LongTensor(_[1]) for _ in e1_info]
        max_len_neighbor = max(len(ids) for ids in neighbor_clusters_ids)
        padded_neighbor_clusters_ids = torch.stack([torch.nn.functional.pad(ids, (0, max_len_neighbor - len(ids)), value=-1) for ids in neighbor_clusters_ids])
        parent_ids = [torch.LongTensor(_[2]) for _ in e1_info]
        max_len_parent = max(len(ids) for ids in parent_ids)
        padded_parent_ids = torch.stack([torch.nn.functional.pad(ids, (0, max_len_parent - len(ids)), value=-1) for ids in parent_ids])
        return cluster_id, padded_neighbor_clusters_ids, padded_parent_ids
        
    def get_batch(self, er_vocab, er_vocab_pairs, entity_info, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = torch.zeros(len(batch), len(d.entities), device='cuda')
        e1_idxs = [pair[0] for pair in batch]
        e1_info = [self.get_entity_info_data(entity_info, e1_idx) for e1_idx in e1_idxs]
        self_cluster_ids, neighbor_clusters_ids, parent_ids = self.process_e1_info(e1_info)
        
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets, self_cluster_ids, neighbor_clusters_ids, parent_ids

    def evaluate(self, model, data, out_lp_constraints=False):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        entity_info_test = read_entity_info(os.path.join(f'{args.process_path}/{args.dataset}',\
            f'entity_info_{args.hierarchy_type}_hier.json'), test_data_idxs, self.idxs_entity)
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _, self_cluster_ids, neighbor_clusters_ids, parent_ids = self.get_batch(er_vocab, test_data_idxs, entity_info_test, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            e1_idx_cpu = e1_idx.tolist()
            r_idx_cpu = r_idx.tolist()
            e2_idx_cpu = e2_idx.tolist()
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
                self_cluster_ids = self_cluster_ids.cuda()
                neighbor_clusters_ids = neighbor_clusters_ids.cuda()
                parent_ids = parent_ids.cuda()
            predictions, text_dist, self_cluster_dist, neighbor_cluster_dist, hier_dist = model.forward(e1_idx, r_idx, self_cluster_ids, neighbor_clusters_ids, parent_ids)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()

            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @5: {0}'.format(np.mean(hits[4])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        
        metrics = {
            'hits@10': np.mean(hits[9]),
            'hits@5': np.mean(hits[4]),
            'hits@3': np.mean(hits[2]),
            'hits@1': np.mean(hits[0]),
            'mr': np.mean(ranks),
            'mrr': np.mean(1./np.array(ranks))
        }
        
        return metrics



    def train_and_eval(self, d, args):
        wandb.init(project="kgfit", config=args, name=f"{args.dataset}-{args.model}-{args.hierarchy_type}-{args.edim}")
        print("Training the {} model...".format(self.model))
        with open(os.path.join(args.data_dir, 'entities.dict')) as fin:
            self.entity_idxs = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity_idxs[entity] = int(eid)
            self.id2entity = {v: k for k, v in self.entity_idxs.items()}
                

        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        self.idxs_entity = {v: k for k, v in self.entity_idxs.items()}
        self.idxs_relation = {v: k for k, v in self.relation_idxs.items()}

        print(f"Number of Entity: {len(self.entity_idxs)}")
        
        # Load the entity hierarchy and text embeddings
        entity_text_embeddings = read_entity_initial_embedding(args)
        # Load the cluster embeddings
        cluster_embeddings = read_cluster_embeddings(args)
        

        if self.model == 'TuckER':
            model = KGFIT_TuckER(
                d, self.ent_vec_dim, self.rel_vec_dim, 
                nentity=len(d.entities), nrelation=len(d.relations),
                entity_text_embeddings=entity_text_embeddings,
                cluster_embeddings=cluster_embeddings,
                distance_metric=args.distance_metric,
                **self.kwargs)
        elif self.model == 'ConvE':
            model = KGFIT_ConvE(d, self.ent_vec_dim, self.rel_vec_dim,
                nentity=len(d.entities), nrelation=len(d.relations),
                entity_text_embeddings=entity_text_embeddings,
                cluster_embeddings=cluster_embeddings, 
                **self.kwargs)
        if self.cuda:
            model.cuda()
            
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        train_data_idxs = self.get_data_idxs(d.train_data)
        entity_info_train = read_entity_info(os.path.join(f'{args.process_path}/{args.dataset}',\
            f'entity_info_{args.hierarchy_type}_hier.json'), train_data_idxs, self.idxs_entity)
        
        print("Number of training data points: %d" % len(train_data_idxs))
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(0, self.num_iterations):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets, self_cluster_ids, neighbor_clusters_ids, parent_ids = self.get_batch(er_vocab, er_vocab_pairs, entity_info_train, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    self_cluster_ids = self_cluster_ids.cuda()
                    neighbor_clusters_ids = neighbor_clusters_ids.cuda()
                    parent_ids = parent_ids.cuda()
                predictions, text_dist, self_cluster_dist, neighbor_cluster_dist, hier_dist = model.forward(e1_idx, r_idx, self_cluster_ids, neighbor_clusters_ids, parent_ids)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))    
                        
                neighbor_cluster_dist_mean = neighbor_cluster_dist.mean(dim=1, keepdim=True)
                hier_dist_mean = hier_dist.mean(dim=1, keepdim=True)
                
                loss = model.loss(predictions, targets)
                loss = model.zeta_3 * loss \
                            + model.zeta_1 * (model.lambda_1 * (self_cluster_dist) \
                                                - model.lambda_2 * (neighbor_cluster_dist_mean) \
                                                - model.lambda_3 * (hier_dist_mean)) \
                            + model.zeta_2 * (text_dist)
                
                loss = loss.mean()
                
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print('Epoch: {}, Time: {}s, Loss: {}'.format(it, time.time()-start_train, np.mean(losses)))
            wandb.log({'train_loss': np.mean(losses)}, step=it)
            model.eval()
            with torch.no_grad():
                if it % 50 == 0:
                    print("Test:")
                    test_metrics = self.evaluate(model, d.test_data, out_lp_constraints=False)
                    wandb.log({f"test_{k}": v for k, v in test_metrics.items()}, step=it)
                if it == self.num_iterations - 1:
                    test_metrics = self.evaluate(model, d.test_data, out_lp_constraints=True)
                    wandb.log({f"test_{k}": v for k, v in test_metrics.items()}, step=it)
                    self.get_train_kge_neg(model, d.train_data)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15K-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument('--process_path', type=str, default='/shared/pj20/lamake_data', help='Path to the entity hierarchy')
    parser.add_argument('--hierarchy_type', type=str, default='seed', choices=['seed', 'llm'],  help='Type of hierarchy to use')
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['euclidean', 'cosine', 'complex', 'pi', 'rotate'],help='Distance metric for link prediction')
    
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--model", type=str, default='TuckER', nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--gpu", type=int, default=2, nargs="?",
                help="Relation embedding dimensionality.") 
    parser.add_argument('--zeta_3', type=float, default=2.0, help='zeta_3')

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "./data/%s/" % dataset
    args.data_dir = data_dir
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, model=args.model, dataset=dataset)
    experiment.train_and_eval(d, args)