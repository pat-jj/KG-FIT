import json
import numpy as np
import os
import torch
import logging
from tqdm import tqdm

def read_hierarchy(args):
    """
    Read the hierarchy of the dataset from the json file.
    """
    file_path = f"{args.process_path}/{args.dataset}/{args.hierarchy_type}_hierarchy.json"
    hierarchy = json.load(open(file_path, "r"))
    
    return hierarchy


def read_entity_initial_embedding(args):
    """Read the initial entity embeddings from the json file."""
    file_path = f"{args.process_path}/{args.dataset}/entity_init_embeddings.npy"
    entity_init_embeddings = np.load(file_path)
    # convert to tensor
    entity_init_embeddings = torch.tensor(entity_init_embeddings, dtype=torch.float32)
    if args.cuda:
        entity_init_embeddings = entity_init_embeddings.cuda()
    return entity_init_embeddings

def read_cluster_embeddings(args):
    """Read the cluster embeddings from the json file."""
    file_path = f"{args.process_path}/{args.dataset}/clusters_embeddings_{args.hierarchy_type}.npy"
    cluster_embeddings = np.load(file_path)
    # convert to tensor
    cluster_embeddings = torch.tensor(cluster_embeddings, dtype=torch.float32)
    if args.cuda:
        cluster_embeddings = cluster_embeddings.cuda()
    return cluster_embeddings


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.get_entity_embedding().detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )
    
    
def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_entity_info(file_path, triples, id2entity):
    '''
    Read entity information from the file.
    '''
    entity_info = []
    with open(file_path, 'r') as f:
        info = json.load(f)
        
    for triple in tqdm(triples):
        head = id2entity[triple[0]]
        tail = id2entity[triple[2]]
        cluster_id_head = info[head]['cluster']
        neighbor_clusters_ids_head = info[head]['nearest_clusters_lca']
        parent_ids_head = info[head]['parent_path']
        cluster_id_tail = info[tail]['cluster']
        neighbor_clusters_ids_tail = info[tail]['nearest_clusters_lca']
        parent_ids_tail = info[tail]['parent_path']
        
        entity_info.append(
            (cluster_id_head, neighbor_clusters_ids_head, parent_ids_head, 
             cluster_id_tail, neighbor_clusters_ids_tail, parent_ids_tail)
        )
        
    return entity_info


def read_entity_info_dict(file_path, triples, id2entity):
    '''
    Read entity information from the file.
    '''
    entity_info = {}
    with open(file_path, 'r') as f:
        info = json.load(f)
        
    for triple in tqdm(triples):
        head = id2entity[triple[0]]
        tail = id2entity[triple[2]]
        cluster_id_head = info[head]['cluster']
        neighbor_clusters_ids_head = info[head]['nearest_clusters_lca']
        parent_ids_head = info[head]['parent_path']
        cluster_id_tail = info[tail]['cluster']
        neighbor_clusters_ids_tail = info[tail]['nearest_clusters_lca']
        parent_ids_tail = info[tail]['parent_path']
        
        entity_info[triple[0]] = (
            cluster_id_head, neighbor_clusters_ids_head, parent_ids_head
        )
        entity_info[triple[2]] = (
            cluster_id_tail, neighbor_clusters_ids_tail, parent_ids_tail
        )
        
    return entity_info
        

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    
def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
