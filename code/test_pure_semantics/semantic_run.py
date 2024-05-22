import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

with open('code/openai_api.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# Helper function to get relation embedding
def get_relation_embedding(relation_, r_dict, mode='llm', args=None):
    if mode == 'llm':
        if relation_ in r_dict:
            return r_dict, r_dict[relation_]
        else:
            relation = relation_.replace("_", " ")
            relation_emb = client.embeddings.create(
                            input=relation,
                            model=f"text-embedding-3-{args.llm_size}",
                            dimensions=1024,
                        ).data[0].embedding
            
            r_dict[relation_] = relation_emb
            
    else:
        if relation_ in r_dict:
            return r_dict[relation_], r_dict[relation_]
        else:
            relation_dict_path = f"data/{args.dataset}/relations.dict"
            relation2id = {}
            with open(relation_dict_path, 'r') as f:
                for line in f:
                    idx, relation = line.strip().split("\t")
                    relation2id[relation] = int(idx)
                    
            relation_embs_path = args.rel_emb_path
            relation_embs = np.load(relation_embs_path)
            r_dict[relation_] = relation_embs[relation2id[relation_]]
        
    return r_dict, r_dict[relation_]

def load_relations(relations, args):
    r_dict = {}
    inv_r_dict = {}
    for relation in relations:
        if args.dataset == 'FB15K-237':
            relation = relation.split('/')[-1].replace('_', ' ')
            print(relation)
        r_dict, emb = get_relation_embedding(relation, r_dict, mode='llm', args=args)
        inv_r_dict, inv_emb = get_relation_embedding(relation + '_inverse', inv_r_dict, mode='llm', args=args)
    return r_dict, inv_r_dict

def get_inverse_relation(relation):
    return relation + '_inverse'

def evaluate_triple(triple, entity_embs, entity_dict, r_dict, inv_r_dict, filter_triples, args):
    h, r, t = triple
    h_idx = entity_dict[h]
    t_idx = entity_dict[t]
    e1_emb = entity_embs[h_idx]
    e2_emb = entity_embs[t_idx]
    r_dict, relation_emb = get_relation_embedding(r, r_dict, mode=args.mode, args=args)
    inv_r_dict, inv_relation_emb = get_relation_embedding(get_inverse_relation(r), inv_r_dict, mode=args.mode, args=args)

    # Compute predicted tail embedding
    predicted_tail_emb = e1_emb + relation_emb
    # normalize
    predicted_tail_emb = predicted_tail_emb / np.linalg.norm(predicted_tail_emb)
    # Compute cosine similarities with all entity embeddings
    similarities_tail = cosine_similarity([predicted_tail_emb], entity_embs)[0]
    # Filter out true triples from train/valid/test set (except the tested triple)
    for true_h, true_r, true_t in filter_triples:
        if true_h == h and true_r == r and true_t != t:
            true_t_idx = entity_dict[true_t]
            similarities_tail[true_t_idx] = -np.inf
    
    similarities_tail[h_idx] = -np.inf
    # Sort entities by similarity
    sorted_indices_tail = np.argsort(similarities_tail)[::-1]
    # Find rank of true tail entity
    rank_tail = np.where(sorted_indices_tail == t_idx)[0][0] + 1
    
    # Compute predicted head embedding using inverse relation
    predicted_head_emb = e2_emb + inv_relation_emb
    # predicted_head_emb = e2_emb - relation_emb
    # normalize
    predicted_head_emb = predicted_head_emb / np.linalg.norm(predicted_head_emb)
    # Compute cosine similarities with all entity embeddings
    similarities_head = cosine_similarity([predicted_head_emb], entity_embs)[0]
    # Filter out true triples from train/valid/test set (except the tested triple)
    for true_h, true_r, true_t in filter_triples:
        if true_h != h and true_r == r and true_t == t:
            true_h_idx = entity_dict[true_h]
            similarities_head[true_h_idx] = -np.inf
    
    similarities_head[t_idx] = -np.inf
    # Sort entities by similarity
    sorted_indices_head = np.argsort(similarities_head)[::-1]
    # Find rank of true head entity
    rank_head = np.where(sorted_indices_head == h_idx)[0][0] + 1
    
    return {
        'triple': (h, r, t),
        'tail_rank': rank_tail,
        'head_rank': rank_head
    }
    
    
def run(args):
    # Load entity embeddings
    if args.mode == 'llm':
        entity_embs = np.load(f"{args.process_path}/{args.dataset}/entity_init_embeddings.npy")
    else:
        entity_embs = np.load(args.ent_emb_path)
        
    # Take the first half dimension of the entity embeddings
    entity_embs = entity_embs[:, :entity_embs.shape[1]//2]

    # Load triples
    triples = []
    relations = set()
    count = 0
    with open(f"{args.data_path}/{args.dataset}/test.txt", "r") as f:
        for line in f:
            # if count < 500:
                h, r, t = line.strip().split("\t")
                triples.append((h, r, t))
                relations.add(r)
            #     count += 1
            # else:
            #     break

    # Load filter triples from train/valid/test set
    filter_triples = []
    for split in ['train', 'valid', 'test']:
        with open(f"{args.data_path}/{args.dataset}/{split}.txt", "r") as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                filter_triples.append((h, r, t))
    
    r_dict, inv_r_dict = load_relations(relations, args)
    
    # Load entity dictionary
    entity_dict = {}
    with open(f"{args.data_path}/{args.dataset}/entities.dict", "r") as f:
        for line in f:
            idx, entity = line.strip().split("\t")
            entity_dict[entity] = int(idx)

    # Metrics initialization
    MR_tail = 0
    MRR_tail = 0
    hits_1_tail = 0
    hits_5_tail = 0
    hits_10_tail = 0
    
    MR_head = 0
    MRR_head = 0
    hits_1_head = 0
    hits_5_head = 0
    hits_10_head = 0
    
    total_triples = len(triples)
    rankings = []

    print("Evaluating model...")

    # Multi-threaded evaluation of triples
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_triple = {executor.submit(evaluate_triple, triple, entity_embs, entity_dict, r_dict, inv_r_dict, filter_triples, args): triple for triple in triples}
        
        for future in tqdm(as_completed(future_to_triple), total=total_triples):
            result = future.result()
            rankings.append(result)
            rank_tail = result['tail_rank']
            rank_head = result['head_rank']

            # Update tail metrics
            MR_tail += rank_tail
            MRR_tail += 1 / rank_tail
            if rank_tail == 1:
                hits_1_tail += 1
            if rank_tail <= 5:
                hits_5_tail += 1
            if rank_tail <= 10:
                hits_10_tail += 1

            # Update head metrics
            MR_head += rank_head
            MRR_head += 1 / rank_head
            if rank_head == 1:
                hits_1_head += 1
            if rank_head <= 5:
                hits_5_head += 1
            if rank_head <= 10:
                hits_10_head += 1

    # Calculate final metrics for tail prediction
    MR_tail /= total_triples
    MRR_tail /= total_triples
    H1_tail = hits_1_tail / total_triples
    H5_tail = hits_5_tail / total_triples
    H10_tail = hits_10_tail / total_triples

    # Calculate final metrics for head prediction
    MR_head /= total_triples
    MRR_head /= total_triples
    H1_head = hits_1_head / total_triples
    H5_head = hits_5_head / total_triples
    H10_head = hits_10_head / total_triples

    # Print metrics
    print("Tail prediction metrics:")
    print(f"MR: {MR_tail}")
    print(f"MRR: {MRR_tail}")
    print(f"Hits@1: {H1_tail}")
    print(f"Hits@5: {H5_tail}")
    print(f"Hits@10: {H10_tail}")
    
    print("Head prediction metrics:")
    print(f"MR: {MR_head}")
    print(f"MRR: {MRR_head}")
    print(f"Hits@1: {H1_head}")
    print(f"Hits@5: {H5_head}")
    print(f"Hits@10: {H10_head}")

    # Write rankings to file
    with open("rankings.txt", "a") as f:
        for result in rankings:
            print(str(result) + '\n', file=f)

def construct_args():
    parser = argparse.ArgumentParser(description='KG-FIT')
    # Data paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to the dataset')
    parser.add_argument('--process_path', type=str, default='/shared/pj20/lamake_data', help='Path to the entity hierarchy')
    parser.add_argument('--dataset', type=str, default='WN18RR', help='Path to the dataset')
    parser.add_argument('--rel_emb_path', type=str, default='/shared/pj20/lamake_data/WN18RR/checkpoints/pRotatE_seed_batch_512_hidden_512_dist_cosine/relation_embedding.npy', help='Path to the dataset')
    parser.add_argument('--ent_emb_path', type=str, default='/shared/pj20/lamake_data/WN18RR/checkpoints/pRotatE_seed_batch_512_hidden_512_dist_cosine/entity_embedding.npy', help='Path to the dataset')
    parser.add_argument('--mode', type=str, default='ft', choices=['llm', 'ft'])
    parser.add_argument('--llm_size', type=str, default='small', choices=['small', 'large'])
    
    return parser.parse_args()

def main():
    args = construct_args()
    run(args)

if __name__ == "__main__":
    main()