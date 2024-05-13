import numpy as np
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
from typing import List
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import argparse
import json
import os
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import pickle
from collections import defaultdict
from utils import *

with open('./openai_api.key', 'r') as f:
    api_key = f.read().strip()
    

client = OpenAI(api_key=api_key)

def gpt_chat_return_response(model, prompt, seed=44):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0,
        seed=seed,
        logprobs=True
    )
    return response


def generate_entity_description(entity, hint=None):
    if hint:
        prompt = f"Please provide a brief description of the entity '{entity}' in the following format:\n\n{entity} is a [description].\n\nFor example:\napple is a round fruit with red, green, or yellow skin and crisp, juicy flesh.\n\nHINT:{hint}\n\nNow, describe {entity}:"
    else:
        # prompt = f"Please provide a brief description of the entity '{entity}' in the following format:\n\n{entity} is a [description].\n\nFor example:\napple is a round fruit with red, green, or yellow skin and crisp, juicy flesh.\n\nNow, describe {entity}:"
        prompt = f"Please provide a brief description of the entity '{entity}' in the following format:\n\n{entity} is a [description].\n\nFor example:\nBill Gates is a technology magnate, philanthropist, and co-founder of Microsoft Corporation, known for his significant contributions to the personal computing industry.\n\nNow, describe {entity}:"

    response = gpt_chat_return_response(model="gpt-3.5-turbo-0125", prompt=prompt)
    description = response.choices[0].message.content.strip()
    return description

def generate_embeddings(args, entity_info, entity_embeddings, dim=1024):
    embeddings = []
    model = "text-embedding-3-large"
    
    entities = list(entity_info.keys())
    original_descriptions = [entity_info[entity]["original_description"] for entity in entities]
    entities_text = [entity_info[entity]["text_label"] for entity in entities]
    
    if os.path.exists(f"{args.output_dir}/{args.dataset}/entity_init_embeddings.json"):
        print(f"Loading existing entity embeddings from {args.output_dir}/{args.dataset}/entity_init_embeddings.json...")
        with open(f"{args.output_dir}/{args.dataset}/entity_init_embeddings.json", 'r') as f:
            entity_embeddings = json.load(f)
        print("Done.")
        
        # Check if all entities have valid embeddings
        if all(entity_embeddings[entity] is not None for entity in entities):
            print("All entities have valid embeddings. Skipping embedding generation.")
            print(f"Loading existing entity info from {args.output_dir}/{args.dataset}/entity_info.json...")
            with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'r') as f:
                entity_info = json.load(f)
            print("Done.")
            return np.array([entity_embeddings[entity] for entity in entities]), entity_info, entity_embeddings
    
    if os.path.exists(f"{args.output_dir}/{args.dataset}/entity_info.json"):
        print(f"Loading existing entity info from {args.output_dir}/{args.dataset}/entity_info.json...")
        with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'r') as f:
            entity_info = json.load(f)
    
    print(f"Generating embeddings for entities (dim={dim}) and their descriptions...")
    
    def process_entity(i):
        entity = entities[i]
        ori_desc = original_descriptions[i] if original_descriptions else None
        entity_text = entities_text[i]
        
        if entity_info[entity]["llm_description"] is None:
            description = generate_entity_description(entity_text, hint=ori_desc)
            # print(f"Entity {entity} - Description: {description}")
            entity_info[entity]["llm_description"] = description
        else:
            description = entity_info[entity]["llm_description"]
        
        if entity_embeddings[entity] is None:
            # print(f"Generating embeddings for entity {entity}...")
            entity_embedding = client.embeddings.create(
                input=entity_text,
                model=model,
                dimensions=dim,
            ).data[0].embedding

            description_embedding = client.embeddings.create(
                input=description,
                model=model,
                dimensions=dim,
            ).data[0].embedding

            combined_embedding = np.concatenate((entity_embedding, description_embedding))
            
            entity_embeddings[entity] = combined_embedding.tolist()
            
        return entity_info[entity], entity_embeddings[entity]
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_entity, i) for i in range(len(entities))]
        for future in tqdm(as_completed(futures), total=len(entities)):
            info, embedding = future.result()
            embeddings.append(embedding)
            entity_info[entities[futures.index(future)]] = info
            
            # if futures.index(future) % 10000 == 0 or futures.index(future) == len(entities) - 1:
            if futures.index(future) == len(entities) - 1:
                with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'w') as f:
                    json.dump(entity_info, f, indent=4)
                
                with open(f"{args.output_dir}/{args.dataset}/entity_init_embeddings.json", 'w') as f:
                    json.dump(entity_embeddings, f, indent=4)
        
    return np.array(embeddings), entity_info, entity_embeddings

def perform_or_load_clustering(args, entities, embeddings, distance_threshold):
    # Check if clustering exists, if not perform clustering
    clustering_file = f"{args.output_dir}/{args.dataset}/clustering/clustering_{distance_threshold:.2f}.pkl"
    if not os.path.exists(clustering_file):
        print(f"Performing agglomerative clustering with distance threshold {distance_threshold:.2f}...")
        clustering = AgglomerativeClustering(metric='cosine', linkage='average', distance_threshold=distance_threshold, n_clusters=None)
        clustering.fit(embeddings)
        with open(clustering_file, 'wb') as f:
            pickle.dump(clustering, f)
    else:
        print(f"Loading existing clustering from {clustering_file}...")
        with open(clustering_file, 'rb') as f:
            clustering = pickle.load(f)
    return clustering


def agglomerative_clustering(args, entities, embeddings, distance_threshold):
    """
    Perform or load agglomerative clustering and return clusters along with clustering object
    """
    clustering = perform_or_load_clustering(args, entities, embeddings, distance_threshold)
    
    clusters = {}
    for i in range(clustering.n_clusters_):
        cluster_indices = np.where(clustering.labels_ == i)[0]
        cluster_entities = [entities[idx] for idx in cluster_indices]
        clusters[f"Cluster_{i+1}"] = cluster_entities
        
    return clusters, clustering

def build_hierarchy(children, n_leaves, entity_labels, clustering):
    """
    Builds a nested dictionary representing the cluster hierarchy with cluster IDs as keys.
    """
    # Initialize with leaf nodes
    hierarchy = {f"Leaf_{i}": entity_labels[i] for i in range(n_leaves)}
    next_cluster_id = n_leaves  # Start numbering clusters from the number of leaves

    # Intermediate nodes formed from merging children
    for i, (left, right) in enumerate(children):
        cluster_id = f"Cluster_{next_cluster_id}"
        hierarchy[cluster_id] = {f"Cluster_{left}": hierarchy.pop(f"Leaf_{left}" if left < n_leaves else f"Cluster_{left}"),
                                 f"Cluster_{right}": hierarchy.pop(f"Leaf_{right}" if right < n_leaves else f"Cluster_{right}")}
        next_cluster_id += 1

    # Return the root of the hierarchy
    return hierarchy[f"Cluster_{next_cluster_id - 1}"]

def build_seed_hierarchy(clusters, initial_hierarchy):
    """
    Builds a nested dictionary representing the cluster hierarchy.
    """
    print("Building seed hierarchy...")
    leaf_keys, leaf_values = find_leaves(initial_hierarchy)
    child2parent = map_child_to_parent(initial_hierarchy)
    clusters_ = {int(i): entities for i, entities in enumerate(clusters.values())}
    entity2clusterid = {}

    for i, cluster in enumerate(clusters_.values()):
        for entity in cluster:
            entity2clusterid[entity] = i
            
    clusterid2count = defaultdict(int)
    
    hierarchy = label_(initial_hierarchy, entity2clusterid, clusterid2count)
    hierarchy = refine_1(hierarchy, clusters_)
    hierarchy = refine_2(hierarchy)
    hierarchy = refine_3(hierarchy)
    hierarchy = refine_4(hierarchy)
    
    return hierarchy

def seed_hierarchy_construction(args, entities, embeddings, distance_threshold):
    """
    Function to perform agglomerative clustering and build hierarchy using clusters.
    """
    clusters, clustering = agglomerative_clustering(args, entities, embeddings, distance_threshold)
    # Get the root of the hierarchy
    initial_hierarchy = build_hierarchy(clustering.children_, len(entities), entities, clustering)
    seed_hierarchy = build_seed_hierarchy(clusters, initial_hierarchy)
    
    seed_hierarchy = {
        "Cluster_top": seed_hierarchy
    }

    return seed_hierarchy

def evaluate_threshold(args, entities, embeddings, threshold):
    clusters = agglomerative_clustering(args, entities, embeddings, threshold)
    logging.info(f"Threshold {threshold:.2f} - Number of clusters: {len(clusters)}")
    logging.info(f"Clusters: {clusters}")
    num_clusters = len(clusters)
    
    if num_clusters == 1:
        logging.info(f"Skipping evaluation for threshold {threshold:.2f}: Only one cluster formed.")
        return threshold, -1.0, None
    elif num_clusters == len(entities):
        logging.info(f"Skipping evaluation for threshold {threshold:.2f}: Each entity is in its own cluster.")
        return threshold, 0.0, None
    else:
        labels = np.zeros(len(entities), dtype=int)
        if type(clusters) == tuple:
            clusters = clusters[0]
        for i, cluster_entities in enumerate(clusters.values()):
            indices = [entities.index(entity) for entity in cluster_entities]
            labels[indices] = i
        score = silhouette_score(embeddings, labels)
        logging.info(f"Threshold {threshold:.2f} - Silhouette score: {score:.3f}")
        return threshold, score, clusters


def find_optimal_threshold(args, entities, embeddings, min_threshold=0.1, max_threshold=1.0, num_thresholds=10):
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    logging.info(f"Starting threshold evaluation with {num_thresholds} thresholds: {thresholds}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(evaluate_threshold, args, entities, embeddings, threshold) for threshold in thresholds]
        
        best_score = -1.0
        best_threshold = None
        best_clusters = None
        
        for future in concurrent.futures.as_completed(futures):
            threshold, score, clusters = future.result()
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_clusters = clusters
    
    if best_threshold is None:
        logging.info("No suitable clustering found. Returning the clustering with the highest threshold.")
        best_threshold = thresholds[-1]
        best_clusters = agglomerative_clustering(args, entities, embeddings, best_threshold)
    
    return best_threshold, best_clusters


def read_entities(path):
    entities = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        entity = line.strip().split("\t")[1]
        entities.append(entity)
    return entities


def create_entity_info_emb_dict(args, entities):
    if args.dataset == "FB15K-237":
        ori_info_path = f"{args.data_dir}/{args.dataset}/entity2info.json"
        entity2label_path = f"{args.data_dir}/{args.dataset}/entity2label.txt"
        entity2label = {}
        with open(ori_info_path, 'r') as f:
            ori_info = json.load(f)
        with open(entity2label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            entity, label = line.strip().split("\t")
            entity2label[entity] = label
        
        entity_info = {}
        entity_embeddings = {}
        for entity in entities:
            if entity not in ori_info:
                entity_info[entity] = {
                    "text_label": entity2label[entity],
                    "original_description": None,
                    "llm_description": None
                }
            else:
                entity_info[entity] = {
                    "text_label": ori_info[entity]["label"],
                    "original_description": ori_info[entity]["description"] if "description" in ori_info[entity] else None,
                    "llm_description": None
                }
            entity_embeddings[entity] = None
            
    elif args.dataset == "YAGO3-10":
        entity2label = {entity: entity.replace("_", " ") for entity in entities}
        entity_info = {}
        entity_embeddings = {}
        for entity in entities:
            entity_info[entity] = {
                "text_label": entity2label[entity],
                "original_description": None,
                "llm_description": None
            }
            entity_embeddings[entity] = None
            
    elif args.dataset == "WN18RR":
        entity_info = {}
        entity_embeddings = {}
        ori_info_path = f"{args.data_dir}/{args.dataset}/entity2text.txt"
        with open(ori_info_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            entity, text_info = line.strip().split("\t")
            # print(entity, text_info)
            text_label = text_info.split(", ")[0]
            original_description = text_info.split(text_label + ", ")[1]
            entity_info[entity] = {
                "text_label": text_label,
                "original_description": original_description,
                "llm_description": None
            }
            entity_embeddings[entity] = None
    
    return entity_info, entity_embeddings


def labeling_hierarchy_to_entities(hierarchy, entity_info):
    """
    Label the hierarchy information to entities.
    """
    leaf_keys, leaf_values = find_leaves(hierarchy)
    
    label2entity = defaultdict(list)
    for entity in entity_info.keys():
        label2entity[entity_info[entity]['text_label']].append(entity)
    
    for i in tqdm(range(len(leaf_values))):
        for entity_label in leaf_values[i]:
            entities = label2entity[entity_label]
            parent_path, _ = node2parentpath(hierarchy, leaf_keys[i])
            parent_map = map_child_to_parent(hierarchy)
            nearest_clusters_lca = find_nearest_keys_lca_based(hierarchy, leaf_keys[i], parent_map, m=5)
            
            for entity in entities:
                entity_info[entity]['cluster'] = leaf_keys[i]
                entity_info[entity]['parent_path'] = parent_path
                entity_info[entity]['nearest_clusters_lca'] = nearest_clusters_lca
    
    return entity_info


def construct_args():
    parser = argparse.ArgumentParser(description='Cluster entities using hierarchical clustering and refine the clusters using LLM.')
    parser.add_argument('--output_dir', type=str, default="/shared/pj20/lamake_data")
    parser.add_argument('--data_dir', type=str, default="/home/pj20/server-03/lamake/data")
    parser.add_argument('--hier_type', type=str, default="seed", choices=['seed', 'llm'], help='Type of hierarchy to construct. Default: seed.')
    parser.add_argument('--dataset', type=str, default="FB15K-237", help='Path to the dataset file containing the list of entities to cluster.')
    parser.add_argument('--dimensions', type=int, default=1024, help='Dimensionality of the embeddings. Default: 1024.')
    parser.add_argument('--num_threads', type=int, default=10, help='Number of threads to use for multi-threaded processes. Default: 10.')
    parser.add_argument('--max_entities', type=int, default=100, help='Maximum number of entities to include in an LLM request. Default: 100.')
    
    args = parser.parse_args()
    args.log_dir = f"{args.output_dir}/{args.dataset}/logs"
    
    return args

def main():
    args = construct_args()
    
    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'output.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
    entities = read_entities(f"{args.data_dir}/{args.dataset}/entities.dict")
    entity_info, entity_embeddings = create_entity_info_emb_dict(args, entities)
    
    entities_text, original_descriptions = [], []
    for entity in entities:
        entities_text.append(entity_info[entity]["text_label"])
        original_descriptions.append(entity_info[entity]["original_description"])
        
    print("Start Generating Embeddings...")
    embeddings, entity_info, entity_embeddings = generate_embeddings(args, entity_info=entity_info, entity_embeddings=entity_embeddings, dim=args.dimensions)

    if not os.path.exists(f"{args.output_dir}/{args.dataset}/seed_hierarchy.json"):
        print("Start Finding Optimal Threshold...")
        best_threshold, best_clusters = find_optimal_threshold(args, entities_text, embeddings, min_threshold=0.84, max_threshold=0.99, num_thresholds=5) 
        # best_threshold = 0.52  #FB15K-237
        # best_threshold = 0.49  # YAGO3-10
        # best_threshold = 0.84  # WN18RR

        print(f"Best Threshold: {best_threshold:.2f}")
        print("Start Creating Seed Clusters ...")
        seed_hierarchy = seed_hierarchy_construction(args, entities_text, embeddings, best_threshold)
        with open(f"{args.output_dir}/{args.dataset}/seed_hierarchy.json", 'w') as f:
            json.dump(seed_hierarchy, f, indent=4)

    else:
        print(f"Loading existing seed clusters from {args.output_dir}/{args.dataset}/seed_hierarchy.json...")
        with open(f"{args.output_dir}/{args.dataset}/seed_hierarchy.json", 'r') as f:
            seed_hierarchy = json.load(f)
            
    seed_hierarchy_int, _, key_map, key_map_inv = rename_clusters_to_ints(seed_hierarchy)
    print("Done.")

    print("Start Computing Clusters Embeddings...")
    if not os.path.exists(f"{args.output_dir}/{args.dataset}/clusters_embeddings_seed.json"):
        label2entity = {entity_info[entity]['text_label']: entity for entity in entity_info.keys()}
        clusters_embeddings = compute_clusters_embeddings(clusters=seed_hierarchy, entity_embeddings=entity_embeddings, label2entity=label2entity)
        for cluster_id in clusters_embeddings:
            clusters_embeddings[cluster_id] = clusters_embeddings[cluster_id].tolist()
        with open(f"{args.output_dir}/{args.dataset}/clusters_embeddings_seed.json", 'w') as f:
            json.dump(clusters_embeddings, f, indent=4)
            
        embs = []
        for i in range(len(key_map_inv)):
            original_key = key_map_inv[i]
            emb = clusters_embeddings[original_key]
            embs.append(emb)
        embs = np.array(embs)
        np.save(f"{args.output_dir}/{args.dataset}/clusters_embeddings_seed.npy", embs)
    print("Done.")
    
    print("Start labeling hierarchy information to entities...")
    if not os.path.exists(f"{args.output_dir}/{args.dataset}/entity_info_seed_hier.json"):
        entity_info_seed_hier = labeling_hierarchy_to_entities(seed_hierarchy_int, entity_info)
        with open(f"{args.output_dir}/{args.dataset}/entity_info_seed_hier.json", 'w') as f:
            json.dump(entity_info_seed_hier, f, indent=4)
    print("Done.")
    
    if not os.path.exists(f"{args.output_dir}/{args.dataset}/entity_init_embeddings.npy"):
        print("Sort Entity Embeddings by ID...")
        with open(f"{args.data_dir}/{args.dataset}/entities.dict", 'r') as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
        sorted_entity_embeddings = sort_entity_embeddings(entity_embeddings_dict=entity_embeddings, entity2id=entity2id)
        # save the sorted entity embeddings
        np.save(f"{args.output_dir}/{args.dataset}/entity_init_embeddings.npy", sorted_entity_embeddings)
        
    
    if not os.path.exists(f"{args.output_dir}/{args.dataset}/llm_hierarchy.json"):
        print('Please first use llm_refine.py to refine the seed hierarchy using LLM, and generate llm_hierarchy.json.')
    else:
        print(f"Loading existing llm clusters from {args.output_dir}/{args.dataset}/llm_hierarchy.json...")
        with open(f"{args.output_dir}/{args.dataset}/llm_hierarchy.json", 'r') as f:
            llm_hierarchy = json.load(f)
        
        llm_hierarchy = rename_unique_keys(llm_hierarchy)
            
        llm_hierarchy_int, _, key_map, key_map_inv = rename_clusters_to_ints(llm_hierarchy)
        with open(f"{args.output_dir}/{args.dataset}/tmp/llm_hierarchy_int.json", 'w') as f:
            json.dump(llm_hierarchy_int, f, indent=4)
        with open(f"{args.output_dir}/{args.dataset}/tmp/key_map_inv.json", 'w') as f:
            json.dump(key_map_inv, f, indent=4)
        print("Done.")
        
        print("Start Computing Clusters Embeddings...")
        if not os.path.exists(f"{args.output_dir}/{args.dataset}/clusters_embeddings_llm.json"):
            label2entity = {entity_info[entity]['text_label']: entity for entity in entity_info.keys()}
            clusters_embeddings = compute_clusters_embeddings(clusters=llm_hierarchy, entity_embeddings=entity_embeddings, label2entity=label2entity)
            for cluster_id in clusters_embeddings:
                clusters_embeddings[cluster_id] = clusters_embeddings[cluster_id].tolist()
            with open(f"{args.output_dir}/{args.dataset}/clusters_embeddings_llm.json", 'w') as f:
                json.dump(clusters_embeddings, f, indent=4)
                
            embs = []
            for i in range(len(key_map_inv)):
                if i in key_map_inv.keys():
                    original_key = key_map_inv[i]
                    emb = clusters_embeddings[original_key]
                else:
                    emb = np.zeros(len(embs[0]))
                embs.append(emb)
            embs = np.array(embs)
            np.save(f"{args.output_dir}/{args.dataset}/clusters_embeddings_llm.npy", embs)
        print("Done.")
        
        print("Start labeling hierarchy information to entities...")
        if not os.path.exists(f"{args.output_dir}/{args.dataset}/entity_info_llm_hier.json"):
            entity_info_llm_hier = labeling_hierarchy_to_entities(llm_hierarchy_int, entity_info)
            with open(f"{args.output_dir}/{args.dataset}/entity_info_llm_hier.json", 'w') as f:
                json.dump(entity_info_llm_hier, f, indent=4)
                
        print("Done.")
        
        
    
    
    
if __name__ == "__main__":
    main() 