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
        prompt = f"Please provide a brief description of the entity '{entity}' in the following format:\n\n{entity} is a [description].\n\nFor example:\napple is a round fruit with red, green, or yellow skin and crisp, juicy flesh.\n\nNow, describe {entity}:"

    response = gpt_chat_return_response(model="gpt-3.5-turbo-0125", prompt=prompt)
    description = response.choices[0].message.content.strip()
    return description

def generate_embeddings(args, entity_info, entity_embeddings, dim=1024):
    embeddings = []
    model = "text-embedding-3-large"
    
    entities = list(entity_info.keys())
    original_descriptions = [entity_info[entity]["original_description"] for entity in entities]
    entities_text = [entity_info[entity]["text_label"] for entity in entities]
    
    if os.path.exists(f"{args.output_dir}/{args.dataset}/entity_embeddings.json"):
        print(f"Loading existing entity embeddings from {args.output_dir}/{args.dataset}/entity_embeddings.json...")
        with open(f"{args.output_dir}/{args.dataset}/entity_embeddings.json", 'r') as f:
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
        
        if entity_embeddings[entity] is None or entity_info[entity]["llm_description"] is None:
            entity_embedding = client.embeddings.create(
                input=entity_text,
                model=model,
                dimensions=dim,
            ).data[0].embedding

            description = generate_entity_description(entity_text, hint=ori_desc)
            description_embedding = client.embeddings.create(
                input=description,
                model=model,
                dimensions=dim,
            ).data[0].embedding

            combined_embedding = np.concatenate((entity_embedding, description_embedding))
            
            entity_info[entity]["llm_description"] = description
            entity_embeddings[entity] = combined_embedding.tolist()
            
        return entity_info[entity], entity_embeddings[entity]
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_entity, i) for i in range(len(entities))]
        for future in tqdm(as_completed(futures), total=len(entities)):
            info, embedding = future.result()
            embeddings.append(embedding)
            entity_info[entities[futures.index(future)]] = info
            
            if futures.index(future) % 1000 == 0 or futures.index(future) == len(entities) - 1:
                with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'w') as f:
                    json.dump(entity_info, f, indent=4)
                
                with open(f"{args.output_dir}/{args.dataset}/entity_embeddings.json", 'w') as f:
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
    
    return hierarchy

def seed_hierarchy_construction(args, entities, embeddings, distance_threshold):
    """
    Function to perform agglomerative clustering and build hierarchy using clusters.
    """
    clusters, clustering = agglomerative_clustering(args, entities, embeddings, distance_threshold)
    # Get the root of the hierarchy
    initial_hierarchy = build_hierarchy(clustering.children_, len(entities), entities, clustering)
    seed_hierarchy = build_seed_hierarchy(clusters, initial_hierarchy)

    return seed_hierarchy

def evaluate_threshold(args, entities, embeddings, threshold):
    clusters = agglomerative_clustering(args, entities, embeddings, threshold)
    num_clusters = len(clusters)
    
    if num_clusters == 1:
        logging.info(f"Skipping evaluation for threshold {threshold:.2f}: Only one cluster formed.")
        return threshold, -1.0, None
    elif num_clusters == len(entities):
        logging.info(f"Skipping evaluation for threshold {threshold:.2f}: Each entity is in its own cluster.")
        return threshold, 0.0, None
    else:
        labels = np.zeros(len(entities), dtype=int)
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


# def llm_refine_hierarchy(clusters, max_entities=100, num_threads=4):
#     logging.info(f"Refining clusters...")
#     logging.info(f"Current clusters: {clusters}")
#     refined_clusters = {}

#     def refine_cluster(cluster_name, cluster_entities):
#         remaining_entities = set(cluster_entities)
#         sub_clusters = {}

#         while remaining_entities:
#             if len(remaining_entities) > max_entities:
#                 sampled_entities = random.sample(remaining_entities, max_entities)
#             else:
#                 sampled_entities = list(remaining_entities)

#             example_entities = ["The Godfather", "The Shawshank Redemption", "The Dark Knight", "Forrest Gump", "Inception", "The Matrix"]
#             example_cluster_name = "Movies"
#             example_suggestion_first_iteration = "Genre:\nDrama: The Godfather, The Shawshank Redemption, Forrest Gump\nAction: The Dark Knight\nScience Fiction: Inception, The Matrix"
#             example_suggestion_subsequent_iterations = "Genre:\nDrama: The Shawshank Redemption\nAction: The Dark Knight\nScience Fiction: Inception, The Matrix\n\nOther Characteristics:\nCrime: The Godfather\nPrison: The Shawshank Redemption\nSuperhero: The Dark Knight\nComedy-Drama: Forrest Gump\nMind-Bending: Inception\nDystopian Future: The Matrix"

#             if not sub_clusters:
#                 prompt = f"Given the following entities from the cluster '{cluster_name}':\n{', '.join(sampled_entities)}\n\nAnalyze the entities and determine if they can be grouped into distinct and meaningful sub-clusters based on their characteristics, themes, or genres. If sub-clusters can be formed, provide a clear and concise name for each sub-cluster that represents the common attribute of its entities. If the entities are already well-grouped and don't require further sub-clustering, simply provide a descriptive name for the cluster.\n\nProvide the output in the following format:\nSub-clusters:\n[Sub-cluster Name 1]:\n[Entity 1]\n[Entity 2]\n...\n\n[Sub-cluster Name 2]:\n[Entity 3]\n[Entity 4]\n...\n\nCluster Name: [Descriptive Name]\n\nExample:\nCluster: {example_cluster_name}\nEntities: {', '.join(example_entities)}\nOutput:\n{example_suggestion_first_iteration}\n\nCluster: {cluster_name}\nEntities: {', '.join(sampled_entities)}\nOutput:"
#             else:
#                 prompt = f"Given the following entities from the cluster '{cluster_name}':\n{', '.join(sampled_entities)}\n\nAnd the existing sub-cluster names:\n{', '.join(sub_clusters.keys())}\n\nAnalyze the entities and classify them into the existing sub-clusters based on their characteristics, themes, or genres. If an entity doesn't fit into any existing sub-cluster, you can create a new sub-cluster for it. If the entities are already well-grouped and don't require further sub-clustering, simply provide a descriptive name for the cluster.\n\nProvide the output in the following format:\nSub-clusters:\n[Existing Sub-cluster Name 1]:\n[Entity 1]\n[Entity 2]\n...\n\n[Existing Sub-cluster Name 2]:\n[Entity 3]\n[Entity 4]\n...\n\n[New Sub-cluster Name 1]:\n[Entity 5]\n[Entity 6]\n...\n\nCluster Name: [Descriptive Name]\n\nExample:\nCluster: {example_cluster_name}\nEntities: {', '.join(example_entities)}\nExisting Sub-cluster Names: Genre, Other Characteristics\nOutput:\n{example_suggestion_subsequent_iterations}\n\nCluster: {cluster_name}\nEntities: {', '.join(sampled_entities)}\nExisting Sub-cluster Names: {', '.join(sub_clusters.keys())}\nOutput:"

#             response = gpt_chat_return_response(model="gpt-3.5-turbo-0125", prompt=prompt)
#             suggestion = response.choices[0].message.content.strip()

#             sub_clusters, remaining_entities = update_cluster(sub_clusters, sampled_entities, suggestion, remaining_entities)
#             logging.info(f'sub_clusters: {sub_clusters}, remaining_entities: {remaining_entities}')

#         return cluster_name, sub_clusters

#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = [executor.submit(refine_cluster, cluster_name, cluster_entities) for cluster_name, cluster_entities in clusters.items()]
#         for future in concurrent.futures.as_completed(futures):
#             cluster_name, sub_clusters = future.result()
#             refined_clusters[cluster_name] = sub_clusters

#     return refined_clusters


# def update_cluster(sub_clusters, sampled_entities, suggestion, remaining_entities):
#     suggestion_lines = suggestion.split("\n")
#     current_section = None
#     cluster_name = None

#     for line in suggestion_lines:
#         line = line.strip()
#         if not line:
#             continue

#         if line == "Sub-clusters:":
#             current_section = "sub_clusters"
#         elif line.startswith("Cluster Name:"):
#             current_section = "cluster_name"
#             cluster_name = line.split(":", maxsplit=1)[1].strip()
#         else:
#             if current_section == "sub_clusters":
#                 if ":" in line:
#                     sub_cluster_name = line[:-1].strip()
#                     if sub_cluster_name not in sub_clusters:
#                         sub_clusters[sub_cluster_name] = []
#                 else:
#                     entity = line.strip()
#                     if entity in sampled_entities:
#                         if sub_cluster_name is not None:
#                             sub_clusters[sub_cluster_name].append(entity)
#                         remaining_entities.discard(entity)

#     if cluster_name is not None and not sub_clusters:
#         sub_clusters[cluster_name] = sampled_entities
#         remaining_entities -= set(sampled_entities)

#     return sub_clusters, remaining_entities

def read_entities(path):
    entities = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        entity = line.strip().split("\t")[1]
        entities.append(entity)
    return entities

def create_entity_info_emb_dict(args, entities):
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
    
    return entity_info, entity_embeddings

def construct_args():
    parser = argparse.ArgumentParser(description='Cluster entities using hierarchical clustering and refine the clusters using LLM.')
    parser.add_argument('--output_dir', type=str, default="/data/pj20/lamake_data")
    parser.add_argument('--data_dir', type=str, default="/home/pj20/server-03/lamake/data")
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

    if not os.path.exists(f"{args.output_dir}/{args.dataset}/seed_clusters.json"):
        print("Start Finding Optimal Threshold...")
        # best_threshold, best_clusters = find_optimal_threshold(args, entities_text, embeddings, min_threshold=0.3, max_threshold=0.9, num_thresholds=20)
        best_threshold = 0.52

        print(f"Best Threshold: {best_threshold:.2f}")
        print("Start Creating Seed Clusters ...")
        seed_clusters = seed_hierarchy_construction(args, entities_text, embeddings, best_threshold)
        with open(f"{args.output_dir}/{args.dataset}/seed_clusters.json", 'w') as f:
            json.dump(seed_clusters, f, indent=4)
    else:
        print(f"Loading existing seed clusters from {args.output_dir}/{args.dataset}/seed_clusters.json...")
        with open(f"{args.output_dir}/{args.dataset}/seed_clusters.json", 'r') as f:
            seed_clusters = json.load(f)
        print("Done.")
        
    # print("Start Refining Clusters with LLM...")
    # refined_clusters = llm_refine_hierarchy(seed_clusters, num_threads=args.num_threads, max_entities=args.max_entities)
    
    # print("Saving Clusters...")
    
    # with open(f"{args.output_dir}/{args.dataset}/clusters.json", 'w') as f:
    #     json.dump(refined_clusters, f, indent=4)
        
    # print("Done.")
    
if __name__ == "__main__":
    main() 