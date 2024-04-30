import numpy as np
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
from typing import List
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import argparse
import json
import os


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
        with open(f"{args.output_dir}/{args.dataset}/entity_embeddings.json", 'r') as f:
            entity_embeddings = json.load(f)
        embeddings = [entity_embeddings[entity] for entity in entities]
    
    if os.path.exists(f"{args.output_dir}/{args.dataset}/entity_info.json"):
        with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'r') as f:
            entity_info = json.load(f)
    
    print(f"Generating embeddings for entities (dim={dim}) and their descriptions...")
    for i in tqdm(range(len(entities))):
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
            embeddings.append(combined_embedding)
            
            entity_info[entity]["llm_description"] = description
            entity_embeddings[entity] = combined_embedding.tolist()
            
            if i % 1000 == 0:
                with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'w') as f:
                    json.dump(entity_info, f, indent=4)
                
                with open(f"{args.output_dir}/{args.dataset}/entity_embeddings.json", 'w') as f:
                    json.dump(entity_embeddings, f, indent=4)
                    
        
    return np.array(embeddings), entity_info, entity_embeddings

def agglomerative_clustering(entities, embeddings, distance_threshold):
    print(f"Evaluating threshold: {distance_threshold}")
    clustering = AgglomerativeClustering(metric='cosine', linkage='average', distance_threshold=distance_threshold, n_clusters=None)
    clustering.fit(embeddings)
    
    clusters = {}
    for i in range(clustering.n_clusters_):
        cluster_indices = np.where(clustering.labels_ == i)[0]
        cluster_entities = [entities[idx] for idx in cluster_indices]
        clusters[f"Cluster_{i+1}"] = cluster_entities
        
    print(f"Clusters: {clusters}")
    
    return clusters


def find_optimal_threshold(entities, embeddings, min_threshold=0.1, max_threshold=1.0, num_thresholds=10):
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    best_score = -1.0
    best_threshold = None
    best_clusters = None

    for threshold in thresholds:
        print(f"Evaluating threshold: {threshold:.2f}")
        clusters = agglomerative_clustering(entities, embeddings, threshold)
        num_clusters = len(clusters)
        
        if num_clusters == 1:
            print("Skipping evaluation: Only one cluster formed.")
            continue
        elif num_clusters == len(entities):
            print("Skipping evaluation: Each entity is in its own cluster.")
            score = 0.0  # Assign a default score of 0 to clusterings with each entity in its own cluster
        else:
            labels = np.zeros(len(entities), dtype=int)
            for i, cluster_entities in enumerate(clusters.values()):
                indices = [entities.index(entity) for entity in cluster_entities]
                labels[indices] = i
            score = silhouette_score(embeddings, labels)
            print(f"Silhouette score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_clusters = clusters

    if best_threshold is None:
        print("No suitable clustering found. Returning the clustering with the highest threshold.")
        best_threshold = thresholds[-1]
        best_clusters = agglomerative_clustering(entities, embeddings, best_threshold)

    return best_threshold, best_clusters

def llm_refine_hierarchy(clusters, depth=0):
    print(f"Refining clusters at depth {depth}...")
    print(f"Current clusters: {clusters}")
    refined_clusters = {}
    
    for cluster_name, cluster_entities in clusters.items():
        example_entities = ["apple", "banana", "orange"]
        example_cluster_name = "Fruits"
        example_suggestion = "Potential sub-clusters:\nCitrus Fruits: orange\nNon-Citrus Fruits: apple, banana\n\nPotential cluster rename:\nFruits and Vegetables"

        prompt = f"Given the following cluster of entities:\n\nCluster: {cluster_name}\nEntities: {', '.join(cluster_entities)}\n\nPlease suggest any potential sub-clusters, further categorization, or a more descriptive name for this cluster to improve its semantic coherence. If no further sub-clusters, categorization, or renaming is needed, simply respond with 'No further changes suggested.'\n\nExample:\nCluster: {example_cluster_name}\nEntities: {', '.join(example_entities)}\nSuggestion: {example_suggestion}\n\nCluster: {cluster_name}\nEntities: {', '.join(cluster_entities)}\nSuggestion:"
        
        response = gpt_chat_return_response(model="gpt-3.5-turbo-0125", prompt=prompt)

        suggestion = response.choices[0].message.content.strip()
        if "No further changes suggested" not in suggestion:
            refined_clusters = update_cluster(refined_clusters, cluster_name, cluster_entities, suggestion)
            if isinstance(refined_clusters.get(cluster_name), dict):
                refined_clusters[cluster_name] = llm_refine_hierarchy(refined_clusters[cluster_name], depth+1)
        else:
            refined_clusters[cluster_name] = cluster_entities
    
    return refined_clusters

def update_cluster(clusters, cluster_name, entities, suggestion):
    sub_clusters = {}
    new_name = None
    
    suggestion_lines = suggestion.split("\n")
    for line in suggestion_lines:
        if ":" in line:
            sub_cluster_name, sub_cluster_entities = line.split(":", maxsplit=1)
            sub_cluster_entities = [entity.strip() for entity in sub_cluster_entities.split(",")]
            sub_clusters[sub_cluster_name.strip()] = sub_cluster_entities
        elif "Potential cluster rename:" in line:
            new_name = line.split("Potential cluster rename:")[1].strip()
    
    if sub_clusters:
        clusters[cluster_name] = sub_clusters
    elif new_name:
        clusters[new_name] = entities
        del clusters[cluster_name]
    else:
        clusters[cluster_name] = entities
    
    return clusters

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
    entity2label_ptah = f"{args.data_dir}/{args.dataset}/entity2label.txt"
    entity2label = {}
    with open(ori_info_path, 'r') as f:
        ori_info = json.load(f)
    with open(entity2label_ptah, 'r') as f:
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
    
    args = parser.parse_args()
    return args


def main():
    args = construct_args()
    
    entities = read_entities(f"{args.data_dir}/{args.dataset}/entities.dict")
    entity_info, entity_embeddings = create_entity_info_emb_dict(args, entities)
    
    entities_text, original_descriptions = [], []
    for entity in entities:
        entities_text.append(entity_info[entity]["text_label"])
        original_descriptions.append(entity_info[entity]["original_description"])
    
    # entities = ["apple", "banana", "orange", "dog", "cat", "elephant", "car", "bike", "train", "oak", "pine", "rose", "lily", "shark", "salmon", "eagle", "pigeon"]
    embeddings, entity_info, entity_embeddings = generate_embeddings(args, entity_info=entity_info, entity_embeddings=entity_embeddings, dim=args.dimensions)
    with open(f"{args.output_dir}/{args.dataset}/entity_info.json", 'w') as f:
        json.dump(entity_info, f, indent=4)
        
    with open(f"{args.output_dir}/{args.dataset}/entity_embeddings.json", 'w') as f:
        json.dump(entity_embeddings, f, indent=4)

    best_threshold, best_clusters = find_optimal_threshold(entities_text, embeddings, min_threshold=0.5, max_threshold=0.9, num_thresholds=20)

    seed_clusters = agglomerative_clustering(entities_text, embeddings, best_threshold)
    refined_clusters = llm_refine_hierarchy(seed_clusters)

    # print("Refined hierarchical clustering:")
    # print(refined_clusters)
    
    with open(f"{args.output_dir}/{args.dataset}/clusters.json", 'w') as f:
        json.dump(refined_clusters, f, indent=4)
    

if __name__ == "__main__":
    main()