import csv
from collections import defaultdict, deque
import argparse
import json
from tqdm import tqdm

# Function to read the entity index to entity ID mapping
def read_entity_mapping(filename):
    entity_to_index = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            index, entity_id = int(row[0]), row[1]
            entity_to_index[entity_id] = index
    return entity_to_index

# Function to read the knowledge graph triples
def read_knowledge_graph(filename, graph=None):
    if graph is None:
        graph = defaultdict(list)
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            e1, relation, e2 = row[0], row[1], row[2]
            graph[e1].append(e2)
            graph[e2].append(e1)  # Assuming the graph is undirected; if directed, remove this line
    return graph

# Function to read entities from a file
def read_entities(filename):
    entities = set()
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            e1, relation, e2 = row[0], row[1], row[2]
            entities.add(e1)
            entities.add(e2)
    return entities

# Function to perform BFS and find k-hop neighbors
def find_k_hop_neighbors(graph, entity_to_index, entities, k):
    k_hop_neighbors = {}
    for entity_id in tqdm(entities):
        if entity_id not in graph:
            continue
        visited = set()
        queue = deque([(entity_id, 0)])
        neighbors = set()

        while queue:
            current, depth = queue.popleft()
            if depth > k:
                break
            if current in visited:
                continue
            visited.add(current)

            if depth <= k:
                neighbors.add(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        k_hop_neighbors[entity_id] = [entity_to_index[neighbor] for neighbor in neighbors if neighbor in entity_to_index]
    
    return k_hop_neighbors

def construct_args():
    parser = argparse.ArgumentParser(description='Cluster entities using hierarchical clustering and refine the clusters using LLM.')
    parser.add_argument('--output_dir', type=str, default="/shared/pj20/lamake_data")
    parser.add_argument('--data_dir', type=str, default="/home/pj20/server-03/lamake/data")
    parser.add_argument('--hier', type=str, default="seed", choices=['seed', 'llm'], help='Type of hierarchy to construct. Default: seed.')
    parser.add_argument('--dataset', type=str, default="FB15K-237", help='Path to the dataset file containing the list of entities to cluster.')
    parser.add_argument('--k', type=int, default=2, help='Number of hops to consider for k-hop neighbors. Default: 2.')
    
    args = parser.parse_args()
    args.log_dir = f"{args.output_dir}/{args.dataset}/logs"
    
    return args

def main():
    args = construct_args()
    # Example usage
    entity_mapping_file = f'{args.data_dir}/{args.dataset}/entities.dict'
    kg_file_train = f'{args.data_dir}/{args.dataset}/train.txt'
    kg_file_valid = f'{args.data_dir}/{args.dataset}/valid.txt'
    original_entity_info_file = f"{args.output_dir}/{args.dataset}/entity_info_{args.hier}_hier.json"
    k = args.k
    
    print(f"Reading entity mapping from {entity_mapping_file} ...")
    entity_to_index = read_entity_mapping(entity_mapping_file)
    print(f"Reading knowledge graph from train set {kg_file_train} ...")
    graph = read_knowledge_graph(kg_file_train)
    # graph = read_knowledge_graph(kg_file_valid, graph)
    
    print(f"Reading test entities from {kg_file_valid} ...")
    test_entities = read_entities(kg_file_valid)
    
    print(f"Finding {k}-hop neighbors for test entities ...")
    k_hop_neighbors = find_k_hop_neighbors(graph, entity_to_index, test_entities, k)
    
    # compute the average number of neighbors
    num_neighbors = 0
    for entity_id in k_hop_neighbors:
        num_neighbors += len(k_hop_neighbors[entity_id])
    print(f"Average number of neighbors: {num_neighbors / len(k_hop_neighbors)}")
    
    with open(original_entity_info_file, 'r') as f:
        entity_info = json.load(f)
        
    for entity_id in entity_info:
        if entity_id in k_hop_neighbors:
            entity_info[entity_id]['k_hop_neighbors'] = list(set(k_hop_neighbors[entity_id]))
        else:
            entity_info[entity_id]['k_hop_neighbors'] = []
    
    with open(original_entity_info_file, 'w') as f:
        json.dump(entity_info, f, indent=4)

if __name__ == '__main__':
    main()
