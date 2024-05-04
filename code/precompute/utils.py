import numpy as np


def find_leaves(d, leaf_keys=None, leaf_values=None):
    if leaf_keys is None:
        leaf_keys = []
    if leaf_values is None:
        leaf_values = []
    for key, value in d.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse into it
            find_leaves(value, leaf_keys, leaf_values)
        else:  # If the value is not a dictionary, then it's a leaf node
            leaf_keys.append(key)
            leaf_values.append(value)
    return leaf_keys, leaf_values

def map_child_to_parent(d, parent_map=None, current_parent=None):
    if parent_map is None:
        parent_map = {}
    for key, value in d.items():
        if current_parent is not None:  # Map current key to its parent
            parent_map[key] = current_parent
        if isinstance(value, dict):  # Recursively process the dictionary
            map_child_to_parent(value, parent_map, key)
    return parent_map

def node2parentpath(d, source_cluster):
    parent_path = []
    parent_distances = []
    child_parent = map_child_to_parent(d)
    current_parent = child_parent[source_cluster]
    while current_parent in child_parent.keys():
        parent_path.append(current_parent)
        # parent_distances.append(distance_between_keys(d, current_parent, source_cluster))
        current_parent = child_parent[current_parent]
        
    parent_path.append(current_parent)
    
    return parent_path, parent_distances

def rename_clusters_to_ints(original_dict, start_index=0, key_map=None):
    """
    Recursively renames keys of the nested dictionary to integers, incrementing from a given start index.
    Also tracks the mapping from original keys to new keys.
    """
    if key_map is None:
        key_map = {}

    new_dict = {}
    index = start_index

    for key, value in original_dict.items():
        key_map[key] = index
        if isinstance(value, dict):
            new_dict[index], index, key_map, key_map_inv = rename_clusters_to_ints(value, index + 1, key_map)
        else:
            new_dict[index] = value
            index += 1

    key_map_inv = {v: k for k, v in key_map.items()}
    return new_dict, index, key_map, key_map_inv

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy(key): convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(element) for element in obj)
    return obj


def label_(d, entity2clusterid, clusterid2count, leaf_keys=None, leaf_values=None):
    if leaf_keys is None:
        leaf_keys = []
    if leaf_values is None:
        leaf_values = []
    for key, value in d.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse into it
            label_(value, entity2clusterid, clusterid2count, leaf_keys, leaf_values)
        else:  # If the value is not a dictionary, then it's a leaf node
            cluster_id = entity2clusterid[value]
            d[key] = [cluster_id, clusterid2count[cluster_id]]
            clusterid2count[entity2clusterid[value]] += 1
    return d


def refine_1(d, clusters_, leaf_keys=None, leaf_values=None):
    if leaf_keys is None:
        leaf_keys = []
    if leaf_values is None:
        leaf_values = []
    
    keys_to_delete = []  # List to hold keys of items to be deleted
    items_to_update = {}  # Dictionary to hold items to be updated

    for key, value in list(d.items()):  # Convert dict_items to a list to safely iterate
        if isinstance(value, dict):  # If the value is another dictionary, recurse into it
            refine_1(value, clusters_, leaf_keys, leaf_values)
        else:
            if value[1] > 0:
                keys_to_delete.append(key)
            else:
                items_to_update[key] = clusters_[value[0]]

    # Now, delete keys marked for deletion
    for key in keys_to_delete:
        del d[key]

    # Update the dictionary with new values
    for key, new_value in items_to_update.items():
        d[key] = new_value

    return d


def refine_2(d):
    # Recursive function to process and refine each dictionary
    def process_dict(sub_dict):
        for key in list(sub_dict.keys()):  # Iterate over a copy of the keys
            value = sub_dict[key]
            if isinstance(value, dict):
                if value:  # Check if the dictionary is not empty
                    result = process_dict(value)
                    # If the result is a single entry with a list, replace the current dict
                    if len(result) == 1 and isinstance(list(result.values())[0], list):
                        sub_dict[key] = list(result.values())[0]
                    else:
                        sub_dict[key] = result
                else:
                    del sub_dict[key]  # Remove empty dictionaries
        return sub_dict

    # Copy the original dictionary to avoid modification issues
    refined_dict = process_dict(d.copy())
    return refined_dict


def refine_3(d):
    # Recursive function to process and refine each dictionary
    def process_dict(sub_dict):
        new_dict = {}  # To accumulate refined results
        for key, value in list(sub_dict.items()):
            if isinstance(value, dict):
                processed = process_dict(value)  # Recursively process
                if processed:  # Only add non-empty results
                    new_dict[key] = processed
            else:  # Keep non-dict items as they are
                new_dict[key] = value
        return new_dict

    # Start the processing with the original dictionary
    refined_dict = process_dict(d)
    return refined_dict

def refine_4(d):
    # Recursive function to process and refine each dictionary
    def process_dict(sub_dict):
        new_dict = {}  # To accumulate refined results
        for key, value in list(sub_dict.items()):
            if isinstance(value, dict):
                processed = process_dict(value)  # Recursively process
                if processed:  # Only add non-empty results
                    if isinstance(processed, dict) and len(processed) == 1:
                        new_dict[key] = list(processed.values())[0]
                    else:
                        new_dict[key] = processed
            else:  # Keep non-dict items as they are
                new_dict[key] = value
        return new_dict

    # Start the processing with the original dictionary
    refined_dict = process_dict(d)
    return refined_dict


def find_lca(root, node1, node2):
    if root is None:
        return None
    
    if isinstance(root, list):
        if node1 in root or node2 in root:
            return root
    
    if root == node1 or root == node2:
        return root

    lca_list = []
    if isinstance(root, dict):
        for child in root.values():
            lca = find_lca(child, node1, node2)
            if lca is not None:
                lca_list.append(lca)
            if len(lca_list) > 1:
                return root

    return lca_list[0] if lca_list else None

def find_distance_from_root_to_node(root, node, distance=0):
    if root is None:
        return -1

    if isinstance(root, list):
        if node in root:
            return distance

    if root == node:
        return distance

    if isinstance(root, dict):
        for child in root.values():
            dist = find_distance_from_root_to_node(child, node, distance + 1)
            if dist != -1:
                return dist

    return -1

def distance_between_nodes(root, node1, node2):
    lca = find_lca(root, node1, node2)
    if lca is None:
        return -1

    distance1 = find_distance_from_root_to_node(lca, node1, 0)
    distance2 = find_distance_from_root_to_node(lca, node2, 0)
    
    return distance1 + distance2 if distance1 != -1 and distance2 != -1 else -1


def find_lca_key(root, key1, key2):
    if root is None:
        return None

    # If the current root (or dict) contains the key directly, we check its keys
    if key1 in root or key2 in root:
        return root  # Found one of the keys at the current level, return this root

    lca_list = []
    if isinstance(root, dict):
        for key, child in root.items():
            if key == key1 or key == key2:
                lca_list.append(key)
            lca = find_lca_key(child, key1, key2)
            if lca is not None:
                lca_list.append(lca)
            if len(lca_list) > 1:
                return root  # Both keys found in different subtrees

    return lca_list[0] if lca_list else None

def find_distance_from_root_to_key(root, key, distance=0):
    if root is None:
        return -1

    # Check if the key is the current root's direct key
    if key in root:
        return distance

    if isinstance(root, dict):
        for child_key, child in root.items():
            if child_key == key:
                return distance + 1
            dist = find_distance_from_root_to_key(child, key, distance + 1)
            if dist != -1:
                return dist

    return -1

def distance_between_keys(root, key1, key2):
    lca = find_lca_key(root, key1, key2)
    if lca is None:
        return -1

    distance1 = find_distance_from_root_to_key(lca, key1, 0)
    distance2 = find_distance_from_root_to_key(lca, key2, 0)
    
    return distance1 + distance2 if distance1 != -1 and distance2 != -1 else -1


def compute_tree_depth(root):
    if root is None:
        return 0
    
    if isinstance(root, list) or isinstance(root, str):
        return 1  # Leaf nodes contribute a depth of 1
    
    if isinstance(root, dict):
        max_depth = 0
        for child in root.values():
            child_depth = compute_tree_depth(child)
            if child_depth > max_depth:
                max_depth = child_depth
        return 1 + max_depth  # Add 1 for the depth from the current node to its children

    return 0


def compute_clusters_embeddings(clusters, entity_embeddings, label2entity):
    # Initialize dictionary to store embeddings
    cluster_embeddings = {}
    
    def compute_cluster_embedding(cluster, embeddings_dict, cluster_id):
        if isinstance(cluster, list):
            # Base case: cluster is a list of entities
            embeddings = [entity_embeddings.get(label2entity.get(entity)) for entity in cluster]
            cluster_embedding = np.mean(embeddings, axis=0)
            # print(f"Computed embedding for {cluster_id} with entities: {cluster}")
            # print(f"Cluster embedding: {cluster_embedding}")
        elif isinstance(cluster, dict):
            # Recursive case: cluster has sub-clusters
            sub_embeddings = []
            for sub_cluster_id, sub_cluster in cluster.items():
                sub_embedding = compute_cluster_embedding(sub_cluster, embeddings_dict, sub_cluster_id)
                sub_embeddings.append(sub_embedding)
            cluster_embedding = np.mean(sub_embeddings, axis=0)
            # print(f"Computed embedding for {cluster_id} with sub-clusters: {list(cluster.keys())}")
            # print(f"Parent cluster embeddings: {cluster_embedding}")

        embeddings_dict[cluster_id] = cluster_embedding
        return cluster_embedding
    
    for cluster_id, cluster_data in clusters.items():
        compute_cluster_embedding(cluster_data, cluster_embeddings, cluster_id)
        
    return cluster_embeddings
