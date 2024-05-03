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
