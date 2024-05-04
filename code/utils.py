import json

def read_hierarchy(args):
    """
    Read the hierarchy of the dataset from the json file.
    """
    file_path = f"{args.process_path}/{args.dataset}/{args.hierarchy_type}_hierarchy.json"
    hierarchy = json.load(open(file_path, "r"))
    
    return hierarchy


def read_entity_initial_embedding(args):
    """
    Read the initial entity embeddings from the json file.
    """
    file_path = f"{args.process_path}/{args.dataset}/entity_init_embeddings.json"
    entity_initial_embedding = json.load(open(file_path, "r"))
    
    return entity_initial_embedding