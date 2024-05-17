import json
import os
from tqdm import tqdm
import copy
import re
import argparse
import traceback
import time
from openai import OpenAI


####################
# Prompt Templates
####################

NAME_CLUSTER_PROMPT = '''Given the following entities from a cluster: {entities}.
Provide a name for these entities which can describe them uniformly as a cluster.
Name: '''



SPLIT_CLUSTER_PROMPT = '''Given entities from the cluster '{cluster_name}'.

Analyze the entities and determine if they can be grouped into distinct and meaningful sub-clusters based on their characteristics, themes, or genres.
If sub-clusters can be formed, provide a clear and concise name for each sub-cluster that represents the common attribute of its entities.
Each sub-cluster should be given a new name that uniformly describes its entities. There needs to be differentiation in the names of different clusters.
The number of sub-clusters is not fixed and should be controlled between 1 and 5.
If the entities are already well-grouped and don't require further sub-clustering, simply provide the original cluster.


Provide the output in the following JSON format:
```json
{{
    "Sub-cluster Name": ["Entity 1", "Entity 2", ...],
    "Sub-cluster Name": ["Entity 3", "Entity 4", ...],
    "Sub-cluster Name": ["Entity 4", "Entity 5", ...],
    ...
}}
```

Example:
Cluster: {split_example_cluster_name}
Entities: {split_example_entities}
Output: {split_example_subclusters}

Cluster: {cluster_name}
Entities: {entities}
Output: '''



UPDATE_TWO_CLUSTERS_PROMPT = '''Given the cluster A '{cluster_name_1}': [{entities_1}];
and the cluster B '{cluster_name_2}': [{entities_2}].

Analyze two clusters and their entities and determine the update mode:
Update Mode 1 - Create New Cluster C: these two clusters cannot be merged, and no cluster belongs to any other.
Update Mode 2 - Merge Cluster A and B: these two clusters can be merged. The name of two clusters should be similar and entities from two clusters should be similar.
Update Mode 3 - Cluster A Covers Cluster B: cluster B belongs to cluster A. cluster B is a subcluster of cluster A. The name of cluster A should uniformly describe the entities from cluster A and the name of cluster B.
Update Mode 4 - Cluster B Covers Cluster A: cluster A belongs to cluster B. cluster A is a subcluster of cluster B. The name of cluster B should uniformly describe the entities from cluster B and the name of cluster A.

You need to select a update mode based on two clusters.
If you select mode 1, you should also suggest a name of new cluster. The new cluster name should uniformly describe two clusters.
If you select mode 2, you should suggest a name of merged cluster. The new name should be similar to cluster A and B.

Example:
Cluster A 'Thermal Insulators': [cork, fiberglass, foam];
Cluster B 'Electrical Conductors': [copper, aluminum, gold];
Select Mode 1.

Cluster A 'Sedans': [Toyota Camry, Honda Accord, Ford Fusion];
Cluster B 'SUVs': [Honda CR-V, Toyota RAV4, Ford Escape];
Select Mode 2.

Cluster A 'Feline Species': [lions, tigers, cheetahs];
Cluster B 'House Cats': [Siamese, Persian, Maine Coon];
Select Mode 3.

Cluster A 'Leafy Vegetables': [lettuce, spinach, kale];
Cluster B 'Root Vegetables': [carrots, potatoes, beets];
Select Mode 4.


Provide the output in the following JSON format:
```json
{{
    "update_mode": 1 or 2 or 3 or 4,
    "name": "merged cluster name or new cluster name"
}}
```

Output: '''




####################
# LLM Utilities
####################

with open('../openai_api.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)


# gpt-4-turbo-2024-04-09
def get_llm_response(prompt, model="gpt-3.5-turbo-0125", seed=44):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        # max_tokens=200,
        temperature=0,
        seed=seed
    ).choices[0].message.content
    return response

# ########## Trialmind ##########
# from trialmind.llm_utils.openai import call_openai, call_azure_openai

# def get_llm_response(prompt, model="gpt-3.5-turbo-0125", seed=44):
#     messages = [{
#             "role": "user",
#             "content": prompt
#     }]
#     response = call_azure_openai(
#         llm='gpt-35',
#         messages=messages,
#         temperature=0,
#         stream=False
#     )
#     response = response.choices[0].message.content
#     return response
# ##############################


def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_json = re.search(pattern, text)
    if matched_json:
        extracted_json = matched_json.group(1)
        return json.loads(extracted_json)
    else:
        # backup plan
        pattern = r"\{.*?\}"
        matched_json = re.search(pattern, text, re.DOTALL)
        if matched_json:
            extracted_json = matched_json.group()
            return json.loads(extracted_json)
        else:
            raise ValueError('No JSON structure found.')


def llm_name_cluster(entities, model, seed):
    entities_str = ', '.join(entities)
    name = get_llm_response(NAME_CLUSTER_PROMPT.format(entities=entities_str), model=model, seed=seed)
    return name


split_example_cluster_name = "Movies"
split_example_entities = ', '.join(["The Godfather", "The Shawshank Redemption", "The Dark Knight", "Forrest Gump", "Inception", "The Matrix"])
split_example_subclusters = json.dumps({
    "Drama": ["The Godfather", "The Shawshank Redemption", "Forrest Gump"],
    "Action": ["The Dark Knight"],
    "Science Fiction": ["Inception", "The Matrix"]
})


def llm_split_cluster(cluster_name, entities, model, seed):
    entities_str = ', '.join(entities)
    while True: 
        try:
            llm_output = get_llm_response(SPLIT_CLUSTER_PROMPT.format(
                cluster_name=cluster_name,
                entities=entities_str,
                split_example_cluster_name=split_example_cluster_name,
                split_example_entities=split_example_entities,
                split_example_subclusters=split_example_subclusters
            ), model=model, seed=seed)
            clusters = extract_json_from_llm_output(llm_output)
            break
        except Exception as e:
            # print('Call LLM Error.')
            # print(e)
            # continue
            print('Split failure.')
            # we assume it is hard for llms to split, so return the original cluster
            return {cluster_name: entities}
    return clusters


def llm_update_two_clusters(cluster_name_1, entities_1, cluster_name_2, entities_2, model, seed):
    entities_1 = [entity for entity in entities_1 if entity is not None][:20]
    entities_2 = [entity for entity in entities_2 if entity is not None][:20]
    entities_str_1 = ', '.join(entities_1)
    entities_str_2 = ', '.join(entities_2)
    while True:
        try:
            llm_output = get_llm_response(UPDATE_TWO_CLUSTERS_PROMPT.format(
                cluster_name_1=cluster_name_1,
                entities_1=entities_str_1,
                cluster_name_2=cluster_name_2,
                entities_2=entities_str_2
            ), model=model, seed=seed)
            structured_output = extract_json_from_llm_output(llm_output)

            update_mode = structured_output['update_mode']
            name = structured_output['name']
            break
        except Exception as e:
            print('Call LLM Error.')
            print(e)
            continue
    return update_mode, name


####################
# Hyperparameters
####################

MIN_ENTITIES_IN_LEAF = 10

MAX_ENTITIES_IN_LEAF = 30


####################
# Algorithm
####################


# construct hierarchy from leaf to bottom
def construct_bot_hierarchy(initial_hierarchy, model, seed):
    current_cluster_id = 0

    def recursion_construct_bot_hierarchy(root):
        nonlocal current_cluster_id, model, seed

        if isinstance(root, list):
            if len(root) < MIN_ENTITIES_IN_LEAF or MAX_ENTITIES_IN_LEAF < len(root) :
                return
            
            cluster_entities = root
            cluster_name = llm_name_cluster(cluster_entities, model, seed)

            splitted_clusters = llm_split_cluster(cluster_name, cluster_entities, model, seed)
            if len(splitted_clusters) == 1:
                return

            for name, entities in splitted_clusters.items():
                current_full_cluster_id = 'Cluster_llm_bot_' + str(current_cluster_id)
                root = {}
                root[current_full_cluster_id] = entities
                current_cluster_id += 1
                recursion_construct_bot_hierarchy(root[current_full_cluster_id])
        else:
            for key, subcluster in root.items():
                recursion_construct_bot_hierarchy(subcluster)


    recursion_construct_bot_hierarchy(initial_hierarchy)
    refined_hierarchy = initial_hierarchy

    return refined_hierarchy


# construct hierarchy from bottom to top
def construct_top_hierarchy(initial_hierarchy, model, seed):

    def recursion_construct_top_hierarchy(root):
        nonlocal model, seed

        # leaf - entity list
        if isinstance(root, list) or root is None:
            updated_hierarchy = {
                'name': llm_name_cluster(root, model, seed),
                'children': {
                    entity: {
                        'name': entity,
                        'children': {}
                    } for entity in root}
            }
            return updated_hierarchy

        
        # binary tree
        left_node_key = list(root.keys())[0]
        left_node_value = list(root.values())[0]
        right_node_key = list(root.keys())[1]
        right_node_value = list(root.values())[1]

        # postorder traversal - update children first
        left_node_value = recursion_construct_top_hierarchy(left_node_value)
        right_node_value = recursion_construct_top_hierarchy(right_node_value)


        left_node_children = [v['name'] for v in left_node_value['children'].values()]
        right_node_children = [v['name'] for v in right_node_value['children'].values()]


        while True:
            update_mode, name = llm_update_two_clusters(left_node_value['name'], left_node_children, right_node_value['name'], right_node_children, model, seed)
            # left cluster covers right cluster
            if update_mode == 1:
                left_node_value['children'][right_node_key] = right_node_value
                updated_hierarchy = {
                    'name': left_node_value['name'],
                    'children': left_node_value['children']
                }
                print('Update mode 1')
            # right cluster covers left cluster
            elif update_mode == 2:
                right_node_value['children'][left_node_key] = left_node_value
                updated_hierarchy = {
                    'name': right_node_value['name'],
                    'children': right_node_value['children']
                }
                print('Update mode 2')
            # merge left and right cluster to one cluster
            elif update_mode == 3:
                merged_cluster_name = name
                merged_node_value = {**left_node_value['children'], **right_node_value['children']}
                updated_hierarchy = {
                    'name': merged_cluster_name,
                    'children': merged_node_value
                }
                print('Update mode 3')
            # cannot cover or merge, create a new cluster
            elif update_mode == 4: 
                new_cluster_name = name
                updated_hierarchy = {
                    'name': new_cluster_name,
                    'children': {
                        left_node_key: left_node_value,
                        right_node_key: right_node_value
                    }
                }
                print('Update mode 4')
            else:
                continue
            break
            
        return updated_hierarchy

    refined_hierarchy = recursion_construct_top_hierarchy(list(initial_hierarchy.values())[0])
    return {'Cluster_llm_root': refined_hierarchy}


def reformat_hierarchy(initial_hierarchy):
    current_id = 0
    
    def recursion_reformat_hierarchy(root):
        # if root == {}:
        #     return None
        entities = []
        is_leaf = True

        for key, child in root['children'].items():
            if child['children'] != {}:
                is_leaf = False
                break
            entities.append(child['name'])

        if is_leaf:
            return entities
        else:
            for key, child in root['children'].items():
                root['children'][key] = recursion_reformat_hierarchy(child)

            # deal with entity of empty list
            empty_list = []
            has_empty_list = False
            new_root = {}
            for key, child in root['children'].items():
                if child == []:
                    has_empty_list = True
                    empty_list.append(key)
                else:
                    new_root[key] = child
            if has_empty_list:
                nonlocal current_id
                new_root[f'Cluster_entity_list_{current_id}'] = empty_list
                current_id += 1

            # root['children'] = new_root
            return new_root
    
    refined_hierarchy = recursion_reformat_hierarchy(list(initial_hierarchy.values())[0])

    return {'Cluster_llm_root': refined_hierarchy}


# main
def llm_refine_hierarchical_knowledge(initial_hierarchy, dataset, model, seed):
    start_time = time.time()

    # 1. construct hierarchical knowledge below the clusters (bottom hierarchical knowledge)
    print('Constructing bottom hierarchy...')
    if os.path.exists(f'./outputs/{dataset}/step1_res.json'):
        with open(f'./outputs/{dataset}/step1_res.json') as f:
            tmp_hierarchy = json.load(f)
    else:
        tmp_hierarchy = construct_bot_hierarchy(initial_hierarchy, model, seed)
        with open(f'./outputs/{dataset}/step1_res.json', 'w') as f:
            json.dump(tmp_hierarchy, f)

    phase1_end_time = time.time()
    phase1_time = (phase1_end_time - start_time) / 3600
    print('Phase 1 time:', phase1_time)

    # 2. construct hierarchical knowledge above the clusters (top hierarchical knowledge)
    print('Constructing top hierarchy...')
    if os.path.exists(f'./outputs/{dataset}/step2_res.json'):
        with open(f'./outputs/{dataset}/step2_res.json') as f:
            refined_hierarchy = json.load(f)
    else:
        refined_hierarchy = construct_top_hierarchy(tmp_hierarchy, model, seed)
        with open(f'./outputs/{dataset}/step2_res.json', 'w') as f:
            json.dump(refined_hierarchy, f)

    phase2_end_time = time.time()
    phase2_time = (phase2_end_time - phase1_end_time) / 3600
    print('Phase 2 time:', phase2_time)

    # 3. reformat
    print('Reformatting...')
    refined_hierarchy = reformat_hierarchy(refined_hierarchy)

    return refined_hierarchy




if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--seed", type=int, required=True, help="Seed of Model")

    args = parser.parse_args()

    with open(f'/shared/pj20/lamake_data/{args.dataset}/seed_hierarchy.json') as f:
        initial_hierarchy = json.load(f)
    refined_hierarchy = llm_refine_hierarchical_knowledge(initial_hierarchy, args.dataset, args.model, args.seed)
    with open(f'/shared/pj20/lamake_data/{args.dataset}/llm_hierarchy.json', 'w') as f:
        json.dump(refined_hierarchy, f)
