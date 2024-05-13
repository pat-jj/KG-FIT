import json


tree_path = 'other/llm_refined_hier_gpt-4-turbo-2024-04-09_777.json'


with open(tree_path) as f:
    tree = json.load(f)



cluster_num = 0 # number of cluster

node_num = 1 # number of node

entity_num = [] # number of entity in a cluster

depth_list = [] # depth

branch_num = [] # branch factor / number of branch of a node

# Balanced Status


def traverse_hierarchy_and_statistic(root, depth):
    global cluster_num, node_num

    if isinstance(root, list):
        entity_num.append(len(root))
        depth_list.append(depth)
        cluster_num += 1
        return

    branch_num.append(len(root.keys()))
    node_num += len(root.keys())

    for key, child in root.items():
        root[key] = traverse_hierarchy_and_statistic(child, depth + 1)

    return


traverse_hierarchy_and_statistic(tree, 0)

print('Number of cluster:', cluster_num)
print('Number of node:', node_num)
print(f'Number of entity in a cluster: Max({max(entity_num)}), Min({min(entity_num)}), Avg({sum(entity_num) / len(entity_num)})')
print(f'Depth: Max({max(depth_list)}), Min({min(depth_list)}), Avg({sum(depth_list) / len(depth_list)})')
print(f'Number of branch of a node: Max({max(branch_num)}), Min({min(branch_num)}), Avg({sum(branch_num) / len(branch_num)})')