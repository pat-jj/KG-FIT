import json



def check_entities(root, initial=False):
    entities = []

    def recursion_check_entities(root):
        nonlocal entities
        if not initial:

            if root['children'] == {}:
                entities.append(root['name'])
                return
            
            for child_key, child_value in root['children'].items():
                recursion_check_entities(child_value)
        else:
            if isinstance(root, list):
                # if root == []:
                    # print('!!!')
                entities.extend(root)
                return
            for child_key, child_value in root.items():
                recursion_check_entities(child_value)
        # print(root)

    
    recursion_check_entities(root)

    return set(entities), list(entities)



if __name__ == '__main__':
    with open('seed_hierarchy_fb.json') as f:
        initial_hierarchy = json.load(f)
    e1, e1_l = check_entities(initial_hierarchy, initial=True)

    with open('llm_refined_hier_gpt-4-turbo-2024-04-09_777.json') as f:
    # with open('step1_res.json') as f:
        llm_hierarchy = json.load(f)
    # e2 = check_entities(llm_hierarchy['Cluster_top'], initial=True)
    # e2, e2_l = check_entities(llm_hierarchy['Cluster_llm_root'])
    e2, e2_l = check_entities(llm_hierarchy['Cluster_llm_root'], initial=True)

    # print(e1-e2)
    # print(len(e1))
    # print(len(e2))
    # print(len(e1_l))
    # print(len(e2_l))
    print(len(e1-e2))
    print(len(e2-e1))
    m = list(e1-e2)
    with open('missed_entities.json', 'w') as f:
        json.dump(m, f)

    # counter = {}
    # for e in e1_l:
    #     if e not in counter:
    #         counter[e] = 1
    #     else:
    #         print(e)

