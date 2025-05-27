# KG-FIT

This repository contains the code for the paper "KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge" (NeurIPS 2024). [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f606d45ae7b991988b6eea2af38b7057-Abstract-Conference.html)

![alt text](/images/abstract.png "Overview of KG-FIT")

## Data Preparing & Precompute:

To enable precompute, you need to put a file named "openai_api.key" (with your OpenAI API key in there) under ```code/precompute```, then run the following command with a specified dataset (FB16K-237 in this case):
```bash
cd code/precompute
python cluster.py --dataset FB15K-237 --output_dir ../../processed_data  # precomputation for seed hierarchy
cd llm_refine
python llm_refine.py --dataset FB15K-237  --model gpt-4o-2024-05-13 # LLM-Guided Hierarchy Refinement (LHR)
cd ..
python cluster.py --dataset FB15K-237  --output_dir ../../processed_data # precomputation for llm hierarchy
```

where the first call of ```cluster.py``` is used to build seed hierarchy; ```llm_refine.py``` is used to refine the seed hierarchy with LLM; The second call of ```cluster.py``` is used to build the final hierarchy with LLM.
<!-- (```rerank_label.py``` is used to create the k-hop training set entities for each entity in the dataset, for graph-based re-ranking purpose. ) -->


## KG-FIT Training & Evaluation:

### Use the scripts runs_xxx.sh to run the experiments for all the models. For example:

```bash
bash runs_rotate.sh
bash runs_tucker.sh
```

### We provide several variants of KG-FIT model under the ```code``` folder:

| File                      | KG-FIT with KGE base models                     | Text and Hierarchical Constraints   | Text Embedding within Entity Embedding |
|---------------------------|--------------------------------------------------|-------------------------------------|----------------------------------------|
| `model_common.py`         | All models except TuckER and ConvE              | On negative batches                 | Frozen                                 |
| `model_flex.py`           | All models except TuckER and ConvE              | On negative batches                 | On Fire                                |
| `model_p_anc.py`          | All models except TuckER and ConvE              | On both positive and negative batches | Frozen                               |
| `model_tucker_conve.py`   | KG-FIT-TuckER and KG-FIT-ConvE                  | On both positive and negative batches | Frozen                               |


## Cite KG-FIT
```bibtex
@article{jiang2024kg,
  title={Kg-fit: Knowledge graph fine-tuning upon open-world knowledge},
  author={Jiang, Pengcheng and Cao, Lang and Xiao, Cao Danica and Bhatia, Parminder and Sun, Jimeng and Han, Jiawei},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={136220--136258},
  year={2024}
}
```

Thank you for your interest in our work!
