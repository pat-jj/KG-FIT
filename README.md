# Knowledge Graph Fine-Tuning with Open-World Entity Knowledge

![alt text](/images/abstract.png "Optional title")

## Data Preparing & Precompute:

To enalbe precompute, you need to put a file named "openai_api.key" under ```code/precompute```, then run the following command with a specified dataset (FB16K-237 in this case):
```bash
cd code/precompute
python cluster.py --dataset FB15k-237 
cd llm_refine
python llm_refine.py
```

where ```cluster.py``` is used to build seed hierarchy and ```llm_refine.py``` is used to refine the seed hierarchy with LLM.

## KG-FIT Training & Evaluation:

### Please use the scripts runs_xxx.sh to run the experiments for all the models. For example:
```bash
bash runs_rotate.sh
bash runs_tucker.sh
```

### We provide several variants of KG-FIT framework under the ```code``` folder:

| File                      | KG-FIT with KGE base models                     | Text and Hierarchical Constraints   | Text Embedding within Entity Embedding |
|---------------------------|--------------------------------------------------|-------------------------------------|----------------------------------------|
| `model_common.py`         | All models except TuckER and ConvE              | On negative batches                 | Frozen                                 |
| `model_flex.py`           | All models except TuckER and ConvE              | On negative batches                 | On Fire                                |
| `model_p_anc.py`          | All models except TuckER and ConvE              | On both positive and negative batches | Frozen                               |
| `model_tucker_conve.py`   | KG-FIT-TuckER and KG-FIT-ConvE                  | On both positive and negative batches | Frozen                               |

