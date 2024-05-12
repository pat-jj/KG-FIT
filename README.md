# Knowledge Graph Fine-Tuning with Open-World Entity Knowledge

### Please use the scripts runs_xxx.sh to run the experiments for all the models. For example:
```bash
bash runs_rotate.sh
bash runs_tucker.sh
```

### We provide several variants of KG-FIT framework under the ```code``` folder:

#### ```model_common.py```:
- **KG-FIT with KGE base models:** all the models other than TuckER and ConvE. 

- **Text and Hierarchical Constraints:** on negative batches.

- **Text Embedding within Entity Embedding:** Frozen

#### ```model_flex.py```:
- **KG-FIT with KGE base models:** all the models other than TuckER and ConvE. 

- **Text and Hierarchical Constraints:** on negative batches.

- **Text Embedding within Entity Embedding:** On Fire

#### ```model_p_anc.py```:
- **KG-FIT with KGE base models:** all the models other than TuckER and ConvE. 

- **Text and Hierarchical Constraints:** on both positive and negative batches.

- **Text Embedding within Entity Embedding:** Frozen

#### ```model_tucker_conve.py```:
- **KG-FIT with KGE base models:** KG-FIT-TuckER and KG-FIT-ConvE. 

- **Text and Hierarchical Constraints:** on both positive and negative batches.

- **Text Embedding within Entity Embedding:** Frozen