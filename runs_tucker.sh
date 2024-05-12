CUDA_VISIBLE_DEVICES=1 python -u code/run_tucker_conve.py \
 --dataset FB15K-237 \
 --model TuckER \
 --distance_metric cosine \
 --hierarchy_type seed \
 --edim 1024 --rdim 1024 \