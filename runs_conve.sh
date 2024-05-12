CUDA_VISIBLE_DEVICES=0 python -u code/run_tucker_conve.py \
 --dataset FB15K-237 \
 --model ConvE \
 --distance_metric cosine \
 --hierarchy_type seed \
 --edim 512 --rdim 512 \