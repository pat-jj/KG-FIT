CUDA_VISIBLE_DEVICES=7 python -u code/run_tucker_conve.py \
 --dataset WN18RR \
 --model ConvE \
 --distance_metric cosine \
 --hierarchy_type seed \
 --edim 512 --rdim 512 \