CUDA_VISIBLE_DEVICES=1 python -u code/run_tucker_conve.py \
 --dataset YAGO3-10 \
 --model ConvE \
 --distance_metric cosine \
 --hierarchy_type seed \
 --edim 512 --rdim 512 \