CUDA_VISIBLE_DEVICES=4 python -u code/run_tucker_conve.py \
 --dataset PrimeKG \
 --model TuckER \
 --distance_metric cosine \
 --hierarchy_type llm \
 --edim 1024 --rdim 1024 \