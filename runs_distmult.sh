# CUDA_VISIBLE_DEVICES=5 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset PrimeKG \
#  --model DistMult \
#  --distance_metric cosine \
#  --hierarchy_type seed \
#  --zeta_3 2.0 \
#  -n 1024 -b 512 -d 1024 \
#  -g 200.0 -a 1.0 -adv \
#  -lr 0.002 --max_steps 300000 \
#  --test_batch_size 8 \
#  -r 0.000005
#  --inter_cluster_constraint false 


# CUDA_VISIBLE_DEVICES=5 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset WN18RR \
#  --model DistMult \
#  --distance_metric cosine \
#  --hierarchy_type seed \
#  --zeta_3 2.0 \
#  -n 1024 -b 256 -d 1024 \
#  -g 200.0 -a 1.0 -adv \
#  -lr 0.002 --max_steps 160000 \
#  --test_batch_size 8 \
#  --inter_cluster_constraint false 

#  CUDA_VISIBLE_DEVICES=2 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model DistMult \
#  --distance_metric cosine \
#   --hierarchy_type llm \
#  --zeta_3 9.0 \
#  -n 256 -b 256 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 1000000 \
#  --test_batch_size 8 \
#  --text_dist_constraint true \
#  --hier_dist_constraint false \
#  --intra_cluster_constraint true \
#  --inter_cluster_constraint true \
#  --valid_steps 10000 \

#  CUDA_VISIBLE_DEVICES=7 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model DistMult \
#  --distance_metric cosine \
#   --hierarchy_type seed \
#  --zeta_3 2.0 \
#  -n 256 -b 128 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 1000000 \
#  --test_batch_size 8 

#  CUDA_VISIBLE_DEVICES=1 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset FB15K-237 \
#  --model DistMult \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 3.5 \
#  -n 256 -b 256 -d 1024 \
#  -g 200.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 400000 \
#  --test_batch_size 8 \
#  --text_dist_constraint true \
#  --hier_dist_constraint false \
#  --intra_cluster_constraint true \
#  --inter_cluster_constraint true \
#  --valid_steps 20000 \