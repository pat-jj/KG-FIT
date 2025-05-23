# CUDA_VISIBLE_DEVICES=5 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset PrimeKG \
#  --model TransE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 1.8 \
#  -n 256 -b 256 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.00001 --max_steps 400000 \
#  --test_batch_size 8 \
 

#  CUDA_VISIBLE_DEVICES=7 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 2.0 \
#  -n 256 -b 256 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 800000 \
#  --test_batch_size 8 \
#  --text_dist_constraint true \
#  --hier_dist_constraint false \
#  --intra_cluster_constraint true \
#  --inter_cluster_constraint true \
#  --valid_steps 20000


#  CUDA_VISIBLE_DEVICES=4 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset FB15K-237 \
#  --model TransE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 8.0 \
#  -n 256 -b 1024 -d 2048 \
#  -g 9.0 -a 1.0 -adv \
#  -lr 0.00005 --max_steps 400000 \
#  --test_batch_size 8 \
#  --text_dist_constraint true \
#  --hier_dist_constraint false \
#  --intra_cluster_constraint true \
#  --inter_cluster_constraint true \
#  --valid_steps 20000 \