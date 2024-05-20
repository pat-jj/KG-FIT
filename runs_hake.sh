CUDA_VISIBLE_DEVICES=4 python -u code/run.py \
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data \
 --dataset PrimeKG \
 --model HAKE \
 --distance_metric cosine \
 --hierarchy_type llm \
 --zeta_3 9.0 \
 -n 512 -b 512 -d 1024 \
 -g 10.0 -a 1.0 -adv \
 -lr 0.00001 --max_steps 400000 \
 --test_batch_size 8 -de -tr \
 --hake_p 1.5 \
 --hake_m 0.5 \
 --text_dist_constraint true \
 --hier_dist_constraint false \
 --intra_cluster_constraint true \
 --inter_cluster_constraint true \


#  CUDA_VISIBLE_DEVICES=6 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model HAKE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 8.0 \
#  -n 512 -b 256 -d 512 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0002 --max_steps 1600000 \
#  --test_batch_size 8 -de -tr \
#  --hake_p 3.5 \
#  --hake_m 0.5 \
#  --text_dist_constraint true \
#  --hier_dist_constraint false \
#  --intra_cluster_constraint false \
#  --inter_cluster_constraint true \
#  -init /shared/pj20/lamake_data/YAGO3-10/checkpoints/HAKE_llm_batch_256_hidden_512_dist_cosine

#  CUDA_VISIBLE_DEVICES=2 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset FB15K-237 \
#  --model HAKE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 3.5 \
#  -n 256 -b 256 -d 1024 \
#  -g 9.0 -a 1.0 -adv \
#  -lr 0.00005 --max_steps 400000 \
#  --test_batch_size 8 \
#  --text_dist_constraint true \
#  --hier_dist_constraint false \
#  --intra_cluster_constraint true \
#  --inter_cluster_constraint true \
#  --valid_steps 20000 -de -tr \