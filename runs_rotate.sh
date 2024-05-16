CUDA_VISIBLE_DEVICES=7 python -u code/run.py \
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data \
 --dataset PrimeKG \
 --model RotatE \
 --distance_metric cosine \
 --hierarchy_type seed \
 --zeta_3 1.8 \
 -n 1024 -b 256 -d 512 \
 -g 6.0 -a 0.5 -adv \
 -lr 0.00005 --max_steps 300000 \
 --test_batch_size 8 -de \
 --inter_cluster_constraint false 


# CUDA_VISIBLE_DEVICES=4 python -u code/run_p_anc.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset WN18RR \
#  --model RotatE \
#  --distance_metric rotate \
#  --hierarchy_type seed \
#  --zeta_3 1.8 \
#  -n 512 -b 256 -d 512 \
#  -g 6.0 -a 0.5 -adv \
#  -lr 0.00005 --max_steps 400000 \
#  --test_batch_size 8 -de \
#  --inter_cluster_constraint false 

#  CUDA_VISIBLE_DEVICES=6 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model RotatE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 2.0 \
#  -n 256 -b 256 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 400000 \
#  --test_batch_size 8 -de


#  CUDA_VISIBLE_DEVICES=6 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset FB15K-237 \
#  --model RotatE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 2.0 \
#  -n 256 -b 256 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 400000 \
#  --test_batch_size 8 -de