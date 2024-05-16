 CUDA_VISIBLE_DEVICES=1 python -u code/run.py \
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data \
 --dataset PrimeKG \
 --model pRotatE \
 --distance_metric cosine \
 --hierarchy_type llm \
 --zeta_3 1.8 \
 -n 256 -b 256 -d 2048 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.00001 --max_steps 400000 \
 --test_batch_size 8

#  CUDA_VISIBLE_DEVICES=4 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset PrimeKG \
#  --model pRotatE \
#  --distance_metric cosine \
#  --hierarchy_type seed \
#  --zeta_3 1.8 \
#  -n 1024 -b 512 -d 512 \
#  -g 6.0 -a 0.5 -adv \
#  -lr 0.00005 --max_steps 400000 \
#  --test_batch_size 8

# CUDA_VISIBLE_DEVICES=5 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset WN18RR \
#  --model pRotatE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 1.8 \
#  -n 1024 -b 512 -d 512 \
#  -g 6.0 -a 0.5 -adv \
#  -lr 0.00005 --max_steps 400000 \
#  --test_batch_size 8


#  CUDA_VISIBLE_DEVICES=2 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model pRotatE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 2.0 \
#  -n 256 -b 128 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 1600000 \
#  --test_batch_size 8 \
#  -init /shared/pj20/lamake_data/YAGO3-10/checkpoints/pRotatE_seed_batch_128_hidden_1024_dist_cosine


#  CUDA_VISIBLE_DEVICES=2 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model pRotatE \
#  --distance_metric cosine \
#  --hierarchy_type seed \
#  --zeta_3 2.0 \
#  -n 256 -b 128 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 1600000 \
#  --test_batch_size 8 \
#  -init /shared/pj20/lamake_data/YAGO3-10/checkpoints/pRotatE_seed_batch_128_hidden_1024_dist_cosine