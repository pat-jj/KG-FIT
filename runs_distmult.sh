CUDA_VISIBLE_DEVICES=5 python -u code/run_1.py \
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data \
 --dataset FB15K-237 \
 --model DistMult \
 --distance_metric cosine \
 --zeta_3 2.0 \
 -n 256 -b 256 -d 2048 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 --test_batch_size 16 