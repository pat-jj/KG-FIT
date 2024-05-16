CUDA_VISIBLE_DEVICES=1 python -u code/run.py \
 --cuda \
 --do_test \
 --data_path data \
 --dataset PrimeKG \
 --model pRotatE \
 --distance_metric cosine \
 --hierarchy_type seed \
 --zeta_3 1.8 \
 -n 1024 -b 512 -d 512 \
 -g 24.0 -a 0.5 -adv \
 -lr 0.00005 --max_steps 400000 \
 --test_batch_size 8 \
 -init /shared/pj20/lamake_data/PrimeKG/checkpoints/pRotatE_seed_batch_256_hidden_1024_dist_cosine \
 --rerank false \
 --fuse_score true \

