CUDA_VISIBLE_DEVICES=1 python -u code/run.py \
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data \
 --dataset PrimeKG \
 --model TransE \
 --distance_metric cosine \
 --hierarchy_type llm \
 --zeta_3 1.8 \
 -n 256 -b 256 -d 1024 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.00001 --max_steps 400000 \
 --test_batch_size 8 \
 

#  CUDA_VISIBLE_DEVICES=3 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model TransE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 2.0 \
#  -n 256 -b 256 -d 1024 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 400000 \
#  --test_batch_size 8 \


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
#  --zeta_3 2.0 \
#  -n 256 -b 256 -d 2048 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0001 --max_steps 400000 \
#  --test_batch_size 8 \