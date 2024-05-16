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
 --zeta_3 1.8 \
 -n 1024 -b 512 -d 512 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.00001 --max_steps 400000 \
 --test_batch_size 8 -de -tr \
 --hake_p 0.5 \
 --hake_m 0.5


#  CUDA_VISIBLE_DEVICES=2 python -u code/run.py \
#  --do_train \
#  --cuda \
#  --do_valid \
#  --do_test \
#  --data_path data \
#  --dataset YAGO3-10 \
#  --model HAKE \
#  --distance_metric cosine \
#  --hierarchy_type llm \
#  --zeta_3 1.8 \
#  -n 256 -b 256 -d 768 \
#  -g 24.0 -a 1.0 -adv \
#  -lr 0.0002 --max_steps 800000 \
#  --test_batch_size 8 -de -tr \
#  --hake_p 1.0 \
#  --hake_m 0.5