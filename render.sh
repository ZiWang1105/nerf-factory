CUDA_VISIBLE_DEVICES=1 python3 run.py --ginc configs/mipnerf360/carla.gin  \
--scene output_clear_world_with_depth_1822-1999 --seed 240504 \
--ginb run.run_train=False --ginb run.run_render=False