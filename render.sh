CUDA_VISIBLE_DEVICES=1 python3 run.py --ginc configs/fognerf/carla.gin  \
--scene output_clear_world_with_depth_1822-1999_physics_fog --seed 240504 \
--ginb run.run_train=False --ginb run.run_render=False