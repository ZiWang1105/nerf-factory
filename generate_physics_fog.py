import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

def get_fog(img_path, depth_path, output_path):
    image = cv2.imread(img_path)
    depth = np.load(depth_path)
    # A = min(np.abs(np.random.normal(210, 15, 1))[0], 255)
    A = 200
    depth_img_3c = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    norm_depth_img = depth_img_3c 
    
    # beta = np.abs(np.random.normal(5, 1, 1))[0]
    beta = 0.015  # 0.02
    trans = np.exp(-norm_depth_img * beta)
    hazy = image * trans + A * (1 - trans)
    hazy = np.array(hazy, dtype=np.uint8)
    cv2.imwrite(os.path.join(output_path, img_path.split('/')[-1]), hazy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--img_dir', type=str, default = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_160-240_physics_fog/images_without_fog')
    # parser.add_argument('--depth_dir', type = str, default = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_160-240_physics_fog/depth')
    # parser.add_argument('--output_path', type = str, default = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_160-240_physics_fog/images')
    
    parser.add_argument('--img_dir', type=str, default = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_728-828_physics_fog/images_without_fog')
    parser.add_argument('--depth_dir', type = str, default = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_728-828_physics_fog/depth')
    parser.add_argument('--output_path', type = str, default = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_728-828_physics_fog/images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    img_path = os.listdir(args.img_dir)
    depth_path = os.listdir(args.depth_dir)
    img_path.sort()
    depth_path.sort()
    
    img_path = [os.path.join(args.img_dir, i) for i in img_path]
    depth_path = [os.path.join(args.depth_dir, i) for i in depth_path]
    
    for img, depth in tqdm(zip(img_path, depth_path), desc='Processing images'):
        get_fog(img, depth, args.output_path)
    
    