import numpy as np  
import cv2

if __name__ == '__main__':
    depth_path = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_160-240_physics_fog/depth/000000.npy' 
    depth = np.load(depth_path)
    
    mask_save_path = '/shortdata/ziwang/projects/nerf-factory/figs/mask_600.png'
    mask = np.zeros_like(depth)
    mask[depth>600] = 255
    cv2.imwrite(mask_save_path, mask)
    
    # import ipdb 
    # ipdb.set_trace()