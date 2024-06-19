import numpy as np
import os
import argparse
import shutil

def transform_extrinsics_to_first_frame_origin(extrinsics):
    # extrinsics should be a NumPy array of shape [N, 4, 4]
    # Calculate the inverse of the first frame's extrinsics matrix
    first_frame_inv = np.linalg.inv(extrinsics[0])

    # Initialize an array to hold the transformed matrices
    transformed_extrinsics = np.zeros_like(extrinsics)

    # Apply the transformation to each matrix
    for i in range(len(extrinsics)):
        transformed_extrinsics[i] = np.dot(first_frame_inv, extrinsics[i])

    return transformed_extrinsics

def convert_ue_to_llff(extrinsics):
    # Create the transformation matrix to convert from UE to LLFF
    transform_matrix = np.array([
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Initialize an array to hold the transformed extrinsics
    transformed_extrinsics = np.zeros_like(extrinsics)

    # Apply the transformation matrix to each extrinsic matrix
    for i in range(extrinsics.shape[0]):
        transformed_extrinsics[i] = np.dot(extrinsics[i], transform_matrix)

    return transformed_extrinsics

def rename_files(folder_path):
    files = [file for file in os.listdir(folder_path) if (file.endswith('.png') or file.endswith('.npy'))]
    files.sort()
    is_image = files[0].endswith('.png')

    new_index = 0

    for filename in files:
        old_file_path = os.path.join(folder_path, filename)
        if is_image:
            new_filename = f'{new_index:06d}.png'
        else:
            new_filename = f'{new_index:06d}.npy'
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(old_file_path, new_file_path)
        new_index += 1

    print(f"Total files renamed: {new_index}")
    
def clear_directory(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"The directory {path} does not exist.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--carla_data_path', 
                        type=str, 
                        default = '/shortdata/ziwang/projects/carla-simulator/PythonAPI/outputs/output_clear_world_with_depth01')
    
    parser.add_argument('--output_path',
                        type=str,
                        default='/shortdata/ziwang/projects/nerf-factory/data/carla')
    
    parser.add_argument('--start_frame', 
                        type=int, 
                        default=2)   
    
    parser.add_argument('--end_frame',
                        type=int,
                        default=101)
    
    parser.add_argument('--scene_name',
                        type=str,
                        default='output_clear_world_with_depth_2-101_physics_fog')
    
    args = parser.parse_args()
    
    scene_name = args.carla_data_path.split('/')[-1]
    output_path = os.path.join(args.output_path, args.scene_name)
    os.makedirs(output_path, exist_ok=True) 
    clear_directory(output_path)
        
    ori_extrinsics_dir = os.path.join(args.carla_data_path, 'extrinsics')
    ori_images_dir = os.path.join(args.carla_data_path, 'images')
    ori_depth_dir = os.path.join(args.carla_data_path, 'depth')
        
    extrinsics_dir = os.path.join(output_path, 'extrinsics')
    images_dir = os.path.join(output_path, 'images')
    depth_dir = os.path.join(output_path, 'depth')
    
    os.makedirs(extrinsics_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # copy images, depth, and extrinsics
    for frame in range(args.start_frame, args.end_frame + 1):
        frame_str = f'{frame:06d}'

        ori_extrinsic_file = os.path.join(ori_extrinsics_dir, frame_str + '.npy')
        ori_image_file = os.path.join(ori_images_dir, frame_str + '.png')
        ori_depth_file = os.path.join(ori_depth_dir, frame_str + '.npy')
        
        new_extrinsic_file = os.path.join(extrinsics_dir, frame_str + '.npy')
        new_image_file = os.path.join(images_dir, frame_str + '.png')
        new_depth_file = os.path.join(depth_dir, frame_str + '.npy')

        shutil.copy(ori_extrinsic_file, new_extrinsic_file)
        shutil.copy(ori_image_file, new_image_file)
        shutil.copy(ori_depth_file, new_depth_file)
        
    # rename files
    rename_files(images_dir)
    rename_files(depth_dir)
    rename_files(extrinsics_dir)

    # convert extrinsics to llff format
    extrinsics_files = os.listdir(extrinsics_dir)
    extrinsics_files.sort()
    extrinsics_files = [os.path.join(extrinsics_dir, f) for f in extrinsics_files]
    
    extrinsics = []
    for f in extrinsics_files:
        extrinsics.append(np.load(f, allow_pickle=True))
        
    extrinsics = np.array(extrinsics)
    print(extrinsics.shape)
    
    intrinsics_dir = os.path.join(args.carla_data_path, 'intrinsics.npy')
    intrinsics_dict = np.load(intrinsics_dir, allow_pickle = True).item()
    hwf = np.array([intrinsics_dict['h'], intrinsics_dict['w'], intrinsics_dict['f']])
    
    extrinsics = convert_ue_to_llff(extrinsics)
    extrinsics = extrinsics[:, :3, :]
    last_row = np.array([[0, 0, 0, 1]]).reshape(1, 1, 4).repeat(extrinsics.shape[0], axis=0)
    extrinsics = np.concatenate([extrinsics, last_row], axis=1)
    extrinsics = transform_extrinsics_to_first_frame_origin(extrinsics)
    extrinsics = extrinsics[:, :3, :]
    
    poses_bounds = np.zeros((extrinsics.shape[0], 17))
    hwf = hwf.reshape(1, 3, 1).repeat(extrinsics.shape[0], axis=0)
    poses = np.concatenate([extrinsics, hwf], axis=2)
    
    poses_bounds[:, :15] = poses.reshape(-1, 15)
    poses_bounds[:, 15] = 0.1
    poses_bounds[:, 16] = 600
    
    poses_bounds_save_path = os.path.join(output_path, 'poses_bounds.npy')
    np.save(poses_bounds_save_path, poses_bounds)
    
    # LLFF format to Opencv format
    poses_bounds = np.load(poses_bounds_save_path)

    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, -1:]    
    poses = np.concatenate([poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)
    
    bottom = np.array([0., 0., 0., 1.])
    extrinsics = np.zeros((poses.shape[0], 4, 4))
    for i in range(poses.shape[0]):
        extrinsics[i] = np.concatenate([poses[i], bottom.reshape(1, 4)], axis=0)
        
    extrinsics_save_path = os.path.join(output_path, 'extrinsics.npy')
    np.save(extrinsics_save_path, extrinsics)
    