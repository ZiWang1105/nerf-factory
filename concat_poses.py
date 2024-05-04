import numpy as np
import os

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

# def convert_ue_to_nerf(extrinsics):
#     transform_matrix = np.array([
#         [0, 1, 0, 0],
#         [0, 0, -1, 0],
#         [-1, 0, 0, 0],
#         [0, 0, 0, 1]
#     ])

#     transformed_extrinsics = np.zeros_like(extrinsics)
#     for i in range(extrinsics.shape[0]):
#         transformed_extrinsics[i] = np.dot(extrinsics[i], transform_matrix)

#     return transformed_extrinsics

# def convert_ue_to_opencv(extrinsics):
#     transform_matrix = np.array([
#         [0, 1, 0, 0],
#         [0, 0, -1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 1]
#     ])

#     transformed_extrinsics = np.zeros_like(extrinsics)
#     for i in range(extrinsics.shape[0]):
#         transformed_extrinsics[i] = np.dot(extrinsics[i], transform_matrix)

#     return transformed_extrinsics

if __name__ == '__main__':
    extrinsics_dir = '/shortdata/ziwang/projects/nerf_carla_data/output_foggy_world_new/extrinsics'
    extrinsics_files = os.listdir(extrinsics_dir)
    extrinsics_files.sort()
    print(extrinsics_files)
    extrinsics_files = [os.path.join(extrinsics_dir, f) for f in extrinsics_files]
    
    extrinsics = []
    for f in extrinsics_files:
        # extrinsics.append(np.linalg.inv(np.load(f, allow_pickle = True)))
        extrinsics.append(np.load(f, allow_pickle=True))

    extrinsics = np.array(extrinsics)
    print(extrinsics.shape)

    intrinsics_dir = '/shortdata/ziwang/projects/nerf_carla_data/output_foggy_world_new/intrinsics.npy'
    intrinsics_dict = np.load(intrinsics_dir, allow_pickle = True).item()
    hwf = np.array([intrinsics_dict['h'], intrinsics_dict['w'], intrinsics_dict['f']])

    # extrinsics = extrinsics[:, :3, :]
    # poses_bounds = np.zeros((extrinsics.shape[0], 17))

    # hwf = hwf.reshape(1, 3, 1).repeat(extrinsics.shape[0], axis=0)
    # poses = np.concatenate([extrinsics, hwf], axis=2)
    # print(poses.shape)

    # poses_bounds[:, :15] = poses.reshape(-1, 15)
    # poses_bounds[:, 15] = 0.1
    # poses_bounds[:, 16] = 600

    # np.save('/home/ilim/ziwang/carla/poses_bounds_tmp.npy', poses_bounds)
    ##### work ##############################################################################################
    
    # extrinsics = convert_ue_to_llff(extrinsics)
    
    # extrinsics = extrinsics[:, :3, :]
    # last_row = np.array([[0, 0, 0, 1]]).reshape(1, 1, 4).repeat(extrinsics.shape[0], axis=0)
    # extrinsics = np.concatenate([extrinsics, last_row], axis=1)
    # extrinsics = transform_extrinsics_to_first_frame_origin(extrinsics)
    # extrinsics = extrinsics[:, :3, :]
    
    # poses_bounds = np.zeros((extrinsics.shape[0], 17))
    # hwf = hwf.reshape(1, 3).repeat(extrinsics.shape[0], axis=0)
    # poses_bounds[:, :12] = extrinsics.reshape(-1, 12)
    # poses_bounds[:, 12:15] = hwf
    # poses_bounds[:, 15] = 0.1
    # poses_bounds[:, 16] = 1000
    ##### work ##############################################################################################
    
    extrinsics = convert_ue_to_llff(extrinsics)
    extrinsics = extrinsics[:, :3, :]
    last_row = np.array([[0, 0, 0, 1]]).reshape(1, 1, 4).repeat(extrinsics.shape[0], axis=0)
    extrinsics = np.concatenate([extrinsics, last_row], axis=1)
    extrinsics = transform_extrinsics_to_first_frame_origin(extrinsics)
    extrinsics = extrinsics[:, :3, :]
    
    poses_bounds = np.zeros((extrinsics.shape[0], 17))
    hwf = hwf.reshape(1, 3, 1).repeat(extrinsics.shape[0], axis=0)
    poses = np.concatenate([extrinsics, hwf], axis=2)
    print(poses.shape)
    # import ipdb 
    # ipdb.set_trace()
    poses_bounds[:, :15] = poses.reshape(-1, 15)
    poses_bounds[:, 15] = 0.1
    poses_bounds[:, 16] = 600


    np.save('/shortdata/ziwang/projects/nerf_carla_data/output_foggy_world_new/poses_bounds.npy', poses_bounds)