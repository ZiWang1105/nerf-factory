import numpy as np

if __name__ == '__main__':
    poses_bounds_path = '/shortdata/ziwang/projects/nerf_carla_data/output_foggy_world_new/poses_bounds.npy'
    extrinsics_path = '/shortdata/ziwang/projects/nerf_carla_data/output_foggy_world_new/extrinsics.npy'
    
    poses_bounds = np.load(poses_bounds_path)
    # extrinsics = np.load(extrinsics_path)

    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, -1:]
    print(poses[0])
    # print(extrinsics[0])
    poses = np.concatenate([poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)
    print(poses[0])
    bottom = np.array([0., 0., 0., 1.])
    extrinsics = np.zeros((poses.shape[0], 4, 4))
    for i in range(poses.shape[0]):
        extrinsics[i] = np.concatenate([poses[i], bottom.reshape(1, 4)], axis=0)
    
    np.save(extrinsics_path, extrinsics)
    
    # poses_hwf_new = np.concatenate([poses, hwf], axis=-1)
    # poses_bounds[:, :15] = poses_hwf_new.reshape(-1, 15)
    # np.save('/shortdata/ziwang/projects/nerf_carla_data/output_clear_world_new/poses_bounds.npy', poses_bounds)