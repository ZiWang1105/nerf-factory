import numpy as np

if __name__ == '__main__':
    poses_bounds_path = '/shortdata/ziwang/projects/nerf-factory/data/waymo_multi_view/segment-10247954040621004675_2180_000_2200_000_with_camera_labels/poses_bounds_c2w.npy'
    poses_bounds = np.load(poses_bounds_path)
    print(poses_bounds.shape)
    
    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, -1:]
    bottom = np.array([0., 0., 0., 1.])
    poses_new = np.zeros((poses.shape[0], 4, 4))
    for i in range(poses.shape[0]):
        poses_new[i] = np.concatenate([poses[i], bottom.reshape(1, 4)], axis=0)
        
    for i in range(poses_new.shape[0]):
        poses_new[i] = np.linalg.inv(poses_new[i])
        
    poses_new = poses_new[:, :3, :4]
        
    poses_hwf_new = np.concatenate([poses_new, hwf], axis=-1)
    poses_bounds[:, :15] = poses_hwf_new.reshape(-1, 15)
    np.save('/shortdata/ziwang/projects/nerf-factory/data/waymo_multi_view/segment-10247954040621004675_2180_000_2200_000_with_camera_labels/poses_bounds_w2c.npy', poses_bounds)
    