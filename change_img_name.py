import os

def rename_images(folder_path):
    files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    files.sort()

    new_index = 0

    for filename in files:
        old_file_path = os.path.join(folder_path, filename)
        new_filename = f'{new_index:05d}.png'
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(old_file_path, new_file_path)
        new_index += 1

    print(f"Total images renamed: {new_index}")

folder_path = '/shortdata/ziwang/projects/nerf_carla_data/output_foggy_world_new/images'

rename_images(folder_path)
