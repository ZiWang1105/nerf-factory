import cv2
import os

image_folder = '/shortdata/ziwang/projects/nerf-factory/logs/mipnerf360_nerf_360_v2_output_clear_world_new_220901/render_model'
video_path = '/shortdata/ziwang/projects/nerf-factory/video/tmp.mp4'
 
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort() 

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
video = cv2.VideoWriter(video_path, fourcc, 2, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
