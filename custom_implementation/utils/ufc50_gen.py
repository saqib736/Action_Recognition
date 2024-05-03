import cv2
import os
import numpy as np
from random import sample

def select_random_activities(base_path, num_activities=10):
    all_activities = os.listdir(base_path)
    selected_activities = sample(all_activities, num_activities)
    return selected_activities

def select_random_videos(activity_path, num_videos=40):
    all_videos = [video for video in os.listdir(activity_path) if video.endswith('.avi')]
    selected_videos = sample(all_videos, min(num_videos, len(all_videos)))
    return selected_videos

def extract_frames(video_path, output_dir, fps=10):
    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate_ratio = int(video_fps / fps)
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frames at 10 fps
        if frame_count % frame_rate_ratio == 0:
            frame_filename = os.path.join(output_dir, f'{saved_frame_count+1:04d}.png')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1
    
    cap.release()

def main():
    base_path = 'UCF50'  # Path to the dataset directory
    root_output_dir = 'ufc_dataset'  # Root directory for output
    
    selected_activities = select_random_activities(base_path)
    
    for activity in selected_activities:
        activity_path = os.path.join(base_path, activity)
        videos = select_random_videos(activity_path)
        
        for video in videos:
            video_path = os.path.join(activity_path, video)
            sequence_output_dir = os.path.join(root_output_dir, activity, video.split('.')[0])
            extract_frames(video_path, sequence_output_dir)

if __name__ == "__main__":
    main()
