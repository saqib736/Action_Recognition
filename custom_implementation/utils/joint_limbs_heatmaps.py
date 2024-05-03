import numpy as np
import pandas as pd
import cv2
import json
import os

def generate_gaussian_heatmap(height, width, joints, sigma=1):
    heatmaps = np.zeros((len(joints) // 3, height, width), dtype=np.float32)
    for idx in range(0, len(joints), 3):
        x, y, v = joints[idx], joints[idx+1], int(joints[idx+2])
        if v == 0:
            continue
        heatmap = np.zeros((height, width), dtype=np.float32)
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        heatmap[int(y), int(x)] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmaps[idx // 3] = heatmap
    return heatmaps

def generate_limb_heatmaps(height, width, keypoints, skeleton, sigma=1):
    limbs = np.zeros((len(skeleton), height, width), dtype=np.float32)
    for idx, (j1, j2) in enumerate(skeleton):
        pt1 = keypoints[3 * j1:3 * j1 + 3]
        pt2 = keypoints[3 * j2:3 * j2 + 3]
        if pt1[2] == 0 or pt2[2] == 0:
            continue
        limb_heatmap = generate_line_heatmap(height, width, pt1[:2], pt2[:2], sigma)
        limbs[idx] = np.maximum(limbs[idx], limb_heatmap)
    return limbs

def generate_line_heatmap(height, width, pt1, pt2, sigma=1):
    img = np.zeros((height, width), dtype=np.float32)
    cv2.line(img, tuple(int(x) for x in pt1), tuple(int(x) for x in pt2), 1, thickness=1)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    if img.max() > 0:
        img /= img.max()
    return img

def save_heatmaps_as_numpy(heatmaps, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, heatmaps)

def load_sequence_order(csv_path):
    sequence_df = pd.read_csv(csv_path)
    return sequence_df['filename'].tolist()

def crop_and_resize_heatmap(heatmap, bbox, target_size=(56, 56)):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h) 
    cropped = heatmap[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized

def generate_and_save_heatmaps(annotation_path, output_path, csv_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    sequence_order = load_sequence_order(csv_path)
    images_info = {img['file_name']: img for img in data['images']}

    for file_name in sequence_order:
        image_info = images_info.get(file_name)
        if not image_info:
            continue

        sequence_label = image_info.get('label')
        sequence_id = image_info.get('sequence_id')
        kp_path = os.path.join(output_path, 'joints', sequence_label, sequence_id)
        limb_path = os.path.join(output_path, 'limbs', sequence_label, sequence_id)

        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_info['id']]
        for ann in annotations:
            keypoints = ann['keypoints']
            height, width = image_info['height'], image_info['width']
            bbox = ann['bbox']
            kp_heatmaps = generate_gaussian_heatmap(height, width, keypoints)
            limb_heatmaps = generate_limb_heatmaps(height, width, keypoints, data['categories'][0]['skeleton'])
            
            # Crop and resize each heatmap
            kp_heatmaps = np.array([crop_and_resize_heatmap(hm, bbox) for hm in kp_heatmaps])
            limb_heatmaps = np.array([crop_and_resize_heatmap(hm, bbox) for hm in limb_heatmaps])

            # Save the joint and limb heatmaps separately
            frame_filename = f"{image_info['file_name'].replace('.png', '.npy')}"
            save_heatmaps_as_numpy(kp_heatmaps, os.path.join(kp_path, frame_filename))
            save_heatmaps_as_numpy(limb_heatmaps, os.path.join(limb_path, frame_filename))

# Example call
generate_and_save_heatmaps('filenames/train_annotations.json', 'dataset_new', 'filenames/label_data.csv')
