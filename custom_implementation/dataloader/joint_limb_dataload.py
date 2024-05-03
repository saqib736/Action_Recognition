import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image 
import os
import glob
import random

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
])

class TemporalHeatmapDataset(Dataset):
    def __init__(self, root_dir, split='train', train_size=90, seed=42):
        """
        Initialize the dataset with a train/test split.
        
        Args:
            root_dir (str): Root directory with all the sequence folders.
            split (str): 'train' or 'test' to denote the type of dataset to return.
            train_size (int): Number of samples in the train set.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.split = split
        self.activity_labels = {}
        self.data_samples = []

        # Set the random seed for reproducibility
        random.seed(seed)

        # Load all sequences
        activity_types = sorted(os.listdir(root_dir))
        for idx, activity in enumerate(activity_types):
            self.activity_labels[activity] = idx
            activity_path = os.path.join(root_dir, activity)
            sequence_dirs = [os.path.join(activity_path, seq) for seq in os.listdir(activity_path) if os.path.isdir(os.path.join(activity_path, seq))]
            for seq_dir in sequence_dirs:
                self.data_samples.append((seq_dir, self.activity_labels[activity]))

        # Shuffle and split the data
        random.shuffle(self.data_samples)
        if split == 'train':
            self.data_samples = self.data_samples[:train_size]
        elif split == 'test':
            self.data_samples = self.data_samples[train_size:]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sequence_path, label = self.data_samples[idx]

        # Get the corresponding limbs path by replacing 'joints' with 'limbs' in the path
        limbs_path = sequence_path.replace('joints', 'RGB')

        # Load joint frames
        joint_frame_files = sorted(glob.glob(f"{sequence_path}/*.npy"))
        joint_frames = [np.load(file) for file in joint_frame_files]
        joint_frames = np.stack(joint_frames, axis=0)

        # Load limb frames
        limb_frame_files = sorted(glob.glob(f"{limbs_path}/*.png"))
        # limb_frames = [np.load(file) for file in limb_frame_files]
        limb_frames = [Image.open(file) for file in limb_frame_files]
        transformed_frames = [transform(frame) for frame in limb_frames]
        if len(transformed_frames) == 0:
            print(limbs_path)
        limb_frames = np.stack(transformed_frames, axis=0)
        
        
        joint_tensor = torch.tensor(joint_frames, dtype=torch.float32).permute(1, 0, 2, 3)
        limb_tensor = torch.tensor(limb_frames, dtype=torch.float32).permute(1, 0, 2, 3)

        return joint_tensor, limb_tensor, label


def collate_fn(batch, fixed_length=60):
    # Helper function to adjust each sequence
    def adjust_sequence(seq, fixed_length):
        if seq.size(0) > fixed_length:
            return seq[:fixed_length]  # Truncate the sequence
        elif seq.size(0) < fixed_length:
            # Pad the sequence
            return torch.cat([seq, seq.new_zeros(fixed_length - seq.size(0), *seq.shape[1:])], dim=0)
        return seq  # Return as is if the size matches

    # Process joint frames
    frames_j = [item[0].permute(1, 0, 2, 3) for item in batch] 
    lengths = torch.tensor([min(f.size(0), fixed_length) for f in frames_j])
    frames_padded_j = [adjust_sequence(f, fixed_length) for f in frames_j]
    frames_padded_j = torch.stack(frames_padded_j)
    
    # Process label frames
    frames_l = [item[1].permute(1, 0, 2, 3) for item in batch]
    frames_padded_l = [adjust_sequence(f, fixed_length) for f in frames_l]
    frames_padded_l = torch.stack(frames_padded_l)
    
    # Prepare labels
    labels = torch.tensor([item[2] for item in batch])
    
    return frames_padded_j.permute(0, 2, 1, 3, 4), frames_padded_l.permute(0, 2, 1, 3, 4), lengths, labels

# Usage example with DataLoader
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset_path = 'dataset_new/joints'
    train_dataset = TemporalHeatmapDataset(dataset_path, split='train', train_size=90, seed=42)
    test_dataset = TemporalHeatmapDataset(dataset_path, split='test', train_size=90, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f'Length Train Loader: {len(train_loader)}')
    print(f'Length Test Loader: {len(test_loader)}')

    # Example processing for training
    for frames_joints, frames_rgb, lengths, labels in train_loader:
        print(f"Train Batch frames joint size: {frames_joints.shape}")
        print(f"Train Batch frames limb size: {frames_rgb.shape}")
        print(f"Batch lengths joints size: {lengths}")
        print(f"Train Labels: {labels}")

    # Example processing for testing
    for frames_joints, frames_rgb, labels in test_loader:
        print(f"Test Batch frames joint size: {frames_joints.shape}")
        print(f"Test Batch frames limb size: {frames_rgb.shape}")
        print(f"Test Labels: {labels}")