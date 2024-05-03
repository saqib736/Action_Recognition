import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class VideoDataset(Dataset):
    def __init__(self, dataset_path, input_shape, sequence_length, training=True, split_ratio=0.9):
        self.dataset_path = dataset_path
        self.training = training
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        all_sequences, self.labels = self._extract_sequences_and_labels()
        split_index = int(len(all_sequences) * split_ratio)
        if self.training:
            self.sequences = all_sequences[:split_index]
        else:
            self.sequences = all_sequences[split_index:]

    def _extract_sequences_and_labels(self):
        sequences = []
        labels = {}
        label_idx = 0
        # Traverse the dataset directory to extract sequences
        for label_name in sorted(os.listdir(self.dataset_path)):
            label_path = os.path.join(self.dataset_path, label_name)
            if os.path.isdir(label_path):
                for sequence in os.listdir(label_path):
                    sequence_path = os.path.join(label_path, sequence)
                    if os.path.isdir(sequence_path):
                        sequences.append(sequence_path)
                        if label_name not in labels:
                            labels[label_name] = label_idx
                            label_idx += 1
        # Shuffle sequences to ensure random distribution for training and testing
        random.shuffle(sequences)
        return sequences, labels

    # def _frame_number(self, image_path):
    #     """ Extracts frame number from filepath """
    #     return int(image_path.split("/")[-1].split(".png")[0])
    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        # Extract the filename from the full image path
        filename = image_path.split("/")[-1]
        # Split the filename by "_" and take the first part, which contains the frame number
        frame_number = filename.split("_")[0]
        # Convert the frame number part to an integer
        return int(frame_number)

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        while len(sequence) < self.sequence_length:
            sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        image_paths = sorted(glob.glob(f"{sequence_path}/*.png"), key=lambda path: self._frame_number(path))
        image_paths = self._pad_to_length(image_paths[:self.sequence_length])

        # Random sampling or uniform sampling based on training or not
        if self.training:
            start_i = np.random.randint(0, max(1, len(image_paths) - self.sequence_length + 1))
            sample_interval = np.random.randint(1, max(1, len(image_paths) // self.sequence_length + 1))
            image_paths = image_paths[start_i:start_i + sample_interval * self.sequence_length:sample_interval]
            flip = np.random.random() < 0.5
        else:
            sample_interval = max(1, len(image_paths) // self.sequence_length)
            image_paths = image_paths[:self.sequence_length * sample_interval:sample_interval]
            flip = False

        image_sequence = [self.transform(Image.open(img_path)) for img_path in image_paths]
        if flip:
            image_sequence = [torch.flip(img, (-1,)) for img in image_sequence]

        image_sequence = torch.stack(image_sequence)
        label_name = os.path.basename(os.path.dirname(sequence_path))
        target = self.labels[label_name]

        return image_sequence, target #.permute(1, 0, 2, 3)

    def __len__(self):
        return len(self.sequences)


from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset_path = 'ufc_dataset'
    
    image_shape = (3, 224, 224)
    
    train_dataset = VideoDataset(dataset_path, input_shape=image_shape, sequence_length=40, training=True)
    test_dataset = VideoDataset(dataset_path, input_shape=image_shape, sequence_length=40, training=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f'Length Train Loader: {len(train_loader)}')
    print(f'Length Test Loader: {len(test_loader)}')

    # Example processing for training
    for frames, labels in train_loader:
        print(f"Train Batch frames size: {frames.shape}")
        print(f"Train Labels: {labels}")

    # Example processing for testing
    for frames, labels in test_loader:
        print(f"Test Batch frames size: {frames.shape}")
        print(f"Test Labels: {labels}")
        
