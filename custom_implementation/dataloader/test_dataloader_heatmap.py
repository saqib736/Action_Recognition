import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, dataset_path, sequence_length, training=True, split_ratio=0.9):
        self.dataset_path = dataset_path
        self.training = training
        self.sequence_length = sequence_length
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
        random.shuffle(sequences)
        return sequences, labels

    def _frame_number(self, file_path):
        filename = file_path.split("/")[-1]
        frame_number = filename.split("_")[0]
        return int(frame_number)

    def _pad_to_length(self, sequence):
        left_pad = sequence[0]
        while len(sequence) < self.sequence_length:
            sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        file_paths = sorted(glob.glob(f"{sequence_path}/*.npy"), key=lambda path: self._frame_number(path))
        file_paths = self._pad_to_length(file_paths[:self.sequence_length])

        if self.training:
            start_i = np.random.randint(0, max(1, len(file_paths) - self.sequence_length + 1))
            sample_interval = np.random.randint(1, max(1, len(file_paths) // self.sequence_length + 1))
            file_paths = file_paths[start_i:start_i + sample_interval * self.sequence_length:sample_interval]
            flip = np.random.random() < 0.5
        else:
            sample_interval = max(1, len(file_paths) // self.sequence_length)
            file_paths = file_paths[:self.sequence_length * sample_interval:sample_interval]
            flip = False

        data_sequence = [np.load(file_path) for file_path in file_paths]
        data_sequence = np.stack(data_sequence, axis=0)  # Shape will be [sequence_length, 27, H, W]
        data_tensor = torch.from_numpy(data_sequence).float()
        
        if flip:
        # Flip the data along the width dimension, which is assumed to be the last dimension (index -1)
            data_tensor = torch.flip(data_tensor, [-1])

        label_name = os.path.basename(os.path.dirname(sequence_path))
        target = self.labels[label_name]

        return data_tensor.permute(1, 0, 2, 3), target #

    def __len__(self):
        return len(self.sequences)


from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset_path = 'dataset'
    
    train_dataset = NumpyDataset(dataset_path, sequence_length=40, training=True)
    test_dataset = NumpyDataset(dataset_path, sequence_length=40, training=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f'Length Train Loader: {len(train_loader)}')
    print(f'Length Test Loader: {len(test_loader)}')

    # Example processing for training
    for frames, labels in train_loader:
        print(f"Train Batch frames size: {frames.shape}")  # Shape [batch_size, sequence_length, 27, H, W]
        print(f"Train Labels: {labels}")

    # Example processing for testing
    for frames, labels in test_loader:
        print(f"Test Batch frames size: {frames.shape}")
        print(f"Test Labels: {labels}")
