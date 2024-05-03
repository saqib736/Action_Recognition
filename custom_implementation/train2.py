import torch
import torch.optim as optim
import sys
import numpy as np
import itertools
from models.test_ccnlstm import *
from dataloader.test_dataloader import *
from dataloader.test_dataloader_heatmap import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shutil
from tqdm import tqdm

from CNN_LSTM.convpooling_LSTM import *

def reset_log_path(path):
    if os.path.exists(path):
        # List all files and directories in the path
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                # If it is a file or a symlink, delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to UCF-101 dataset")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=224, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    path = 'runs/action_recognition_experiment'
    
    class_to_idx = {
        'moving_stuff': 0, 
        'picking_up_from_ground': 1, 
        'sitting_down_on_chair': 2, 
        'sitting_down_on_ground': 3,
        'sitting_on_chair': 4,
        'sitting_on_ground': 5,
        'standing': 6,
        'standing_up_from_chair': 7,
        'standing_up_from_ground': 8,
        'walking': 9
        }
    
    # class_to_idx = {
    #     'clearandjerk': 0, 
    #     'diving': 1, 
    #     'horseriding': 2, 
    #     'playingpiano': 3,
    #     'pushups': 4,
    #     'skiing': 5,
    #     'soccerjuggling': 6,
    #     'tennisswing': 7,
    #     'throwdiscus': 8,
    #     'walkingwithdog': 9
    #     }
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    reset_log_path(path)
    
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Setup TensorBoard
    writer = SummaryWriter(path)

    # Define training and test sets
    train_dataset = NumpyDataset(dataset_path=opt.dataset_path, sequence_length=opt.sequence_length, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    test_dataset = NumpyDataset(dataset_path=opt.dataset_path, sequence_length=opt.sequence_length, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)


    model = cnn_lstm(num_class=10).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def test_model(epoch):
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        all_labels = []
        all_predictions = []
        for X, y in tqdm(test_dataloader, desc="Testing", leave=False):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # model.lstm.reset_hidden_state()
                predictions = model(image_sequences)
                loss = cls_criterion(predictions, labels)
                acc = (predictions.argmax(1) == labels).float().mean()

            test_metrics["loss"].append(loss.item())
            test_metrics["acc"].append(acc.item())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.argmax(1).cpu().numpy())

        mean_loss = np.mean(test_metrics["loss"])
        mean_acc = np.mean(test_metrics["acc"])
        writer.add_scalar("Test/Loss", mean_loss, epoch+1)
        writer.add_scalar("Test/Accuracy", mean_acc*100, epoch+1)

        all_labels = [idx_to_class[label] for label in all_labels]
        all_predictions = [idx_to_class[pred] for pred in all_predictions]

        cm = confusion_matrix(all_labels, all_predictions, labels=list(idx_to_class.values()))
        model.train()
        
        return cm, acc

    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        model.train()
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{opt.num_epochs}", leave=True) as pbar:
            for X, y in train_dataloader:
                if X.size(0) == 1:
                    continue

                image_sequences = Variable(X.to(device), requires_grad=True)
                labels = Variable(y.to(device), requires_grad=False)

                optimizer.zero_grad()
                # model.lstm.reset_hidden_state()
                predictions = model(image_sequences)

                loss = cls_criterion(predictions, labels)
                acc = (predictions.argmax(1) == labels).float().mean()

                loss.backward()
                optimizer.step()

                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["acc"].append(acc.item())
                pbar.update()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            scheduler.step()
        # Log average metrics and a classification report at the end of the epoch
        mean_loss = np.mean(epoch_metrics["loss"])
        mean_acc = np.mean(epoch_metrics["acc"])
        writer.add_scalar("Train/Epoch Loss", mean_loss, epoch+1)
        writer.add_scalar("Train/Epoch Accuracy", mean_acc*100, epoch+1)

        # Evaluate on test set
        cm, acc = test_model(epoch)

    print(f'Final Test Batch Accuracy: {acc*100}')
    
    fig, ax = plt.subplots(figsize=(12, 10)) 

    sns.heatmap(cm, annot=True, fmt='g', cmap='viridis', ax=ax,
                xticklabels=sorted(idx_to_class.values()), 
                yticklabels=sorted(idx_to_class.values()))

    ax.set_title('Confusion Matrix', size=16)
    ax.set_xlabel('Predicted Labels', size=14)
    ax.set_ylabel('True Labels', size=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, size=10)

    fig.tight_layout()

    writer.add_figure('Confusion Matrix', fig, epoch+1)
    writer.close()
