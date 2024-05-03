import torch
import sys
import numpy as np
import itertools
from models.test_model import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime

from dataloader.test_dataloader import *
from dataloader.test_dataloader_heatmap import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

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
    parser.add_argument("--num_epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=224, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    path = 'runs/action_recognition_experiment'
    reset_log_path(path)
    
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
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)
    
    writer = SummaryWriter(path)

    train_dataset = NumpyDataset(dataset_path=opt.dataset_path, sequence_length=opt.sequence_length, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    test_dataset = NumpyDataset(dataset_path=opt.dataset_path, sequence_length=opt.sequence_length, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)

    # Define network
    model = ConvLSTM(
        num_classes=10,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def test_model(epoch):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        all_labels = []
        all_predictions = []
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # Reset LSTM hidden state
                model.lstm.reset_hidden_state()
                # Get sequence predictions
                predictions = model(image_sequences)
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = cls_criterion(predictions, labels).item()
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.argmax(1).cpu().numpy())
            
            # Log test performance
            writer.add_scalar("Test/Loss", np.mean(test_metrics["loss"]), epoch+1)
            writer.add_scalar("Test/Accuracy", np.mean(test_metrics["acc"]), epoch+1)
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
        all_labels = [idx_to_class[label] for label in all_labels]
        all_predictions = [idx_to_class[pred] for pred in all_predictions]
        cm = confusion_matrix(all_labels, all_predictions, labels=list(idx_to_class.values()))
            
        model.train()
        print("")
        
        return acc, cm

    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch} ---")
        for batch_i, (X, y) in enumerate(train_dataloader):

            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            # Get sequence predictions
            predictions = model(image_sequences)

            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            writer.add_scalar("Train/Loss", np.mean(epoch_metrics["loss"]), epoch+1)
            writer.add_scalar("Train/Accuracy", np.mean(epoch_metrics["acc"]), epoch+1)
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    time_left,
                )
            )

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate the model on the test set
        acc, cm = test_model(epoch)
        
    print(f'Final Test Batch Accuracy: {acc}')
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
