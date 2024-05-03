import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.Res3D import Res3D
# from models.cnv_lstm import ConvLSTM
# from models.cnn_lstm import ConvLSTM
from models.test_ccnlstm import ConvLSTM
from models.rgbposec3d import resnet50

from dataloader.temporal_heatmap_dataset import TemporalHeatmapDataset
# from dataloader.joint_limb_dataload import TemporalHeatmapDataset, collate_fn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

def main():
    
    path = 'runs/action_recognition_experiment'
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
                   
    writer = SummaryWriter(path)
    
    # Dataset and DataLoader setup
    dataset_path = 'ufc_dataset'
    train_dataset = TemporalHeatmapDataset(dataset_path, split='train', train_size=370, sequence_length=60, seed=42)
    test_dataset = TemporalHeatmapDataset(dataset_path, split='test', train_size=370, sequence_length=60, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    # Model, optimizer, and scheduler setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConvLSTM(
        num_classes=10,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    ).to(device)
    # model = Res3D(num_class=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # class_counts = [9, 8, 9, 6, 26, 6, 10, 9, 6, 19]
    # class_weights = torch.tensor([1.0 / x for x in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss().to(device) # weight=class_weights
  
    model.train()
    for epoch in range(20):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100, leave=True, file=sys.stdout) as t:
            for i, (inputs_joints, labels) in enumerate(t):
                inputs_joints, labels = inputs_joints.to(device), labels.to(device)
                optimizer.zero_grad()
                model.lstm.reset_hidden_state()
                outputs = model(inputs_joints)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                writer.add_scalar('Training loss/Step', loss.item(), i + epoch * len(t))
                t.set_postfix(loss=loss.item())
            
            scheduler.step() 
            epoch_loss /= len(train_dataset)
            writer.add_scalar('Training loss/Epoch', epoch_loss, epoch)

    # Testing phase
    true_labels = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs_joints, labels in test_loader:
            inputs_joints, labels = inputs_joints.to(device), labels.to(device)
            model.lstm.reset_hidden_state()
            outputs = model(inputs_joints)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    # Metrics and results
    print("Classification Report:")
    print(classification_report(true_labels, predictions, zero_division=0))
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    writer.add_figure('Confusion Matrix', fig)
    accuracy = 100 * sum(1 for x, y in zip(predictions, true_labels) if x == y) / len(true_labels)
    print(f'Overall Test Accuracy: {accuracy:.2f}%')
    writer.close()

if __name__ == "__main__":
    main()
