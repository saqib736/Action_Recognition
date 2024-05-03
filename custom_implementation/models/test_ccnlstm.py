import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet152
from torchvision.models.resnet import ResNet152_Weights

##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = resnet152(weights=ResNet152_Weights.DEFAULT) 
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

# class Encoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Encoder, self).__init__()
#         # Load pre-trained ResNet-152
#         resnet = resnet152(weights=ResNet152_Weights.DEFAULT)

#         # Modify the first convolution layer to accept 27 input channels
#         original_first_layer = resnet.conv1
#         new_first_layer = nn.Conv2d(27, 
#                                     original_first_layer.out_channels, 
#                                     kernel_size=original_first_layer.kernel_size, 
#                                     stride=original_first_layer.stride, 
#                                     padding=original_first_layer.padding, 
#                                     bias=False)

#         # Compute the mean of the weights across the RGB channels and replicate across all 27 channels
#         with torch.no_grad():
#             new_first_layer.weight[:,:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True).repeat(1, 27, 1, 1)

#         # Replace the first conv layer
#         resnet.conv1 = new_first_layer

#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#         self.final = nn.Sequential(
#             nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
#         )

#     def forward(self, x):
#         with torch.no_grad():
#             x = self.feature_extractor(x)  # Extract features without gradient accumulation
#         x = x.view(x.size(0), -1)  # Flatten the features
#         return self.final(x)  # Pass through the final layer to get latent dimensions



##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


##############################
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


##############################
#         ConvLSTM
##############################


class ConvLSTM(nn.Module):
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)


def main():
    model = ConvLSTM(
        num_classes=10,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
    random_input1 = torch.randn(2, 30, 3, 192, 256)
    outputs = model(random_input1)
    print("Model Output Shape:", outputs.shape)

if __name__ == "__main__":
    main()