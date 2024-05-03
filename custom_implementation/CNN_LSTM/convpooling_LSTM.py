import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet18_Weights

def modify_resnet_input_channels(model, new_input_channels=14):
        original_first_layer = model.conv1
        new_first_layer = nn.Conv2d(new_input_channels,
                                    original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size,
                                    stride=original_first_layer.stride,
                                    padding=original_first_layer.padding,
                                    bias=original_first_layer.bias)

        # Here we clone the weights of the first three channels to the rest
        with torch.no_grad():
            with torch.no_grad():
                new_first_layer.weight[:,:] = torch.mean(original_first_layer.weight, dim=1, 
                                                         keepdim=True).repeat(1, 27, 1, 1)

        model.conv1 = new_first_layer
        return model
    
class cnn_lstm(nn.Module):

    def __init__(self, num_class):
        super(cnn_lstm, self).__init__()

        resnet_model = torchvision.models.resnet18() #weights=ResNet18_Weights.DEFAULT
        modified_resnet = modify_resnet_input_channels(resnet_model, 27)
        self.conv = nn.Sequential(*list(modified_resnet.children())[:-1])
        self.lstm = nn.LSTM(2048,512,5,batch_first=True)
        self.fc=nn.Linear(512,num_class)

    def forward(self, x):
        print("Initial input shape:", x.shape)
        t_len = x.size(2)
        conv_out_list = []
        for i in range(t_len):
            frame_output = torch.squeeze(x[:, :, i, :, :])
            print(f"Shape after squeezing frame {i}:", frame_output.shape)
            conv_output = self.conv(frame_output)
            print(f"Conv output shape for frame {i}:", conv_output.shape)
            conv_out_list.append(conv_output)
        conv_out = torch.stack(conv_out_list, 1)
        print("Stacked conv output shape:", conv_out.shape)
        
        # Reshaping for LSTM
        lstm_input = conv_out.view(conv_out.size(0), conv_out.size(1), -1)
        print("Shape before LSTM:", lstm_input.shape)
        conv_out, hidden = self.lstm(lstm_input)
        print("Output shape from LSTM:", conv_out.shape)
        
        lstm_out = []
        for j in range(conv_out.size(1)):
            lstm_frame_output = self.fc(torch.squeeze(conv_out[:, j, :]))
            print(f"FC output shape at step {j}:", lstm_frame_output.shape)
            lstm_out.append(lstm_frame_output)
        
        output = torch.stack(lstm_out, 1)
        output = torch.mean(output, dim=1)
        print("Final output shape:", output.shape)
        return output

    
def main():

    # Create the models
    model = cnn_lstm(num_class=10)

    random_input1 = torch.randn(2, 27, 30, 192, 256)
    outputs = model(random_input1)
    print("Model Output0 Shape:", outputs.shape)

if __name__ == "__main__":
    main()