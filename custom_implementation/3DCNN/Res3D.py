import torch
import torch.nn as nn
import torch.nn.init as init

class ResBlock(nn.Module):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel,kernel_size=(3,3,3),stride=(temporal_stride,spatial_stride,spatial_stride),padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channel, out_channel,kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=(temporal_stride,spatial_stride,spatial_stride),bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res3D(nn.Module):
    # Input size: 8x224x224
    def __init__(self, num_class):
        super(Res3D, self).__init__()

        self.conv1=nn.Conv3d(3,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(ResBlock(64,64,spatial_stride=2),
                                 ResBlock(64, 64))
        self.conv3=nn.Sequential(ResBlock(64,128,spatial_stride=2,temporal_stride=2),
                                 ResBlock(128, 128))
        self.conv4 = nn.Sequential(ResBlock(128, 256, spatial_stride=2,temporal_stride=2),
                                   ResBlock(256, 256))
        self.conv5 = nn.Sequential(ResBlock(256, 512, spatial_stride=2,temporal_stride=2),
                                   ResBlock(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,6,8))
        self.linear=nn.Linear(2048,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        print("avg_pool:", x.shape)
        x = x.view(x.size(0),-1)
        return self.linear(x)
    
############################################################# TESTING ####################################################################
def main():
    model = Res3D(num_class=10)
    random_input = torch.randn(4, 3, 30, 240, 320) 
    outputs = model(random_input)
    print("Model Output Shape:", outputs.shape)

if __name__ == "__main__":
    main()