import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, input_channel, output_channel, padding=0, drop=0.01, dilated_conv=False,
                 depth_sep_conv=False):

        super(Block, self).__init__()
        if depth_sep_conv:
            '''
            Depthwise Separable Convolution
            '''
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channel, input_channel, 3, groups=input_channel, padding=padding),
                nn.Conv2d(input_channel, output_channel, 1)
            )
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=padding)

        self.n1 = nn.BatchNorm2d(output_channel)
        self.drop1 = nn.Dropout2d(drop)

        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)
        self.n2 = nn.BatchNorm2d(output_channel)
        self.drop2 = nn.Dropout2d(drop)

        if dilated_conv:
            '''
            Dilated Convolution
            '''
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=padding + 1, dilation=2)
        else:
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=padding, stride=2)

        self.n3 = nn.BatchNorm2d(output_channel)
        self.drop3 = nn.Dropout2d(drop)


        '''
        Depending on the model requirement, Convolution block with number of layers is applied to the input image
        '''

    def __call__(self, x, layers=2):

        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.drop1(x)

        if layers >= 2:
            x = self.conv2(x)

            x = self.n2(x)
            x = F.relu(x)
            x = self.drop2(x)

        if layers == 3:
            x = self.conv3(x)
            x = self.n3(x)
            x = F.relu(x)
            x = self.drop3(x)

        return x


class Net(nn.Module):
    def __init__(self, base_channels, drop=0.01):

        super(Net, self).__init__()

        '''
        In the first convolution block, the Dilated Convolution is applied (3rd layer) to increase receptive field.
        '''
        self.block1 = Block(3, base_channels, padding=1, drop=drop, dilated_conv=True)

        ''' RF: 1 => 3 > 5 > 9 '''


        '''
        In the second convolution block, the Depthwise Separable Convolution is applied (1st layer) to reduce param count.
        '''
        self.block2 = Block(base_channels, base_channels * 2, padding=1, drop=drop, dilated_conv=False,
                            depth_sep_conv=True)
        ''' RF: 9 => 11 > 13 > 15 '''


        self.block3 = Block(base_channels * 2, base_channels * 4, padding=1, drop=drop, dilated_conv=False,
                            depth_sep_conv=True)
        ''' RF: 15 => 19 > 23 > 27 '''

        self.block4 = Block(base_channels * 4, base_channels * 2, padding=1, drop=drop, dilated_conv=False)
        ''' RF: 27 => 35 > 43 '''

        self.block4LastLayer = nn.Conv2d(base_channels * 2, base_channels * 2, 3)
        ''' RF: 43 => 51 '''

        self.gap = nn.AvgPool2d(6)
        self.linear = nn.Conv2d(base_channels * 2, 10, 1)


    def forward(self, x):
        x = self.block1(x, layers=3)

        x = self.block2(x, layers=3)

        x = self.block3(x, layers=3)

        x = self.block4(x, layers=2)
        x = self.block4LastLayer(x)

        x = self.gap(x)

        x = self.linear(x)

        x = x.view(x.size(0), 10)

        return F.log_softmax(x, dim=1)