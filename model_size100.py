import torch.nn as nn

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 16, kernel_size=4)
        self.max2d1 = nn.MaxPool2d(2, stride=2)
        self.conv2d2 = nn.Conv2d(16, 16, kernel_size=4)
        self.max2d2 = nn.MaxPool2d(2, stride=2)
        self.conv2d3 = nn.Conv2d(16, 16, kernel_size=4)
        self.max2d3 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(1296, 500) 
        self.fc2 = nn.Linear(500, 10) 
        self.fc3 = nn.Linear(10, 5) 


    def forward_once(self, x):
        #print x
        x = self.conv2d1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.max2d1(x)
        x = self.conv2d2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.max2d2(x)
        x = self.conv2d3(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.max2d3(x)
        #print x.size()
        x = x.view(x.size()[0], -1)
        #print x.size()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def forward_test(self, x):
        #print x
        x = self.conv2d1(x)
        #x = nn.ReLU(inplace=True)(x)
        #x = self.max2d1(x)
        #x = self.conv2d2(x)
        #x = nn.ReLU(inplace=True)(x)
        #x = self.max2d2(x)
        #x = self.conv2d3(x)
        #x = nn.ReLU(inplace=True)(x)
        #x = self.max2d3(x)
        #print x.size()
        #x = x.view(x.size()[0], -1)
        #print x.size()
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        return x

    def forward(self, input1, input2, input3):
        output1 = self.forward_test(input1)
        output2 = self.forward_test(input2)
        output3 = self.forward_test(input3)
        return output1, output2, output3