import torch
import torchvision

# ExemplarCNN, replaced max pooling by stride=2 as is standard now
class SmallCNN(torch.nn.Module):
    def __init__(self,num_classes):
        super(SmallCNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,64,(5,5),2)
        self.conv2 = torch.nn.Conv2d(64,128,(5,5),2)
        self.conv3 = torch.nn.Conv2d(128,256,(5,5),2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(256,num_classes)
        
        # print(self.model)
        self.softmax = torch.nn.LogSoftmax(dim=-1) # apparently this is better with NLLLoss

    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x