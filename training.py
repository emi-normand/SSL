import torch
import torchvision
from dataset import DiscriminativeDataset
from transforms import discriminative_transforms
from torch.utils.data import DataLoader
from models import SmallCNN
from torch.utils.tensorboard import SummaryWriter


# Discriminative Unsupervised Feature Leargning with Convolutional Neural Networks (2014)
def train(model,dataloader,num_samples,epoch=100,n_surrogate_classes=8000):
    criterion = torch.nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(),0.001)
    tensorboard = SummaryWriter()

    for i in range(epoch):
        n = 0
        for id,batch in enumerate(dataloader):
            patches,classes = batch
            
            # preprocess
            input = patches.to(device=torch.device('mps'),dtype=torch.float32)/255 # convert to floating point
            opt.zero_grad()
            pred = model(input)
            loss = criterion(pred,classes.to(device=torch.device('mps')))  
            tensorboard.add_scalar('training loss',loss,n*(i+1)) 
            if (n % 100 == 0):
                tensorboard.add_images('input',input)    
            loss.backward()
            opt.step()
            print(n)
            if(n>(n_surrogate_classes*num_samples/batch_size)):
                break
            else:
                n+=1
        print(F"Finished epoch {i}")

batch_size=8
num_samples=100
# Get dataset
# cifar_10 = torchvision.datasets.CIFAR10(download=True,transform=discriminative_transforms,root='cifar10')
# dataset = DiscriminativeDataset('cifar10','cifar-10-batches-py/data_batch_1',num_samples,32,10000)
dataset = DiscriminativeDataset('stl10','stl10_binary/unlabeled_X.bin',num_samples,32,10000)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
model = SmallCNN(8000).to(device=torch.device('mps'))
train(model,dataloader,num_samples)