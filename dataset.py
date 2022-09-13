from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
from transforms import discriminative_transforms
import math
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class DiscriminativeDataset(Dataset):
    """ Discriminative dataset contains  N patches of taken from a set of image X. 
        Each patches is transformed K different time to obtain a dataset of size N*K.
        See Disciminative Unsupervised Feature Learning with Convolutional Neural
        Networks(A.Dosovitskiy,2014)"""
    def __init__(self,name,root,num_samples,patch_size,dataset_len):
        self.num_samples = num_samples
        self.dataset_len = dataset_len
        if (name == 'cifar10'):
            dict = unpickle(root) 
            images = dict[b'data'] # contains a 10000x3072 numpy array (uint8)
            self.images = np.reshape(images,(self.dataset_len,3,patch_size,patch_size))
            self.images = np.moveaxis(self.images,1,-1)
        elif(name == 'stl10'):
            with open(root, 'rb') as f:
                everything = np.fromfile(f, dtype=np.uint8)
                images = np.reshape(everything, (-1, 3, 96, 96))
                self.images = np.transpose(images, (0, 3, 2, 1))
                print(self.images.shape)
        
        self.patches = torch.empty((self.dataset_len,self.num_samples,3,patch_size,patch_size))
        self.patch_created = torch.zeros(self.patches.shape[0])
    
    def __len__(self):
        return self.dataset_len*self.num_samples

    def __getitem__(self,idx):
        """Each id refers to a specific patch, each image is a class"""
        # print(self.num_samples,idx,idx%self.num_samples)
        image_id = math.floor(idx / self.num_samples)
        patch_id = idx % self.num_samples
        if(self.patch_created[image_id] == 0):
            self.patches[image_id,:,:,:] = discriminative_transforms(self.images[image_id,:,:,:],self.num_samples)
            self.patch_created[image_id] = 1
        patch = self.patches[image_id,patch_id]
        return patch,image_id

