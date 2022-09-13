import torchvision
import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
from torchmetrics.functional import image_gradients
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PatchExtractor(torch.nn.Module):
    """Extracts a random patch from an image with composite transformations"""
    def __init__(self,img,patch_size=32, min_grad=3, cifar=False):
        """ img: a PIL image"""
        self.patch_size = patch_size
        self.min_grad = min_grad
        self.max_scale = 1.4
        self.min_scale = 0.7
        self.max_translate = 0.2

        if cifar:
            self.img = Image.Image.resize(Image.fromarray(img),(64,64))
        else:
            self.img = img
        while True:
            i = 0
            self.top, self.left = self.get_patch_coordinates()
            if self.check_gradient(np.asarray(self.img)[self.top:self.top+self.patch_size,self.left:self.patch_size+self.left])\
                or i>10:
                break
            i+=1
    
    def get_initial_patch(self):
        print(F"Original Shape is {self.top},{self.left},{self.patch_size}")
        return TF.crop(self.img,self.top,self.left,self.patch_size,self.patch_size)
    
    def get_patch_coordinates(self):
        # Within maximum scales and translation values to prevent padding
        min_top = int(self.max_translate * (self.patch_size*self.max_scale))
        image_height = self.img.size[0]
        max_top = image_height - int((self.max_translate+1) * self.max_scale*self.patch_size)
        top = random.randint(min_top,max_top) # can translate by 0.2 * patch_size, prevents padding

        min_left = min_top
        image_width = self.img.size[1]
        max_left = image_width - int((self.max_translate+1) * self.max_scale*self.patch_size)
        left = random.randint(min_left,max_left)
        return top, left
    
    def generate_parameter_vector(self):
        translate_vertical = int(random.uniform(-0.2,0.2)*self.patch_size)
        translate_horizontal = int(random.uniform(-0.2,0.2)*self.patch_size)

        scale_factor = random.uniform(0.7,1.4)

        rotation_degrees = random.uniform(-20,20)

        contrast_one_factor_one = random.uniform(0.5,2)
        contrast_one_factor_two = random.uniform(0.5,2)
        contrast_one_factor_three = random.uniform(0.5,2)

        contrast_two_power = random.uniform(0.25,4)
        contrast_two_multiply = random.uniform(0.7,1.4)
        contrast_two_additive = random.uniform(-0.1,0.1)

        color_values = random.uniform(-0.1,0.1)
        return [translate_vertical,translate_horizontal,scale_factor,rotation_degrees,contrast_one_factor_one,
            contrast_one_factor_two,contrast_one_factor_three, contrast_two_power,contrast_two_multiply,contrast_two_additive,
            color_values]
    
    def __call__(self,parameters):
        #TODO: random order?
        top,left,patch_size,img = self.top,self.left,self.patch_size,self.img
        top,left = self.translate(parameters[0],parameters[1],self.top,self.left)
        patch_size,top,left = self.scale(parameters[2],self.patch_size,top,left)
        # print(f"Patch size is {patch_size} and coordinates are {top},{left}")
        img = self.rotate(self.img,parameters[3])
        img = self.contrast_one(img,parameters[4:7])
        img = self.contrast_two(img,power_increase=parameters[7],factor=parameters[8],additive_increase=parameters[9])
        img = self.color(img,parameters[10])
        patch =  TF.crop(img,top,left,patch_size,patch_size)
        return Image.Image.resize(patch,(32,32))
    
    def check_gradient(self,patch):
        """Used to verify if image has sufficient gradient (edges)
        """
        dx, dy = image_gradients(torch.tensor(patch).unsqueeze(0))
        grad = torch.mean(torch.sqrt(torch.pow(dx,2) + torch.pow(dy,2)))
        return grad >= self.min_grad
    
    def translate(self,vertical,horizontal,top,left):
        top += vertical
        left += horizontal
        return top,left
    
    def scale(self,scale,patch_size,top,left):
        patch_size =int(self.patch_size*scale)
        top = int(self.top * scale/2)
        left = int(self.left * scale/2)
        return patch_size,top,left
    
    def rotate(self,img,degrees):
        #TODO: rotations gives padding
        return Image.Image.rotate(img,degrees) # rotates the img instead of the patch, yields same result
    
    def contrast_one(self,img,factors,n_components=3):
        img_array = np.asarray(img)/255
        img_centered = img_array - np.mean(img_array,axis=0)
        # Reshape image into matrix where each row is a vector of the RGB channels
        X = np.reshape(img_centered,(img_array.shape[0]*img_array.shape[1],img_array.shape[2]))
        # Principal components on the set of all pixels
        pca = PCA(n_components)
        pca.fit(X)
        patch = img_array[self.left:self.left+self.patch_size,self.top:self.top+self.patch_size,:]
        # Projection of each patch pixel onto the principal components of the set of all pixels
        X_patch = np.reshape(patch,(patch.shape[0]*patch.shape[1],patch.shape[2])) # flatten so that each pixel is a vector
        transformed_patch = pca.transform(X_patch) # [?,n_components]
        for i in range(0,n_components):
            transformed_patch[:,i] *= factors[i]
        new_patch = pca.inverse_transform(transformed_patch)
        new_patch = np.reshape(new_patch,patch.shape)
        img_array[self.left:self.left+self.patch_size,self.top:self.top+self.patch_size,:] = new_patch
        return Image.fromarray(np.uint8(img_array*255))

    def contrast_two(self,img,power_increase, factor, additive_increase):
        hsv_image = np.asarray(img.convert('HSV'))/255
        hsv_image[:,:,1:3] = np.power(hsv_image[:,:,1:3],np.ones((hsv_image.shape[0],hsv_image.shape[1],2))*power_increase)
        hsv_image[:,:,1:3] = hsv_image[:,:,1:3]*factor
        hsv_image[:,:,1:3] += additive_increase

        return Image.fromarray(np.uint8(hsv_image*255)).convert('RGB')
    
    def color(self,img,value):
        hsv_image = np.asarray(img.convert('HSV'))/255
        hsv_image[:,:,0] += value
        return Image.fromarray(np.uint8(hsv_image*255)).convert('RGB')
    
    def visualize_patches(self,patches):
        rows = 4
        cols = 3
        fig = plt.figure(figsize=(rows,cols))
        for i in range(1,10):
            fig.add_subplot(rows,cols,i)
            plt.imshow(torch.moveaxis(patches[i,:,:,:],0,-1).type(torch.uint8))
            plt.axis('off')
        # Original patch
        fig.add_subplot(rows,cols,10)
        plt.imshow(self.get_initial_patch())
        plt.axis('off')
        plt.title('Original Patch',fontsize=7,y=-0.4)
        # Original Image
        fig.add_subplot(rows,cols,11)
        plt.imshow(self.img)
        plt.axis('off')
        plt.title('Input Image',fontsize=7,y=-0.4)
        plt.savefig('patch_visualize.png')


# Discriminative Unsupervised Feature Learning with Convolutional Neural Networks (2014)
def discriminative_transforms(img,num_samples=100):
    ''' translation, scaling, rotation, contrast 1, contrast 2, color
        Expects the input img to be a numpy array of shape [w,h,c] of type uint8
    '''
    patch_extractor = PatchExtractor(img,cifar=True) # Gets the initial patch from image
    transformed_patches = torch.empty((num_samples,3,32,32))
    for i in range(0,num_samples):
        parameter_vector = patch_extractor.generate_parameter_vector()
        patch = np.asarray(patch_extractor(parameter_vector))
        tensor_patch = torch.from_numpy(patch).moveaxis(-1,0)
        transformed_patches[i,:,:,:] = tensor_patch
        
    # Subtract the mean of each pixel over the whole resulting dataset
    transformed_patches = transformed_patches - torch.mean(transformed_patches)
    # patch_extractor.visualize_patches(transformed_patches[:10,:,:,:])
    return transformed_patches



