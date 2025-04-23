import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import time 
import os
import scipy.io
from scipy.ndimage.interpolation import rotate
from skimage.color import rgb2gray
from multiprocessing import Pool
from PIL import Image

###calcuate std

def load_grayscale_images_from_directory(directory, expected_shape=(704, 704, 1), extension='.tiff'):
    """
    Returns:
        list: List of grayscale image arrays with shape (704, 704, 1) in [0, 1].
    """
    image_files = [f for f in os.listdir(directory) if f.endswith(extension)]
    image_files.sort()  # Ensure consistent ordering
    images = []
    for file in image_files:
        img_path = os.path.join(directory, file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img, dtype=np.float32)  # Shape: (704, 704)
        # Add channel dimension to match (704, 704, 1)
        img_array = img_array[..., np.newaxis]
        # Verify shape
        if img_array.shape != expected_shape:
            raise ValueError(f"Image {file} has shape {img_array.shape}, expected {expected_shape}")
        # Normalize to [0, 1]
        #img_array /= 255.0
        images.append(img_array)
    return images

def calculate_noise_std_grayscale(noisy_dir, gt_dir,  out_path='noise_std.npy',extension='.png'):

    noisy_images = load_grayscale_images_from_directory(noisy_dir, extension)
    gt_images = load_grayscale_images_from_directory(gt_dir, extension)
    
    # Ensure the number of images matches
    assert len(noisy_images) == len(gt_images), "Mismatch in number of noisy and GT images"
    
    # Verify that all images are grayscale (2D arrays)
    assert all(len(img.shape) == 2 for img in noisy_images), "Noisy images must be grayscale"
    assert all(len(img.shape) == 2 for img in gt_images), "GT images must be grayscale"
    
    # Stack images into arrays with shape (N, H, W)
    noisy_images = np.stack(noisy_images, axis=0)
    gt_images = np.stack(gt_images, axis=0)
    noisy_images /= 255.0
    gt_images    /= 255.0

    # Compute noise as the difference between noisy and ground truth images
    noise = noisy_images - gt_images
    
    # Flatten the noise array and compute the standard deviation
    noise_flat = noise.flatten()
    std = float(np.std(noise_flat))
    np.save(out_path, std)
    return std




def process_image(train_noisy):
    STD_train = []
    for h in range(3,train_noisy.shape[1]-3):
        for w in range(3,train_noisy.shape[2]-3):
            STD_train.append(np.std((train_noisy[:,h-3:h+3,w-3:w+3,:]/255).reshape([-1,36,1]),1).reshape([-1,1,1]))   
    return np.mean(np.concatenate(STD_train,1),1)

def horizontal_flip(image, rate=0.5):
    image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    image = image[::-1, :, :]
    return image

def random_rotation(image, angle):
    h, w, _ = image.shape
    image = rotate(image, angle)
    return image
    
class benchmark_data(Dataset):

    def __init__(self, data_dir, task, transform=None):
        start_time = time.time()
        self.task = task
        self.data_dir = data_dir
        files_tmp = open(self.data_dir+'Scene_Instances.txt','r').readlines()

        self.Validation_Gt = scipy.io.loadmat(self.data_dir+'ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
        self.Validation_Noisy = scipy.io.loadmat(self.data_dir+'ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb']

        ##check SIDD grayscale
        self.Validation_Gt = np.array([rgb2gray(img) for img in self.Validation_Gt])
        self.Validation_Noisy = np.array([rgb2gray(img) for img in self.Validation_Noisy])
        self.Validation_Gt = self.Validation_Gt[..., np.newaxis]
        self.Validation_Noisy = self.Validation_Noisy[..., np.newaxis]



        # print('SIDD dataset')
        # print(self.Validation_Gt.shape)
        # end_time = time.time()
        # load_time = end_time - start_time
        # print(f"Time to load dataset: {load_time:.2f} seconds")
        # assert 0
        self.Validation_Gt = self.Validation_Gt.reshape([-1,256,256,1])
        self.Validation_Noisy = self.Validation_Noisy.reshape([-1,256,256,1])
        self.data_num = self.Validation_Noisy.shape[0]


        
        self.files = []
        for i in range(160):
            f = files_tmp[i].split("\n")[0]
            #if f[-1]=='N':
            if i >=0:
               self.files.append(f)   
               #print(f)
        self.indices = self._indices_generator()
        self.patch_size = 40
        

        

        
        if os.path.exists(self.data_dir+'/'+'std.npy'):
           STD = np.load(self.data_dir+'/'+'std.npy')


        else:
           STD = process_image(self.Validation_Noisy)
           np.save(self.data_dir+'/'+'std.npy',STD)
        self.std = STD
        
              

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):

        def data_loader():
            
            if self.task=="test": 
               Img_noisy = self.Validation_Noisy[index]
               Img_GT = self.Validation_Gt[index]
               Img_noisy = (np.transpose(Img_noisy,(2, 0, 1)))
               Img_GT = (np.transpose(Img_GT,(2, 0, 1)))   
               std = self.std[index]
               #
                #self.Validation_Gt = Image.open('data_dir')
                #self.Validation_Noisy = Image.open('data_dir') #directory load / file name 
               

            if self.task=="train":
               Img_noisy = self.Validation_Noisy[index]
               Img_GT = self.Validation_Gt[index]
               
               # Augmentation
               horizontal = torch.randint(0,2, (1,))
               vertical = torch.randint(0,2, (1,))
               rand_rot = torch.randint(0,4, (1,))
               rot = [0,90,180,270]
               if horizontal ==1:
                  Img_noisy = horizontal_flip(Img_noisy)
                  Img_GT = horizontal_flip(Img_GT)
               if vertical ==1:
                 Img_noisy = vertical_flip(Img_noisy)
                 Img_GT = vertical_flip(Img_GT)        
               Img_noisy = random_rotation(Img_noisy,rot[rand_rot])
               Img_GT = random_rotation(Img_GT,rot[rand_rot])         
                 
               Img_noisy = (np.transpose(Img_noisy,(2, 0, 1)))
               Img_GT = (np.transpose(Img_GT,(2, 0, 1)))
               std = self.std[index]
               x_00 = torch.randint(0, Img_noisy.shape[1] - self.patch_size, (1,))
               y_00 = torch.randint(0, Img_noisy.shape[2] - self.patch_size, (1,))
               Img_noisy = Img_noisy[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]
               Img_GT = Img_GT[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]

 


            
            return np.array(Img_noisy, dtype=np.float32), np.array(Img_GT, dtype=np.float32),  np.array(std, dtype=np.float32), index #,Img_train, Img_train_noisy, std_train[0]


        def _timeprint(isprint, name, prevtime):
            if isprint:
                print('loading {} takes {} secs'.format(name, time() - prevtime))
            return time()

        if torch.is_tensor(index):
            index = index.tolist()

        input_noisy, input_GT, std, idx = data_loader()
        target = {
                  'dir_idx': str(idx)
                  }

        return target, input_noisy, input_GT, std 

    def _indices_generator(self):

        return np.arange(self.data_num,dtype=int)
   
        

if __name__ == "__main__":
    time_print = True

    prev = time()
