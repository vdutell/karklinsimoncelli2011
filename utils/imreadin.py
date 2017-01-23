import os
import numpy as np
import h5py

#def extract_images(filename):
    #function from Dylan
#    with h5py.File(filename, "r") as f:
        #full_img_data = np.array(f['van_hateren_good'], dtype=np.float32)
    #return full_img_data

class vanHateren:
    def __init__(self,
                 img_dir,
                 patch_edge_size=None,
                 normalize=False,
                 rand_state=np.random.RandomState()):
        self.images = self.extract_images(img_dir, patch_edge_size, normalize)

    """
    adapted from Dylan Payton's code for Sparse coding here: https://github.com/dpaiton/FeedbackLCA/blob/master/data/input_data.py
    load in van hateren dataset
    if patch_edge_size is specified, rebuild data array to be of sequential image
    patches.
    if preprocess is true, subtract mean from each full-size image, and rescale image variance to 1
    Note: in K&S2011, methods report input images' piel values were 'linear with respect to light intensity'
    I'm not sure if this is true for the VH images we are using, and how to properly normalize for this if not.
    """

    def extract_images(self, filename, patch_edge_size=None, normalize=False):
        with h5py.File(filename, "r") as f:
            full_img_data = np.array(f['van_hateren_good'], dtype=np.float32)
            if(normalize):
                print('normalizing...')
                full_img_data = full_img_data - np.mean(full_img_data,axis=(1,2),keepdims=True)
                full_img_data = full_img_data/np.std(full_img_data,axis=(1,2),keepdims=True)
            if patch_edge_size is not None:
                print('sectioning into patches....')
                print(full_img_data.shape)
                (num_img, num_px_rows, num_px_cols) = full_img_data.shape
                num_img_px = num_px_rows * num_px_cols
                #print(num_px_rows)
                #print(patch_edge_size)
                #print(num_px_rows % patch_edge_size)
                assert (num_px_rows % patch_edge_size == 0)  , ("The number of image row edge pixels % the patch edge size must be 0.")
                assert (num_px_cols % patch_edge_size == 0)  , ("The number of image column edge pixels % the patch edge size must be 0.")
                self.num_patches = int(num_img_px / patch_edge_size**2)
                full_img_data = np.reshape(full_img_data, (num_img, num_img_px))
                #data = np.vstack([full_img_data[idx,...].reshape(self.num_patches, patch_edge_size, patch_edge_size) for idx in range(num_img)])
                data = np.asarray(np.split(full_img_data, num_px_cols/patch_edge_size,2)) # tile column-wise
                data = np.asarray(np.split(data, num_px_rows/patch_edge_size,2)) #tile row-wise
                data = np.reshape(np.transpose(data,(3,4,0,1,2)),(patch_edge_size,patch_edge_size,-1)) #stack tiles together

            else:
                data = full_img_data
                self.num_patches = 0
            return data
