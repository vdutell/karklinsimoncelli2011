import os
import numpy as np
import h5py
import glob
from scipy import io
from scipy import stats

class imageFile:
    def __init__(self,
                 imset,
                 patch_edge_size=None,
                 normalize_im=False,
                 patch_multiplier = 1,
                 normalize_patch=False,
                 invert_colors=False,
                 subset=500,
                 rand_state=np.random.RandomState()):
        
        # readin images
        self.images = self.extract_images(imset = imset,
                                          subset = subset)
        # process images
        self.images = self.process_images(self.images, patch_edge_size, normalize_im, 
                                          patch_multiplier, normalize_patch, invert_colors)

    def extract_images(self, imset, subset):
        #load in our images        
        #input is white noise
        if(imset=='whitenoise'):
            full_img_data = np.random.rand(10000,100,100)-0.5 #uniform in range -0.5 to 0.5
        #input is gaussian noise
        if(imset == 'gaussnoise'):
            full_img_data = np.random.randn(10000,100,100) #Gaussian about zero -0.5 to 0.5
        #input is van hateren with log normalization
        elif(imset=='vhlognorm'):
            self.image_files = '/home/vasha/datasets/vanHaterenNaturalImages/VanHaterenNaturalImagesCurated.h5'
            with h5py.File(self.image_files, "r") as f:
                full_img_data = np.array(f['van_hateren_good'], dtype=np.float32)
            full_img_data = full_img_data[0:subset,:,:]
        elif(imset=='kyoto'):
            self.image_files = '/home/vasha/datasets/eizaburo-doi-kyoto_natim-c2015ff/*.mat'
            bw_ims = []
            for file in sorted(glob.glob(self.image_files,recursive=True))[:subset]:
                mat = io.loadmat(file)
                #short medium and long activations
                sml_acts = np.array([mat['OS'],mat['OM'],mat['OL']])
                #mean over three for luminance
                bw_acts = np.mean(sml_acts,axis=0)
                # transpose if we need to ***I think this is allowed***
                if(np.shape(bw_acts)[0] > np.shape(bw_acts)[1]):
                    bw_acts = bw_acts.T
                bw_ims.append(np.array(bw_acts))
            full_img_data = np.array(bw_ims)
        elif('vh_' in imset):
            #using abberation corrected images
            if(imset=='vh_corr'):
                imdir = '/home/vasha/datasets/vanHaterenNaturalImages/pirsquared/vanhateren_imc/*.imc'
            #using abberation uncorrected images
            elif(imset=='vh_uncorr'):
                imdir = '/home/vasha/datasets/vanHaterenNaturalImages/pirsquared/vanhateren_iml/*.iml'
            dim = [1024,1536]
            full_img_data = []
            for file in sorted(glob.glob(imdir,recursive=True))[:subset]:
                dtype = np.dtype ('uint16').newbyteorder('>')
                a = np.fromfile(file, dtype).reshape(dim)
                full_img_data.append(np.array(a))
            full_img_data = np.array(full_img_data)

            #normalize each full image - divide by geom norm and log transform 
            invn = 1/np.prod([full_img_data.shape[1],full_img_data.shape[2]])
            geom_means = stats.mstats.gmean(full_img_data+1,axis=(1,2))[:,np.newaxis,np.newaxis]
            full_img_data = np.log(full_img_data+1) - np.log(geom_means)

        else:
            print('\"{}\" is an Unsupported Image Type!!!'.format(imset))
            

        return(full_img_data)

    def patch_maker(self, full_img_data, patch_edge_size, offset):
        (num_img, num_px_rows, num_px_cols) = full_img_data.shape
        #crop to patch rows
        if(num_px_rows % patch_edge_size != 0):
            nump = int(num_px_rows/patch_edge_size)
            full_img_data = full_img_data[:,:nump*patch_edge_size,:]
            (num_img, num_px_rows, num_px_cols) = full_img_data.shape
        #crop to patch cols
        if(num_px_cols % patch_edge_size != 0):
            nump = int(num_px_cols/patch_edge_size)
            full_img_data = full_img_data[:,:,:nump*patch_edge_size]
        (num_img, num_px_rows, num_px_cols) = full_img_data.shape
        num_img_px = num_px_rows * num_px_cols
        #calc number of patches & calculate them        
        self.num_patches = int(num_img_px / patch_edge_size**2)  
        
        data = np.asarray(np.split(full_img_data, int(num_px_cols/patch_edge_size),2)) # tile column-wise
        data = np.asarray(np.split(data, int(num_px_rows/patch_edge_size),2)) #tile row-wise
        data = np.transpose(np.reshape(np.transpose(data,(3,4,0,1,2)),(patch_edge_size,patch_edge_size,-1)),(2,0,1)) #stack tiles together
        return(data)
            
    def process_images(self, full_img_data, patch_edge_size=None, 
                       normalize_im=False, patch_multiplier = 1,
                       normalize_patch=False, invert_colors=False):
        if(normalize_im):
            print('normalizing full images...')
            full_img_data = full_img_data - np.mean(full_img_data,axis=(1,2),keepdims=True)
            full_img_data = full_img_data/np.std(full_img_data,axis=(1,2),keepdims=True)
        if(invert_colors):
            print('inverting colors...')
            full_img_data = full_img_data*(-1)
        if patch_edge_size is not None:
            print('sectioning into patches....')
            data = []
            if(patch_multiplier>1):
                print('multipying patches by {}...'.format(patch_multiplier))
            for i in range(patch_multiplier):
                offset_px = patch_edge_size/patch_multiplier # size in pixels of each offset
                data.append(np.array(self.patch_maker(full_img_data, patch_edge_size,offset=offset_px*i)))
            data = np.array(data)
            data = np.reshape(data,(-1,np.shape(data)[2],np.shape(data)[3]))
            print('now we have {} patches'.format(np.shape(data)[0]))
            self.num_patches = np.shape(data)[0]
        else:
            data = full_img_data
            self.num_patches = 1
        if(normalize_patch):
            print('normalizing patches...')
            data = data - np.mean(data,axis=(1,2),keepdims=True)
            data = data/np.std(data,axis=(1,2),keepdims=True)
        return data
        
        
#check for patchsize
def load_images(imset,
                patch_edge_size,
                normalize_im = False,
                patch_multiplier = 1,
                normalize_patch = False,
                invert_colors = False,
                start=0,
                subset = 500):

    print("Loading Natural Image Database...")
    vhimgs = imageFile(
            imset = imset,
            patch_edge_size = patch_edge_size,
            normalize_im = normalize_im,
            patch_multiplier = patch_multiplier,
            normalize_patch = False,
            invert_colors = False,
            subset = subset)
    print("Done Loading!")    
    np.random.shuffle(vhimgs.images)
    print("Done Shuffling!")
    
    vhimgs = vhimgs.images
    
    #params of images
    imxlen = len(vhimgs[0,0,:])
    imylen = len(vhimgs[0,:,0])
    nimages = len(vhimgs[:,0,0])
    
    return(vhimgs, nimages)
