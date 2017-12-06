import numpy as np
import tensorflow as tf

#vector version of reshape

#change from batch by 1d vector form to batch by 2 dim image form
def vecs2ims(imvec,patchparm):
    nims = np.shape(imvec)[0]
    patchsize = patchparm
    return(imvec.reshape((nims,patchsize,patchsize)))
           
#change from batch by 2 dim image form to  batch by 1d vector form
def ims2vecs(ims,patchparm):
    nims = np.shape(ims)[0]
    patchvecsize = patchparm**2
    return(ims.reshape((nims,-1)))

#tensor versions

#change from batch by 1d vector form to batch by 2 dim image form
def tsr_vecs2ims(imvec,patchparm):
    nims = tf.shape(imvec)[0]
    patchsize = patchparm
    return(tf.reshape(imvec,(nims,patchsize,patchsize)))
           
#change from batch by 2 dim image form to  batch by 1d vector form
def tsr_ims2vecs(ims,patchparm):
    nims = np.shape(ims)[0]
    patchvecsize = patchparm**2
    return(tf.reshape(ims,(nims,-1)))