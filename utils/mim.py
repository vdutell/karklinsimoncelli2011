import os
import tensorflow as tf
import numpy as np
import shutil
import utils.tensorutils as tu

class mi_model(object):
    
    def __init__(self, params):
        params = self.add_params(params)
        self.params = params
        self.make_dirs()
        self.graph = self.make_graph()

    def add_params(self, params):  
        params['compression'] = params['imxlen'] * params['imylen'] / params['nneurons']
        params['savefolder'] = str('./output/image_output/' + 
                                   str(params['ims']) +
                                   '_' + str(params['nimgs']) +
                                   '_nonlin1_' + str(params['nonlin1'])+ 
                                   '_nonlin2_' + str(params['nonlin2'])+
                                   '_neurons_'+ str(params['nneurons'])+
                                   '_nin_'+ str(params['noise_x'])+
                                   '_nout_'+ str(params['noise_r'])+
                                   '_bsze_'+ str(params['batchsize'])+
                                   '_epochs_'+ str(params['epochs'])+
                                   '_lrate_'+ str(params['learning_rate'])+
                                   '_lambda_act'+ str(params['lambda_act'])+
                                   '_lambda_wgt' + str(params['lambda_wgt'])+
                                   '_invertcolors_' + str(params['colorinvert']) + '/')

        return(params)
        
    def make_dirs(self):
        if os.path.exists(self.params['savefolder']):
            shutil.rmtree(self.params['savefolder'])
        os.makedirs(self.params['savefolder'])
        os.makedirs(self.params['savefolder']+'param_evolution/')
        
    def make_graph(self):
    
        print('Compressing by',self.params['compression'],'for a total of',self.params['nneurons'],'neurons')

        #setup our graph
        #tf.reset_default_graph()
        mygraph = tf.Graph()
        with mygraph.as_default():
            
            #input images
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32, 
                                        shape=[self.params['batchsize'],self.params["imxlen"]**2])
            #now can define batch size
            #batch_size = tf.shape(self.x)[0]
            self.nbatches = tf.constant([self.params['batchsize'],1],dtype='int32')

                
            #activation function type
            with tf.name_scope('nonliearities'):
                self.nonlin1 = self.params['nonlin1']
                self.nonlin2  = self.params['nonlin2']

            #noises
            with tf.name_scope('noises'):
                self.noisexsigma = self.params['noise_x']
                self.noisersigma = self.params['noise_r']
                
            #Correlation matrices
            with tf.name_scope('corr_mats'):
                self.cx = tf.expand_dims(tf.constant(self.params['cx']),0)
                #self.cxbatch = tf.expand_dims(self.cx,0)
                #self.cxbatch = tf.tile(self.cx, self.nbatches)
                #self.cxbatch = tf.reshape(self.cxbatch,(self.params['batchsize'],tf.shape(self.cxbatch)[0]))

                self.cx_i = tf.matrix_inverse(self.cx)
                
                self.cnx = tf.expand_dims(tf.to_float(tf.constant(np.eye(self.params['patchsize']**2) * self.noisexsigma**2)),0)
                #self.cnxbatch = self.cnx
                #self.cnxbatch = tf.expand_dims(self.cnx,0)
                #self.cnxbatch = tf.tile(self.cnx, self.nbatches)
                #self.cnxbatch = tf.reshape(self.cnxbatch,(self.params['batchsize'],tf.shape(self.cnx)[0]))
                
                self.cnr = tf.cast( tf.expand_dims(tf.constant(np.eye(self.params['nneurons']) * self.noisersigma**2),0),tf.float32)
                #self.cnrbatch = tf.tile(self.cnr, self.nbatches)
                #self.cnrbatch = tf.reshape(self.cnrbatch,(self.params['batchsize'],tf.shape(self.cnr)[0]))
                                       
            #function to add noise
            with tf.name_scope("add_noise"):
                def add_noise(input_layer, std):
                    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
                    return tf.add(input_layer,noise)
             
            #weights
            with tf.variable_scope("weights"):
                
                self.weights_kernel_in = tf.random_uniform([self.params['imxlen'] * self.params['imylen'], self.params['nneurons']], dtype=tf.float32,minval=-1,maxval=1)
                #self.weights_kernel_in = tf.expand_dims(self.weights_kernel_in,0)
                
                #weights seeded as random uniform
                #self.weights_kernel_out = tf.random_uniform([self.params['nneurons'],
                #self.params['imxlen']*self.params['imylen']],
                #dtype=tf.float32,minval=-1,maxval=1)
                
                self.w = tf.get_variable('w',
                                         initializer = self.weights_kernel_in)

            #bias
            with tf.variable_scope("bias"):
                offset = 0 #to keep values positive
                self.bias = tf.Variable(tf.random_normal([self.params['nneurons']],dtype=tf.float32,
                                                         stddev=0.1)+offset)
           
            #lambda
            with tf.name_scope('lambda_activation'):
                self.lambda_act = self.params['lambda_act']
                
            #lambda2
            with tf.name_scope('lambda_weights'):
                self.lambda_wgt = self.params['lambda_wgt']

            #learning_rate
            with tf.name_scope('learning_rate'):
                self.learning_rate = self.params['learning_rate']

            #nonlienarities
            with tf.name_scope("nonlienarities"):
                def sigmoid(x):
                    return 1 / (1 + tf.exp(-x))

                def ddxsigmoid(x):
                    return sigmoid(x)*(1-sigmoid(x))

            #encoding part of model
            with tf.name_scope("encoding"):
                noisy_input = add_noise(self.x,self.params['noise_x'])
                #add noise to input, and multiply by weights
                linearin = tf.matmul(noisy_input, self.w) + self.bias 
                self.activation = tf.map_fn(sigmoid,linearin)

            def broadcast_matmul(A, B):
                "Compute A @ B, broadcasting over the first `N-2` ranks"
                with tf.variable_scope("broadcast_matmul"):
                    return tf.reduce_sum(A[..., tf.newaxis] * B[..., tf.newaxis, :, :],axis=-2)
                
                
            with tf.name_scope("G"):
                self.slopes = tf.expand_dims(tf.map_fn(ddxsigmoid,self.activation),-1)
                self.G = tf.expand_dims(tf.to_float(tf.eye(self.params['nneurons'])),0) 
                self.G = tf.multiply(self.G, self.slopes)
                
            with tf.name_scope("crx"):
                self.crx = broadcast_matmul(self.G, tf.transpose(self.w))
                self.crx = broadcast_matmul(self.crx, self.cnx)
                self.crx = broadcast_matmul(self.crx, self.w)
                self.crx = broadcast_matmul(self.crx, self.G)
                self.crx = self.crx + self.cnr 
                self.crx_i = tf.matrix_inverse(self.crx)
                
            with tf.name_scope("cxr"):                
                self.cxr = broadcast_matmul(self.w, self.G)
                self.cxr = self.cxr @ self.crx_i
                self.cxr = self.cxr @ self.G 
                self.cxr = broadcast_matmul(self.cxr, tf.transpose(self.w))
                self.cxr = self.cx_i + self.cxr
                
            #mut info calc part of model
            with tf.name_scope("mut_info_calc"):
           
                self.Hxr = 0.5*tf.log(2*np.pi*np.e*tf.matrix_determinant(self.cxr))

            #calculate cost
            with tf.name_scope("cost_function"):
                self.mean_act = tf.reduce_mean(self.activation,axis=0)
                self.cost = (self.Hxr +
                             self.lambda_act * tf.norm(self.mean_act,ord=1) + 
                             self.lambda_wgt * tf.norm(self.w, ord=1))
                     
            #train our model
            with tf.name_scope("training_step"):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
                #self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
                
            # create a summary for our cost, im, & weights
            with tf.name_scope('cost_viz'):
                tf.summary.scalar("cost", self.cost)

            with tf.name_scope('image_viz'):
                x_t = tu.tsr_vecs2ims(self.x,self.params['patchsize'])
                tf.summary.image("image", x_t, max_outputs=self.params['batchsize'])

            with tf.name_scope('weights_viz'):    
                inwin_t = tf.reshape(tf.transpose(self.w),
                                   (self.params['nneurons'],
                                    self.params['imxlen'],
                                    self.params['imylen'],1))
                tf.summary.image("inweights", inwin_t, max_outputs=self.params['nneurons'])
                

            # merge all summaries into a single "operation" which we can execute in a session 
            self.summary_op = tf.summary.merge_all()

        return(mygraph)
