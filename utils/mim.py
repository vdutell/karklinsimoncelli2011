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
                self.nonlin2 = self.params['nonlin2']

            #noise parameters
            with tf.name_scope('noises'):
                self.noisexsigma = self.params['noise_x']
                self.noisersigma = self.params['noise_r']
                
            #in noise values
            with tf.name_scope("in_noise"):
                self.in_noise = tf.random_normal(shape=tf.shape(self.x), mean=0.0, stddev=self.noisexsigma, dtype=tf.float32)         
                
            #Input correlation matrix   
            with tf.name_scope('input_corr_mat'):
                #cx
                self.cx = tf.expand_dims(tf.constant(self.params['cx']),0)
                self.cx_i = tf.matrix_inverse(self.cx)
            
            #Input Noise correlation matrix
            with tf.name_scope('in_noise_corr_mat'):
                self.cnx = tf.constant(self.params['cnx'])

                #self.cnx = tf.expand_dims(self.cnx,0)
                #self.cnx = tf.expand_dims(tf.to_float(tf.eye(self.params["imxlen"]**2)),0) 
                #self.cnx = tf.multiply(self.cnx, self.noisexsigma)        

                #cnx
                #self.cnx = np.float32(np.corrcoef(self.in_noise,rowvar=False))
                #cnx_t = tf.transpose(self.in_noise)
                ##tf.Print(tf.shape(cnx_t), cnx_t)
                #mean_cnx = tf.reduce_mean(cnx_t, axis=1, keep_dims=True)
                #cov_cnx = ((cnx_t-mean_cnx) @ tf.transpose(cnx_t-mean_cnx))/(self.params["imxlen"]**2-1)
                #cov2_cnx = tf.diag(1/tf.sqrt(tf.diag_part(cov_cnx)))
                #self.cnx = cov2_cnx @ cov_cnx @ cov2_cnx
                #self.cns = (self.in_noise)

                #tf.expand_dims(tf.to_float(np.eye(self.params['patchsize']**2) * self.noisexsigma**2),0)
                #self.cnxbatch = self.cnx
                #self.cnxbatch = tf.expand_dims(self.cnx,0)
                #self.cnxbatch = tf.tile(self.cnx, self.nbatches)
                #self.cnxbatch = tf.reshape(self.cnxbatch,(self.params['batchsize'],tf.shape(self.cnx)[0]))
                
                
            #weights
            with tf.variable_scope("weights"):
                
                self.weights_kernel_in = tf.random_normal([self.params['imxlen'] * self.params['imylen'], self.params['nneurons']], stddev=0.1, dtype=tf.float32)
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
                                                         stddev=0.01)+offset)
                #self.bias = tf.expand_dims(self.bias,0)
           
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
              
            with tf.name_scope("encoding"):
                noisy_input = self.x + self.in_noise
                #add noise to input, and multiply by weights
                self.linearin = noisy_input @ self.w + self.bias
                self.activation = tf.map_fn(sigmoid,self.linearin)
            
            with tf.name_scope("out_noise"):
                self.out_noise = tf.random_normal(shape=tf.shape(self.activation), mean=0.0, stddev=self.noisersigma, dtype=tf.float32) 
                
            def broadcast_matmul(A, B):
                "Compute A @ B, broadcasting over the first `N-2` ranks"
                with tf.variable_scope("broadcast_matmul"):
                    return tf.reduce_sum(A[..., tf.newaxis] * B[..., tf.newaxis, :, :],axis=-2)
            
            def broadcast_ccf(A):
                "Compute corcoeff(A), the autocorrelation, broadcasting over the 0th dimension"
                with tf.variable_scope("broadcast_ccf"):
                    nbatches = tf.shape(A)[0]
                    mean_a = tf.reduce_mean(A,axis=1,keep_dims=True)
                    cov_a = broadcast_matmul((A-mean_a), tf.transpose(A-mean_a))/(nbatches-1)
                    cov2_a = tf.diag(1/tf.sqrt(tf.diag_part(cov_a)))
                    cmat_a = broadcast_matmul(cov2_cnx, cov_cnx)
                    cmat_a = broadcast_matmul(cmat_a, cov2_cnx)
                    return cmat_a 
                
            with tf.name_scope("out_noise_corr_mat"):
                self.cnr = tf.constant(self.params['cnr'])
                #self.cnr = tf.ones([self.params["nneurons"],self.params["nneurons"]])
                #self.cnr = self.cnr*self.noisersigma**2
                #cnrdiag = tf.fill([self.params["nneurons"]],1.0)
                #self.cnr = tf.matrix_set_diag(self.cnr,cnrdiag)
                #self.cnr = tf.ones([self.params["nneurons"]**2,self.params["nneurons"]**2])*self.noisersigma
                #self.cnr = tf.matrix_set_diag(self.cnr,1)
                #self.cnr = tf.expand_dims(tf.to_float(tf.eye(self.params['nneurons'])),0) 
                #self.cnr = tf.multiply(self.cnr, self.noisersigma)
                #cnr
                #cnr_t = tf.transpose(self.out_noise)
                ##mean_cnr = tf.reuce_mean(cnr_t, axis=1, keep_dims=True)
                #cov_cnr = ((cnr_t-mean_cnr) @ tf.transpose(cnr_t-mean_cnr))/(self.params["nneurons"]-1)
                #cov2_cnr = tf.diag(1/tf.sqrt(tf.diag_part(cov_cnr)))
                #self.cnr = cov2_cnr @ cov_cnr @ cov2_cnr
                #self.cnr = np.float32(np.corrcoef(self.out_noise,rowvar=False))
                #self.cnr = tf.cast(tf.expand_dims(np.eye(self.params['nneurons']) * self.noisersigma**2,0),tf.float32)
                #self.cnr = broadcast_ccf(self.out_noise)
             
                   
            with tf.name_scope("noisy_output"):
                self.output = self.activation + self.out_noise
               
            with tf.name_scope("G"):
                self.slopes = tf.expand_dims(tf.map_fn(ddxsigmoid,self.activation),-1)
                self.G = tf.expand_dims(tf.to_float(tf.eye(self.params['nneurons'])),0) 
                self.G = tf.multiply(self.G, self.slopes)

            with tf.name_scope("crx"):
                #this is SLOW AF
                #self.crx = broadcast_matmul(self.G, tf.transpose(self.w))
                #self.crx = broadcast_matmul(self.crx, self.cnx)
                #self.crx = broadcast_matmul(self.crx, self.w)
                #self.crx = broadcast_matmul(self.crx, self.G)
                #self.crx = self.crx + self.cnr
                
                #self.crx = self.G @ tf.transpose(self.w)
                #self.crx = self.crx @ self.w
                #self.crx = self.crx @ self.G
                #self.crx = self.crx + self.cnr 
                
                #fewer broadcast matmuls
                self.crx = tf.transpose(self.w) @ self.cnx @ self.w
                self.crx = broadcast_matmul(self.G,self.crx)
                self.crx = broadcast_matmul(self.crx, self.G)
                self.crx = self.crx + self.cnr             
                self.crx_i = tf.matrix_inverse(self.crx,adjoint=True)

            with tf.name_scope("cxr"):                
                self.cxr = broadcast_matmul(self.w, self.G)
                self.cxr = self.cxr @ self.crx_i
                self.cxr = self.cxr @ self.G 
                self.cxr = broadcast_matmul(self.cxr, tf.transpose(self.w))
                self.cxr = self.cx_i + self.cxr
                self.cxr_i = tf.matrix_inverse(self.cxr)
                
            #mut info calc part of model
            with tf.name_scope("mut_info_calc"):
                #self.Hxr = 0.5*tf.log(2*np.pi*np.e*tf.matrix_determinant(self.cxr))
                #use identity log(det(a)) = tr(log(a)) and log(a*b) = log(a)+log(b)
                #self.Hxr = 0.5*tf.log(2*np.pi*np.e) + tf.trace(tf.log(self.cxr_i))
                #and remove constant
                self.Hxr = -tf.trace(tf.log(self.cxr_i))

            #calculate cost
            with tf.name_scope("cost_function"):
                self.mean_act = tf.reduce_mean(self.activation,axis=1)
                self.cost = (self.Hxr +
                             self.lambda_act * tf.norm(self.mean_act,ord=1) + 
                             self.lambda_wgt * tf.norm(self.w, ord=1))

            #train our model
            with tf.name_scope("training_step"):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cxr)
                #self.train_step = #tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.crx)
                
            # create a summary for our cost, im, & weights
#             with tf.name_scope('cost_viz'):
#                 tf.summary.scalar("cost", self.cost)

#             with tf.name_scope('image_viz'):
#                 x_t = tu.tsr_vecs2ims(self.x,self.params['patchsize'])
#                 x_t = tf.expand_dims(x_t,-1)
#                 tf.summary.image("image", x_t, max_outputs=self.params['batchsize'])

#             with tf.name_scope('weights_viz'):    
#                 inwin_t = tf.reshape(tf.transpose(self.w),
#                                    (self.params['nneurons'],
#                                     self.params['imxlen'],
#                                     self.params['imylen'],1))
#                 tf.summary.image("inweights", inwin_t, max_outputs=self.params['nneurons'])
                

            # merge all summaries into a single "operation" which we can execute in a session 
#             self.summary_op = tf.summary.merge_all()

            return(mygraph)
