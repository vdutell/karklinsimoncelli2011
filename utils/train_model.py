import tensorflow as tf
import numpy as np

def train_model(mim,vhimgs):
    with tf.device("/gpu:0"):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #don't allocate the entire GPU's memory
        config.log_device_placement=True #tell me where devices are placed
        with tf.Session(graph = mim.graph, config=config) as sess:

            #initialize vars
            init = tf.global_variables_initializer()
            sess.run(init)

            #summary writer for tensorboard
            writer = tf.summary.FileWriter(mim.params['savefolder'],
                                           graph=tf.get_default_graph())

            #save evolution of system over training
            cost_evolution = []
            wmean_evolution = []

            inweights_evolution = []
            inbias_evolution = []
            activation_evolution = []

            activations = []

            images = []
            print('neurons={}, noise_in={}, noise_out={}, lambda_w={}, lambda_act={}'
                  .format(mim.params['nneurons'],
                          mim.params['noise_x'],
                          mim.params['noise_r'],
                          mim.params['lambda_wgt'],
                          mim.params['lambda_act']))

            print('Training {} iterations in {} epochs... '.format(mim.params['iterations'],
                                                                   mim.params['epochs']))
            for epoch in range(mim.params['epochs']):
                #print('Epoch {}: '.format(epoch+1))
                np.random.shuffle(vhimgs)
                for ii in range(mim.params['iterations']):

                    #reshape our images for feeding to dict
                    image = np.reshape(vhimgs[ii*mim.params['batchsize']:(1+ii)*mim.params['batchsize'],:,:],
                                       (mim.params['batchsize'],
                                        mim.params['imxlen']*mim.params['imylen'])).astype(np.float32)

                    #setup params to send to dictionary
                    feeddict = {mim.x: image}

                    #run our session
                    sess.run(mim.train_step, feed_dict=feeddict)

                    #save evolution of params
                    objcost, inws, acts = sess.run([mim.recon_err, mim.w, mim.activation], feed_dict=feeddict)  #mim.cost
                    cost_evolution.append(objcost)
                    wmean_evolution.append(np.mean(np.abs(inws)))
                    activations.append(np.mean(acts,axis=0))

                    #save detailed parameters 10 times over the total evolution
                    if(ii%(int((mim.params['iterations']*mim.params['epochs'])/10))==0):
                        print(str(ii)+', ',end="")
                        #dump our params
                        w, img, recon, inbias, activation = sess.run([mim.w, mim.x, mim.xp, mim.inbias, mim.activation], feed_dict=feeddict)
                        #save our weights, image, and reconstruction
                        inweights_evolution.append(w)
                        inbias_evolution.append(inbias)
                        activation_evolution.append(activation)
                        
                        #reshape images and append
                        imshape = [mim.params['batchsize'],
                                   mim.params['imxlen'],
                                   mim.params['imylen']]   
                        images.append(np.reshape(img, imshape))
                        

            #summarizeparams
            inweights_evolution = np.array(inweights_evolution)
            activation_evolution = np.mean(activation_evolution,axis=1)
            
            test_acts = activation_evolution[-1]
        
            #find order based on activations
            order_test_acts = np.argsort(-np.mean(np.array(test_acts),axis=0))

            #reorder our data based on this ordering
            weights_kernel_in_ordered = weights_kernel_in[:,order_test_acts] #reorder based on activations
            test_inweights_ordered = test_w[:,order_test_acts] #reorder based on activations
            test_outweights_ordered = test_wout.T[:,order_test_acts] #reorder based on activations
            test_acts_ordered = test_acts[:,order_test_acts] #reorder based on activations
           
            
            #save summary
            writer.add_summary(summary,ii)
            writer.close()


            return(mim,
                   cost_evolution,
                   wmean_evolution,
                   inweights_evolution,
                   activation_evolution,
                   inbias_evolution,
                   weights_kernel_in_ordered,
                   test_inweights_ordered,
                   test_acts_ordered,
                   test_acts)