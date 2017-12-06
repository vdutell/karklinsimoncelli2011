import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.ticker import NullFormatter
from sklearn import manifold
import scipy.spatial.distance as scpd
    
"""
Author: Dylan Payton taken from FeedbackLCA code
Pad data with ones for visualization
Outputs:
  padded version of input
Args:
  data: np.ndarray
"""

def pad_data(data):
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1))                       # add some space between filters
    + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
    padded_data = np.pad(data, padding, mode="constant", constant_values=1)
    # tile the filters into an image
    padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
    padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
    return padded_data


def normalize_data(data):
    norm_data = data.squeeze()
    if np.max(np.abs(data)) > 0:
        norm_data = (data / np.max(np.abs(data))).squeeze()
    return norm_data


"""
Author: Dylan Payton taken from FeedbackLCA code
Display input data as an image with reshaping
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: np.ndarray of shape (height, width) or (n, height, width)
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: string for title of figure
"""
def display_data_acts_tiled(data, acts, normalize=False, title=""):
       
    #calculate mean of each picture of weights
    mean_list =[]
    for x in data:
        mean_list.append(np.linalg.norm(np.reshape(x,-1),ord=2))
    
    mean_list = np.array(mean_list)
    
    #Rescale data    
    mean_data = np.mean(data)
    min_data = np.amin(data)
    max_data = np.amax(data)
    data = (((data-min_data)/(max_data-min_data))*2)-1
    
    if normalize:
        data = normalize_data(data)
    if len(data.shape) >= 3:
        data = pad_data(data)
        
    fig = plt.figure(figsize=(10,3))
    
    sub_axis = fig.add_subplot(1,2,2)  
    axis_image = sub_axis.imshow(data, 
                                 cmap="Greys_r",
                                 interpolation=None)
    axis_image.set_clim(vmin=-1.0, vmax=1.0)
    # Turn off tick labels
    sub_axis.set_yticklabels([])
    sub_axis.set_xticklabels([])
    cbar = fig.colorbar(axis_image)
    sub_axis.tick_params(
        axis="both",
        bottom="off",
        top="off",
        left="off",
        right="off")  
    
    bar_chart = fig.add_subplot(1,2,1)

    bar_chart.bar(range(0, np.shape(acts)[0]), acts, edgecolor = 'black', color = 'black')

    #bar_chart.title()
    fig.canvas.draw()
    #plt.show()
    
    return (fig, sub_axis, axis_image)


def plot_tiled_rfs(data, normalize=False):
       
    #calculate mean of each picture of weights
    mean_list =[]
    for x in data:
        mean_list.append(np.linalg.norm(np.reshape(x,-1),ord=2))
    mean_list = np.array(mean_list)
    
    #Rescale data    
    mean_data = np.mean(data)
    min_data = np.amin(data)
    max_data = np.amax(data)
    data = (((data-min_data)/(max_data-min_data))*2)-1
    
    if normalize:
        data = normalize_data(data)
    if len(data.shape) >= 3:
        data = pad_data(data)
        
    fig = plt.imshow(data, 
                     cmap="Greys_r",
                     interpolation="none")
    fig.set_clim(vmin=-1.0, vmax=1.0)
    # Turn off tick labels
    #fig.set_major_formatter(NullFormatter())
    plt.xticks([])
    plt.yticks([])
    #fig.set_yticklabels([])
    #fig.set_xticklabels([])
    plt.tick_params(
        axis="both",
        bottom="off",
        top="off",
        left="off",
        right="off") 
    plt.axis('off')
    plt.colorbar()
   
    return (fig)

"""
Author: Vasha DuTell
Plot to visualize the tiling of the center RF of on and off cells separately.
Outputs:
  Figure object with two tiling plots, one with on, and the other with off cells.
Args:
  data: np.ndarray or list of weights, each an individiaul neuron RF
"""
def plotonoff(allws):

    #Rescale data    
    mean_data = np.mean(allws)
    min_data = np.amin(allws)
    max_data = np.amax(allws)
    data = (((allws-min_data)/(max_data-min_data))*2)-1
    #data = normalize_data(data)
    
    #extract on center
    onws = np.mean(allws,axis=0)>0
    onws = allws[:,onws]
    #extract off center
    offws = np.mean(allws,axis=0)<0
    offws = allws[:,offws]
    #keep track of the circles
    oncircs = []
    offcircs = []
    ambiguous = []
    labels = []

    circthresh = 0.5
    onoffthresh = 1e-6
    
    for ws in allws:
        if(np.mean(ws)>onoffthresh):
            circ = (ws>(circthresh*np.sign(np.mean(ws))))
            oncircs.append(circ)
            labels.append(1)
        elif(np.mean(ws)<-onoffthresh):
            circ = (ws<(circthresh*np.sign(np.mean(ws))))
            offcircs.append(circ)
            labels.append(-1)
        else:
            labels.append(0)

    #plot
    fig = plt.figure(figsize=(10,6))
    plt.subplot(1,2,1,title='On')    
    oncolors = iter(plt.cm.jet(np.linspace(0,1,len(oncircs))))           
    for onc in oncircs: 
        plt.contour(onc,[0.3],linewidths = 3,colors=[next(oncolors)])
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1,2,2,title='Off')
    offcolors = iter(plt.cm.jet(np.linspace(0,1,len(offcircs))))  
    for ofc in offcircs:
        plt.contour(ofc,[0.3], linewidths = 3, colors=[next(offcolors)])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    return(np.array(labels), fig)

def measure_plot_dist(weight_mat, norm, normalize=True, plot=True):
    ## measures pairwise norm of hidden node weights.
    ## Inputs:
    ## weight_mat: matrix of weights of shape nneurons by input shape (input shape can be 1 or 2d)
    ## norm: String describing the type of norm to take - see acceptible norms in documentation for scipy.spatial.distance.pdist
    ## normalize: boolean whether to normalize weight vectors to unit norm
    
    ## Outputs:
    ## dist: a nneurons by nneurons matrix, with pairwise distances of the weight matrices in each element
    ## fig: plot of the distance matrix as a heatmap
    
    #vectorize
    fwv = weight_mat.reshape(weight_mat.shape[0],-1)
    
    #make each weigth vector unit norm
    if(normalize):
        fwv /= np.linalg.norm(fwv, axis=1, ord=norm)[:,np.newaxis]
    #print(np.linalg.norm(fwv,axis=1))
        
    dist = scpd.pdist(fwv, metric='minkowski', p=norm)
    dist = scpd.squareform(dist)
    dists = dist[np.nonzero(np.triu(dist))]
    meandist = np.mean(dists)
    
    if(plot):

        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1, 2, 1)
        plt.title('Pairwise Distances')
        plt.pcolormesh(dist)
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        plt.title('Mean Dist = {0:.3f}, Sqrt(2) = {1:.3f}'.format(meandist, np.sqrt(2)))
        plt.hist(dists, 50);
        plt.axvline(meandist,color='r')
        plt.axvline(np.sqrt(2),color='g')
        return(dist, fig)
    
    else:
        return(dists)


def measure_plot_act_corrs(activations):
    ccf = np.corrcoef(np.array(activations).T)
    fig = plt.figure(figsize = (10,10))
    plt.pcolormesh(ccf)
    plt.colorbar()
    
    return(ccf,fig)


def plot_dist_embeddings(distmat, onofflabels, n_neighbors = 10, n_components = 2):
     
    fig = plt.figure(figsize = (10,10))

    #isomap
    iso = manifold.Isomap(n_neighbors, n_components).fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 1)
    plt.scatter(iso[:, 0], iso[:, 1], c=onofflabels)
    plt.title('Isomap - {} Neighbors'.format(n_neighbors))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    #Spectral
    spec = manifold.SpectralEmbedding(n_components=n_components, n_neighbors = n_neighbors).fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 2)
    plt.scatter(spec[:, 0], spec[:, 1], c=onofflabels)
    plt.title('Spectral - {} Neighbors'.format(n_neighbors))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    #TSNE
    tsne = manifold.TSNE(n_components, init='pca', random_state=0).fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 3)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=onofflabels)
    plt.title('t-SNE - {} Components'.format(n_components))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    #MDS
    max_iter = 10
    mds = manifold.MDS(n_components=n_components, metric=True, max_iter=max_iter, dissimilarity="precomputed").fit_transform(distmat)
    ax = fig.add_subplot(2, 2, 4)
    plt.scatter(mds[:, 0], mds[:, 1], c=onofflabels)
    plt.title('MDS - {} Components'.format(n_components))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    return(fig)


def dist_init_final(weights_kernel_in, final_weights_in, norm = 1):
    fig = plt.figure(figsize = (8,8))
    
    #plot initial weights
    ax = fig.add_subplot(2, 2, 1)
    plt.title('Initial Weights')
    ax = plot_tiled_rfs(weights_kernel_in, normalize=False)
                            
    #plot final weights
    ax = fig.add_subplot(2, 2, 2)
    plt.title('Final Weights')
    ax = plot_tiled_rfs(final_weights_in, normalize=False)
    
    diff = weights_kernel_in - final_weights_in
    
    #plot difference weights
    ax = fig.add_subplot(2, 2, 3)
    plt.title('Difference')
    ax = plot_tiled_rfs(diff, normalize=False)
    
    ax = fig.add_subplot(2, 2, 4)
    plt.title('Difference Distribution')
    ax = plt.hist(diff.flatten(),50)
    
    return(fig)

def test_activation_distributions(test_acts_ordered, onofflabel, norm=True):
    #distribution of activations for on and off cells
    actv = np.array(test_acts_ordered).T
    on = actv[np.where(onofflabel==1)[0]]
    off = actv[np.where(onofflabel==-1)[0]]
    other = actv[np.where(onofflabel==0)[0]]
    fig = plt.figure()
    p = plt.hist(on.flatten(), 100, alpha = 0.5, label = 'on', normed=norm)
    p = plt.hist(off.flatten(), 100, alpha = 0.5, label = 'off', normed=norm)
    #if some are classified as other, plot them
    if(other.size != 0):
        p = plt.hist(other.flatten(), 100, alpha = 0.5, label = 'other', normed=norm)
    plt.legend(loc='upper right')

    return(fig)
    

def save_plots(model,
               cost_evolution,
               wmean_evolution,
               inweights_evolution,
               outweights_evolution,
               activation_evolution,
               inbias_evolution,
               weights_kernel_in_ordered,
               test_patches,
               test_recons,
               test_inweights_ordered,
               test_outweights_ordered,
               test_acts_ordered):
    
     
    savefolder = model.params['savefolder']
  
    ## trained in weights
    fiw = test_inweights_ordered.reshape(model.params['imxlen'],
                                  model.params['imylen'],
                                  model.params['nneurons']).T
    
    ## initial weights
    iw = weights_kernel_in_ordered.reshape(model.params['imxlen'],
                                     model.params['imylen'],
                                     model.params['nneurons']).T
    
    ## trained out weights
    fow = test_outweights_ordered.reshape(model.params['imxlen'],
                                  model.params['imylen'],
                                  model.params['nneurons']).T
    
    (f,sa,ai) = display_data_acts_tiled(fiw, np.mean(test_acts_ordered,axis=0), normalize=True, title="final_in_weights");
    f.savefig(savefolder+'trained_weights_in.png')
    plt.close()    
    

    (f,sa,ai) = display_data_acts_tiled(fow, np.mean(test_acts_ordered,axis=0), normalize=True, title="final_out_weights");
    
    f.savefig(savefolder+'trained_weights_out.png')
    plt.close()
   
    #save evolving weights
    inweights_evolution_r = np.rollaxis(np.reshape(inweights_evolution,
                                         (len(inweights_evolution),
                                          model.params['imxlen'],
                                          model.params['imylen'],
                                          model.params['nneurons'])),3,1)
    outweights_evolution_r = np.reshape(outweights_evolution,
                                         (len(outweights_evolution),
                                          model.params['nneurons'],
                                          model.params['imxlen'],
                                          model.params['imylen'])) #no rollaxis needed b/c shape is already nnuerons in pos 1.    
  
    for i in range(len(inweights_evolution_r)):
        (f,sa,ai) = display_data_acts_tiled(inweights_evolution_r[i], np.mean(test_acts_ordered,axis=0), normalize=True, title="inweights_evolving");
        f.savefig(savefolder+'param_evolution/inweights_evolution_'+str(i)+'.png')
        plt.close()
        
        (f,sa,ai) = display_data_acts_tiled(outweights_evolution_r[i],  np.mean(test_acts_ordered,axis=0), normalize=True, title="outweights_evolving");
        f.savefig(savefolder+'param_evolution/outweights_evolution_'+str(i)+'.png')
        plt.close()
        
    #save plots of on and off tiling
    onofflabels, f = plotonoff(fiw);
    f.savefig(savefolder+'/trained_in_on_off_RFs.png') 
    plt.close()
    
    #save plot of trained activations of individual weights
    f = test_activation_distributions(test_acts_ordered, onofflabels, norm=True)
    plt.title('Trained Activations')
    f.savefig(savefolder+'/trained_node_activations.png') 
    plt.close()
    
    #save weights and cost evolution
    f = plt.figure(figsize=(10,10))
    plt.subplot(2,1,1,title='Weights_Mean')
    plt.plot(wmean_evolution)
    plt.subplot(2,1,2,title='Cost')
    plt.plot(cost_evolution)
    plt.tight_layout()
    f.savefig(savefolder+'/summary_weights_cost.png') 
    plt.close()
    
    #save reconstruction polots
    
    #show an example image and reconstruction from the last iteration of learning
    patchnum = 3
    plots = 4
    
    f = plt.figure()
    for i in range(plots):
        plt.subplot(plots,2,2*i+1)#,title='Patch')
        plt.imshow(test_patches[patchnum+i,:],cmap='gray',interpolation='none')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(plots,2,2*i+2)#,title='Recon')
        plt.imshow(test_recons[patchnum+i,:],cmap='gray',interpolation='none')
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    f.savefig(savefolder+'/reconstruction.png') 
    plt.close() 
    

    #save weight plots
    
    #trained weight distances
    weight_distmat, f = measure_plot_dist(fiw, norm = 1);
    f.savefig(savefolder+'/trained_weight_distances.png') 
    plt.close()
    
    #trained weight distnace clustering
    f = plot_dist_embeddings(weight_distmat, onofflabels, n_neighbors=5)
    f.savefig(savefolder+'/trained_weight_distances_manifold_embeddings.png') 
    plt.close()
    
    #init vs final weights
    f = dist_init_final(iw, fiw, norm = 1);
    f.savefig(savefolder+'/train_dist_init_final_inweights.png') 
    plt.close()
    
    
    #save activation plots
    
    #trained activaiton distances
    act_distmat, f = measure_plot_dist(test_acts_ordered[1:1000,:], norm = 2);
    f.savefig(savefolder+'/trained_act_distances.png') 
    plt.close()
    
    #trained weight distnace clustering
    f = plot_dist_embeddings(act_distmat, np.ones(act_distmat.shape[0]), n_neighbors=5)
    f.savefig(savefolder+'/trained_act_distances_manifold_embeddings.png') 
    plt.close()
        
    #trained activation correlation plots
    corrs, f =  measure_plot_act_corrs(test_acts_ordered);
    f.savefig(savefolder+'/trained_act_corrs.png') 
    plt.close()
    
    
    #save plots of activation evolution
    for i in range(len(activation_evolution)):
        f = plt.figure()
        plt.bar(range(0, len(activation_evolution[i])), activation_evolution[i], edgecolor = 'black', color = 'black')
        f.savefig(savefolder+'param_evolution/activation_'+str(i)+'.png')
        plt.close()
        
    #save plots of inbias evolution
    for i in range(len(inbias_evolution)):
        f = plt.figure()
        plt.bar(range(0, len(inbias_evolution[i])), inbias_evolution[i], edgecolor = 'black', color = 'black')
        f.savefig(savefolder+'param_evolution/inbias_'+str(i)+'.png')
        plt.close()
    

        
        