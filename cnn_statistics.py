from skimage.transform import resize
from image_generator import ImageGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
import sys
import caffe
import scipy
import math
import shutil
import string
from scipy import ndimage
from scipy.sparse.linalg import eigs



def var_sep(generator,net,results_folder,test_layers):
# compute the relative variance along the two directions of variations without
# computing the covariance matrix
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(results_folder+'/pca'):
        os.makedirs(results_folder+'/pca')
    translation=string.maketrans('/', '_')
    var={}
    mean={}
    size={}
    var_dim1={}
    var_dim2={}
    tmp=np.zeros([256,256,3])
    caffe_input = np.asarray([net.transformer.preprocess('data', tmp)])
    results=net.forward_all(data=caffe_input,blobs=test_layers)
    t = time.time()
    for layer in test_layers:
        size[layer]=len(results[layer].flatten())
        var[layer] = 0
        mean[layer] = np.zeros([size[layer] ])
        var_dim1[layer]=0
        var_dim2[layer]=0

    N_dim1=generator.N1

    t = time.time()
    for instance_id in range(0,N_dim1):
        caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in generator.generator()])
        results=net.forward_all(data=caffe_input,blobs=test_layers)
        for layer in test_layers:
            R=np.reshape(results[layer],[generator.N2,size[layer]])
	    mean[layer]=mean[layer]+R
            var[layer]=var[layer]+np.mean(np.sum(R*R,axis=1))
            m=np.mean(R,axis=0)
            v=np.sum(m*m)
            var_dim2[layer]=var_dim2[layer]+v
        print('\rcomputed '+str(instance_id+1)+' instances in ' +str(time.time()-t)+' seconds'),
    print('\r')
    for layer in test_layers:
        layer_t=layer.translate(translation)
        mean[layer]=mean[layer]/N_dim1
        mean_tot=np.mean(mean[layer],axis=0)
        var_dim1=  np.sum(mean[layer]*mean[layer])/generator.N2-np.sum(mean_tot*mean_tot)

        var[layer]=var[layer]/N_dim1-np.sum(mean_tot*mean_tot)

        var_dim2[layer]=var_dim2[layer]/N_dim1-np.sum(mean_tot*mean_tot)

        var_x=var[layer]-var_dim2[layer]-var_dim1


        print('Variance repartition layer '+layer+' :')
	print('Dimension 1 : '+'%.1f'%(100*var_dim1/var[layer])+'%')
	print('Dimension 2 : '+'%.1f'%(100*var_dim2[layer]/var[layer])+'%')
        print('Residual    : '+'%.1f'%(100*var_x/var[layer])+'%')
    return

def pca(generator,net,results_folder,test_layers,N_projections=100):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(results_folder+'/pca'):
        os.makedirs(results_folder+'/pca')
    translation=string.maketrans('/', '_')
    var={}
    mean={}
    size={}
    var_dim1={}
    var_dim2={}
    tmp=np.zeros([256,256,3])
    caffe_input = np.asarray([net.transformer.preprocess('data', tmp)])            
    results=net.forward_all(data=caffe_input,blobs=test_layers)
    t = time.time()  
    t_caffe=0
    t_python=0
    for layer in test_layers:
        size[layer]=len(results[layer].flatten())
        var[layer] = np.zeros([size[layer],size[layer] ])
        mean[layer] = np.zeros([size[layer] ])
        var_dim1[layer]=np.zeros([size[layer],size[layer] ])
        var_dim2[layer]=np.zeros([size[layer],size[layer] ])
        
    N_dim1=generator.N1
    for instance_id in range(0,N_dim1):
        caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in generator.generator()])
        results=net.forward_all(data=caffe_input,blobs=test_layers)    
        t_caffe=t_caffe+t-time.time()
        t = time.time()  
        for layer in test_layers: 
            R=np.reshape(results[layer],[generator.N2,size[layer]])
            mean[layer]=mean[layer]+R
            v=np.dot(R.transpose(),R)/generator.N2
            var[layer]=var[layer]+v
            m=np.mean(R,axis=0)
            var_dim2[layer]=var_dim2[layer]+np.outer(m,m)
        t_python=t_python+t-time.time()
        t = time.time()  
        print('\r computed instance '+str(instance_id)+' in ' +str(t_caffe)+' seconds caffe '+' and ' +str(t_python)+' seconds python '),
        

    for layer in test_layers:
        layer_t=layer.translate(translation)
        
        mean[layer]=mean[layer]/N_dim1
        mean_tot=np.mean(mean[layer],axis=0)
        np.save(results_folder+'/pca/mean_layer_'+layer_t, mean_tot )
        mean[layer]=mean[layer]-np.tile(np.reshape(mean_tot,[1, len(mean_tot)]),[generator.N2,1])
        np.save(results_folder+'/pca/mean_full_layer_'+layer_t,mean[layer] )
        
        var_dim1=  np.dot(mean[layer].transpose(),mean[layer])/generator.N2      
        np.save(results_folder+'/pca/var_dim1_layer_'+layer_t,var_dim1 )
        np.save(results_folder+'/pca/tr_var_dim1_layer_'+layer_t,np.trace(var_dim1) )
#        eig_vals, eig_vecs =eigs(var_dim1,k=N_projections)    
#        eig_vals=np.real(eig_vals)
#        order=np.argsort(eig_vals)[::-1]
#        eig_vals=eig_vals[order]
#        eig_vecs=eig_vecs[:,order]
#        np.save(results_folder+'/pca/evecs_dim1_layer_'+layer_t,eig_vecs )
#        np.save(results_folder+'/pca/evals_dim1_layer_'+layer_t,eig_vals)
       
       
        
       # np.save(results_folder+'/pca/var_bm_layer_'+layer_t,var[layer] )
        var[layer]=var[layer]/N_dim1-np.outer(mean_tot,mean_tot)
        np.save(results_folder+'/pca/var_layer_'+layer_t,var[layer] )
        np.save(results_folder+'/pca/tr_var_layer_'+layer_t,np.trace(var[layer]) )
#        eig_vals, eig_vecs =eigs(var[layer],k=N_projections) 
#        eig_vals=np.real(eig_vals)
#        order=np.argsort(eig_vals)[::-1]
#        eig_vals=eig_vals[order]
#        eig_vecs=eig_vecs[:,order]
#        np.save(results_folder+'/pca/evecs_layer_'+layer_t,eig_vecs)
#        np.save(results_folder+'/pca/evals_layer_'+layer_t,eig_vals )
        
        var_dim2[layer]=var_dim2[layer]/N_dim1-np.outer(mean_tot,mean_tot)
        np.save(results_folder+'/pca/var_dim2_layer_'+layer_t,var_dim2[layer] )
        np.save(results_folder+'/pca/tr_var_dim2_layer_'+layer_t,np.trace(var_dim2[layer]) )
#        eig_vals, eig_vecs =eigs(var_dim2[layer],k=N_projections)    
#        eig_vals=np.real(eig_vals)
#        order=np.argsort(eig_vals)[::-1]
#        eig_vals=eig_vals[order]
#        eig_vecs=eig_vecs[:,order]
#        np.save(results_folder+'/pca/evecs_dim2_layer_'+layer_t,eig_vecs)
#        np.save(results_folder+'/pca/evals_dim2_layer_'+layer_t,eig_vals )
        
        var_x=var[layer]-var_dim2[layer]-var_dim1  
        np.save(results_folder+'/pca/var_x_layer_'+layer_t,var_x )
        np.save(results_folder+'/pca/tr_var_x_layer_'+layer_t,np.trace(var_x) )
#        eig_vals, eig_vecs =eigs(var_x,k=N_projections)   
#        eig_vals=np.real(eig_vals)
#        order=np.argsort(eig_vals)[::-1]
#        eig_vals=eig_vals[order]
#        eig_vecs=eig_vecs[:,order]
#        np.save(results_folder+'/pca/evecs_x_layer_'+layer_t,eig_vecs )
#        np.save(results_folder+'/pca/evals_x_layer_'+layer_t,eig_vals )
       
        
        print('pca layer '+layer+' in ' +str(time.time()-t)+' seconds')
        t = time.time()
    return
    
def eigdec(results_folder,test_layers,N_projections=1000):
    translation=string.maketrans('/', '_')
   
    t = time.time()  
    for layer in test_layers:
        layer_t=layer.translate(translation)
        
          
        var_dim1= np.load(results_folder+'/pca/var_dim1_layer_'+layer_t +'.npy')
        eig_vals, eig_vecs =eigs(var_dim1,  k=min(N_projections,var_dim1.shape[0]-2)) 
        eig_vals=np.real(eig_vals)
        order=np.argsort(eig_vals)[::-1]
        eig_vals=eig_vals[order]
        eig_vecs=eig_vecs[:,order]
        np.save(results_folder+'/pca/evecs_dim1_layer_'+layer_t,eig_vecs )
        np.save(results_folder+'/pca/evals_dim1_layer_'+layer_t,eig_vals)
       
        var=np.load(results_folder+'/pca/var_layer_'+layer_t+'.npy' )
        eig_vals, eig_vecs =eigs(var,k=min(N_projections,var.shape[0]-2)) 
        eig_vals=np.real(eig_vals)
        order=np.argsort(eig_vals)[::-1]
        eig_vals=eig_vals[order]
        eig_vecs=eig_vecs[:,order]
        np.save(results_folder+'/pca/evecs_layer_'+layer_t,eig_vecs)
        np.save(results_folder+'/pca/evals_layer_'+layer_t,eig_vals )
        
        var_dim2=np.load(results_folder+'/pca/var_dim2_layer_'+layer_t +'.npy')
        eig_vals, eig_vecs =eigs(var_dim2,k=min(N_projections,var_dim2.shape[0]-2)) 
        eig_vals=np.real(eig_vals)
        order=np.argsort(eig_vals)[::-1]
        eig_vals=eig_vals[order]
        eig_vecs=eig_vecs[:,order]
        np.save(results_folder+'/pca/evecs_dim2_layer_'+layer_t,eig_vecs)
        np.save(results_folder+'/pca/evals_dim2_layer_'+layer_t,eig_vals )
        
        var_x=np.load(results_folder+'/pca/var_x_layer_'+layer_t+'.npy')
        np.save(results_folder+'/pca/tr_var_x_layer_'+layer_t,np.trace(var_x) )
        eig_vals, eig_vecs =eigs(var_x, k=min(N_projections,var_x.shape[0]-2)) 
        eig_vals=np.real(eig_vals)
        order=np.argsort(eig_vals)[::-1]
        eig_vals=eig_vals[order]
        eig_vecs=eig_vecs[:,order]
        np.save(results_folder+'/pca/evecs_x_layer_'+layer_t,eig_vecs )
        np.save(results_folder+'/pca/evals_x_layer_'+layer_t,eig_vals )
       
        
        print('pca layer '+layer+' in ' +str(time.time()-t)+' seconds')
        t = time.time()
        
        
