import os
import glob
import numpy as np
from operator import add
import math
from scipy import misc
import scipy.io
from skimage.color import lab2rgb
import caffe

class ImageGenerator:
## Provides images for the analysis using two directions of variation.
# Is initialized with parameters defining the kinds of variations expected in 
# the images (parameter['variation_name']) and if necessary a folder containing images (parameter['data_folder'])
# These variations can either use rendered CAD models previously generated or be synthetic 2D images. 
# In all cases, this class implement:
# - generator(): generate a set of images with a fixed value ofparameter_1 and all values of parameter_2 varying
# - recover(i,j): recover the image with parameter_1=i and parameter2=j
# - renew(): reinitialize generator()
# - N1: number of values of parameter_1
# - N2: number of values of parameter_2 

    def __init__(self,parameters):
        self.generated_id=0
        try:
            var=parameters['variation_name']
        except :
            raise KeyError('You must define parameters.variation_name')


        if var=='translationxy':
            self.shape=[300,300,3]
            self.generator=self.generate_translationxy
            self.recover=self.recover_translationxy
            self.folder=parameters['data_folder']
            self.subfolders=os.listdir(self.folder)
            self.subfolders.sort()
            self.N1=len(self.subfolders)
            self.max_x=80 # maximum translation
            self.min_x=-self.max_x
            Ntrans=6 # number of translations to generate in each dimension
            coo=np.tile(np.linspace(self.min_x,self.max_x,Ntrans),[Ntrans,1])
            N=coo.size
            self.N2=N
            self.coo=np.zeros([N,2]) # coordinates of the translation
            self.coo[:,0] = coo.transpose().flatten()
            self.coo[:,1] = coo.flatten()
            self.crop=-40 # crop of the rendered images
            self.N2=N

        elif var=='rectangle':
            self.shape=[150,150,3]
            self.generator=self.generate_rectangle
            self.recover=self.recover_rectangle
            self.max_x=30
            self.min_x=-self.max_x
            Ntrans=6
            coo=np.tile(range(self.min_x,self.max_x,Ntrans),[Ntrans,1])
            N=coo.size
            basic_half_size=20
            alpha=np.array(range(-6,7))
            alpha=alpha/6.0
            alpha=2**alpha
            self.size=np.zeros([alpha.size,2]) # size of the rectangle
            self.size[:,0]=np.round(basic_half_size*alpha)
            self.size[:,1]=np.round(basic_half_size*alpha[::-1])
            self.coo=np.zeros([N,2]) # coordinates of the translation
            self.coo[:,0] = coo.transpose().flatten()+self.shape[0]/2
            self.coo[:,1] = coo.flatten()+self.shape[1]/2
            self.coo=np.round(self.coo)
            self.size=np.round(self.size)
            self.N2=N
            self.N1=self.size.shape[0]

        elif var=='subfolders':
            self.generator=self.generate_subfolders
            self.recover=self.recover_subfolders
            self.folder=parameters['data_folder']            
            self.subfolders=os.listdir(self.folder)
            self.N1=len(self.subfolders)
            self.subfolders.sort()
            self.crop=10
            images_names=glob.glob(self.folder+'/'+self.subfolders[self.generated_id]+'/*.jpg')
            self.N2=len(images_names)
            self.N2_data=self.N2

        elif var=='scale':
            self.shape=[300,300,3]
            self.generator=self.generate_scale
            self.recover=self.recover_scale        
            self.folder=parameters['data_folder']            
            self.subfolders=os.listdir(self.folder)
            self.N1=len(self.subfolders)
            self.subfolders.sort()
            self.step=-3
            self.min=120
            self.max=0
            self.N2=len(range(self.min,self.max,self.step))

        elif var=='color':
            self.generator=self.generate_color
            self.recover=self.recover_color
            self.N2=121
            self.N1=11
            self.generated_id=0

        elif var=='bicolor':
            self.generator=self.generate_bicolor
            self.recover=self.recover_bicolor
            self.Ncols=5
            coo=np.tile(np.linspace(0,1,self.Ncols),[self.Ncols,1])
            self.colors=np.zeros((self.Ncols**3,3))
            self.colors[:,0] =np.tile( coo.transpose().flatten(),[self.Ncols,1]).transpose().flatten()
            self.colors[:,1] = np.tile( coo.transpose().flatten(),[self.Ncols,1]).flatten()
            self.colors[:,2] = np.tile(np.linspace(0,1,self.Ncols),[self.Ncols**2])
            self.N2=self.Ncols**3
            self.N1=self.Ncols**3
            self.generated_id=0 

        else:
            raise NameError('The variation parameter \''+ var +'\' does not exist' )
         
         
    def renew(self):
        self.generated_id=0    
     
    
        
    def generate_color(self):
        results=[]
        for a in range(0,11,1):
            for b in range(0,11,1):
                I=np.ones([10,10,3], dtype=np.float32)
                I[:,:,0]=self.generated_id/10.0
                I[:,:,1]=a/10.0
                I[:,:,2]=b/10.0
                results.append(I)
        self.generated_id=self.generated_id+1
        return results
   
    def recover_color(self,i,j):
        L=len(range(0,11,1))
        I=np.ones([10,10,3], dtype=np.float32)
        I[:,:,0]=i/10.0
        a,b=np.unravel_index(j,[L,L])
        I[:,:,1]=a/10.0
        I[:,:,2]=b/10.0
        return I
                
    def generate_bicolor(self):
        results=[]
        I=np.ones([20,20,3], dtype=np.float32)
        I[:,:,0]=self.colors[self.generated_id,0]
        I[:,:,1]=self.colors[self.generated_id,1]
        I[:,:,2]=self.colors[self.generated_id,2]
        for id in range(0,self.N2):
                I[5:15,5:15,0]=self.colors[id,0]
                I[5:15,5:15,1]=self.colors[id,1]
                I[5:15,5:15,2]=self.colors[id,2]
                results.append(np.copy(I))
        self.generated_id=self.generated_id+1
        return results

    def recover_bicolor(self,d,id):
        I=np.ones([20,20,3], dtype=np.float32)
        I[:,:,0]=self.colors[d,0]
        I[:,:,1]=self.colors[d,1]
        I[:,:,2]=self.colors[d,2]
        I[5:15,5:15,0]=self.colors[id,0]
        I[5:15,5:15,1]=self.colors[id,1]
        I[5:15,5:15,2]=self.colors[id,2]
        return I   
  
    def generate_rectangle(self):
        results=[]
        s=self.size[self.generated_id]
        for i in range(self.coo.shape[0]):
            I=np.ones(self.shape)
            I[self.coo[i,0]-s[0]:self.coo[i,0]+s[0],self.coo[i,1]-s[1]:self.coo[i,1]+s[1],:]=0
            results.append(I)
        self.generated_id=self.generated_id+1
        return results
     
    def recover_rectangle(self,i1,i):
        s=self.size[i1]
        I=np.ones(self.shape)
        I[self.coo[i,0]-s[0]:self.coo[i,0]+s[0],self.coo[i,1]-s[1]:self.coo[i,1]+s[1],:]=0
        return I

    def init_translationxy(self):
        self.images_names=glob.glob(self.folder+'/'+self.subfolders[self.generated_id]+'/*.jpg')
        self.images_names.sort()
        I0=caffe.io.load_image(self.images_names[0])
        crop=self.crop
        if crop>0:
            self.I=self.I[crop:-crop, crop:-crop,:]
        if crop<=0:
             self.I=np.ones(self.shape,dtype=np.float32)
             self.I[-crop:crop, -crop:crop,:]=caffe.io.resize_image(I0, (self.shape[0]+2*crop,self.shape[1]+2*crop)) 
        I=np.ones(shape=map(add,self.I.shape,[2*self.max_x,2*self.max_x,0]),dtype=np.float32)
        I[self.max_x:self.max_x+self.I.shape[0],self.max_x:self.max_x+self.I.shape[1],:]=self.I
        self.I=I
        self.current_x=0
        self.generated_id=self.generated_id+1

    def generate_translationxy(self):
        results=[]
        self.init_translationxy()
        while (self.current_x< self.N2):
            I=self.I[self.max_x+self.coo[self.current_x,0]:self.max_x+self.shape[0]+self.coo[self.current_x,0],self.max_x+self.coo[self.current_x,1]:self.max_x+self.shape[1]+self.coo[self.current_x,1],:]
            results.append(I)
            self.current_x=self.current_x+1
        return results
       

    def recover_translationxy(self,i1,i2):
        self.generated_id=i1 
        self.init_translationxy()
        self.current_x=i2
        I=self.I[self.max_x+self.coo[self.current_x,0]:self.max_x+self.shape[0]+self.coo[self.current_x,0],self.max_x+self.coo[self.current_x,1]:self.max_x+self.shape[1]+self.coo[self.current_x,1],:]
       
        return I
    
    def generate_scale(self):
        images_names=glob.glob(self.folder+'/'+self.subfolders[self.generated_id]+'/*.jpg')
        images_names.sort()
        I0=caffe.io.load_image(images_names[0])
        results=[]
        for crop in range(self.min,self.max,self.step):
            I=np.ones(self.shape,dtype=np.float32)
            I[crop:-crop, crop:-crop,:]=caffe.io.resize_image(I0, (self.shape[0]-2*crop,self.shape[1]-2*crop))   
            results.append(I)
        self.generated_id=self.generated_id+1
        return results

    def recover_scale(self,i1,crop_id):
        images_names=glob.glob(self.folder+'/'+self.subfolders[i1]+'/*.jpg')  
        images_names.sort()
        tmp=range(self.min,self.max,self.step)
        crop=tmp[crop_id]
        I0=caffe.io.load_image(images_names[0])  
        I=np.ones(self.shape,dtype=np.float32)
        I[crop:-crop, crop:-crop,:]=caffe.io.resize_image(I0, (self.shape[0]-2*crop,self.shape[1]-2*crop)) 
        return I
   
    def generate_subfolders(self):
        results=[]
        self.images_names=glob.glob(self.folder+'/'+self.subfolders[self.generated_id]+'/*.jpg')
        self.images_names.sort()
        crop=self.crop
        for var_id in range(0,self.N2):
            I=caffe.io.load_image(self.images_names[var_id])
            if crop>0:
                I=I[crop:-crop, crop:-crop,:]
            results.append(I)
        self.generated_id=self.generated_id+1
        return results
   
    def recover_subfolders(self,i1,i2):
        crop=self.crop
        images_names=glob.glob(self.folder+'/'+self.subfolders[i1]+'/*.jpg')  
        images_names.sort()
        I=caffe.io.load_image(images_names[i2])
        I=I[crop:-crop, crop:-crop,:]
        return I
        
