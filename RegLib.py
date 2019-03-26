# -*- coding: utf-8 -*-

import os
import scipy
import scipy.ndimage
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as scim


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#FONCTIONS GENERALES
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class OpImage:
    
    def __init__(self,FileName):
        """
        Options:
          FileName is the file name, plus its path. It can be either a nifti image or a csv file
          UseNN: =1 -> nearest neighbor interpolation is used to get the intensities (default) /  =0 trilinear interpolation is used otherwise
          frame: -> frame loaded in case we have a 3D+t nifti image /  all frames are imported if = -1
        """
        
        self.InitFileName=os.path.join(FileName)
        
        self.data=scim.imread(FileName)[:,:,0]*1.
        self.data=self.data.max()-self.data

        
        
    def show(self,LabelImage=''): 
        """
        Show the image
        """
        plt.imshow(self.data,cmap='Greys')
        plt.title(LabelImage)
        plt.colorbar()
        plt.show()
    
    def SaveImage(self,LabelImage='',filename='toto.png'): 
        """
        Show the image
        """
        plt.imshow(self.data,cmap='Greys')
        plt.title(LabelImage)
        plt.colorbar()
        plt.savefig(filename)
        plt.clf()

    
    def CompareWithAnotherImage(self,ComparedImage,LabelSelfIm='Im1',LabelComparedIm='Im2',ShowAll=0):
      if ShowAll!=0:
        plt.figure(1)
        plt.imshow(self.data,cmap='Greys')
        plt.title(LabelSelfIm)
        plt.colorbar()
        
        plt.figure(2)
        plt.imshow(ComparedImage.data,cmap='Greys')
        plt.title(LabelComparedIm)
        plt.colorbar()
        
        plt.figure(3)
      plt.imshow(self.data-ComparedImage.data,cmap='Greys')
      plt.title('('+LabelSelfIm+')  -  ('+LabelComparedIm+')')
      plt.colorbar()
      plt.show()
      
    def SaveComparisonWithAnotherImage(self,ComparedImage,LabelSelfIm='Im1',LabelComparedIm='Im2',filename='toto.png'):
      plt.imshow(self.data-ComparedImage.data,cmap='Greys')
      plt.title('('+LabelSelfIm+')  -  ('+LabelComparedIm+')')
      plt.colorbar()
      plt.savefig(filename)
      plt.clf()
    
            
    def get(self,x,y): 
        """
        Get the image intensity at (x,y,z).
        """
        
        shape=(self.data).shape
        xNN=int(x+0.5)
        yNN=int(y+0.5)
        if xNN<0:
          xNN=0
        if xNN>=shape[0]:
          xNN=shape[0]-1
        if yNN<0:
          yNN=0
        if yNN>=shape[1]:
          yNN=shape[1]-1
        result=self.data[xNN,yNN]
                
        return(float(result))
    
    
    def put(self,value,x,y):     
        """
        Set intensty 'value' at point x,y
        """
        self.data[x,y]=value
    
    def putToAllPoints(self,InputArray):        
        """
        Define the intensities of an image using a 3D array.
        """
        self.data[:,:]=InputArray
    
    def size(self):
        """          
        Return the image size
        """
        result=self.data.shape
        return(result)
        
               
    def GaussianFiltering(self,stddev):
        """
        Gaussian filtering of the image with standard deviation stddev (4d vector in voxels)
        """
        self.data[:]=scipy.ndimage.gaussian_filter(self.data, sigma=stddev,mode='constant', cval=0.0)
        
    def data(self):
        """
        Return the array corresponding to the image intensities
        """
        return self.data
    
    def grad(self):
        """
        Return the image gradients
        """
        gradx,grady=np.gradient(self.data[:,:])
        
        return gradx,grady
        



def Cpt_SSD(imgResampled,imgFixed):
    SSD=((imgFixed.data-imgResampled.data)*(imgFixed.data-imgResampled.data)).sum()
    
    return SSD


def GenerateNullDisplacementField(ImFile):
    DefX=OpImage(ImFile)
    DefX.putToAllPoints(0.)

    DefY=OpImage(ImFile)
    DefY.putToAllPoints(0.)
    
    return DefX,DefY


def TransportImage(DFx,DFy,img,img_resampled):
    """
    Transport image img to img_resampled using the displacement field DF. 
    * img_resampled is in the same image domain as img
    * DF is a displacement from img_resampled to img
    """
    for i in range(img_resampled.size()[0]):
      for j in range(img_resampled.size()[1]):
          img_resampled.put( img.get(i+DFx[i,j],j+DFy[i,j]) , i,j)


def TranslateAndRotateImage(RCX,RCY,theta,dx,dy,img,img_resampled):
    """
    Rotate img to img_resampled using the deformation parameters. 
    * img_resampled is in the same image domain as img
    * DF is a displacement from img_resampled to img with these parameters
      -> RCX,RCY:  Rotation center
      -> theta:  Rotation angle
      -> dx,dy:  Translation parameters
    """
    M1=np.eye(3)
    M2=np.eye(3)
    M3=np.eye(3)
    M1[0,2]=RCX
    M1[1,2]=RCY
    M2[0,0]=np.cos(theta)
    M2[0,1]=-np.sin(theta)
    M2[1,0]=np.sin(theta)
    M2[1,1]=np.cos(theta)
    M3[0,2]=-RCX
    M3[1,2]=-RCY
    M4=np.dot(M2,M3)
    RotMat=np.dot(M1,M4)
    for i in range(img_resampled.size()[0]):
      for j in range(img_resampled.size()[1]):
        rsp_ij=np.dot(RotMat,np.array([i+dx,j+dy,1.]))
        img_resampled.put( img.get(rsp_ij[0],rsp_ij[1]) , i,j)
    



