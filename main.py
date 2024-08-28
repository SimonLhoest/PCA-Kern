# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:23:42 2022

@author: lhoes
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mabiblio as mb

n=10000 #Nombre de pixels
image=cv2.imread('immeuble.jpg') #Récupération de l'image
data_img = mb.data_pixels(image,mb.choixpts(image,n))
Xq,qqch=mb.ACPimg(data_img)
rien, qkayser = mb.ACP(data_img)
L,LL = mb.Kmoyimg(Xq,qkayser)
data_img = np.block([data_img,np.zeros((n,1))]) 
 
#Ajoute le libélé du groupe à chaque ligne de data_img, pour ensuite faire le masque  
for i in range(qkayser):
    for j in range(len(LL[i])):
        data_img[LL[i][j],8]=i

imgmasqueponctuel,colors = mb.Masque(data_img,qkayser,image)
imgmasque = mb.RemplissageMasque(imgmasqueponctuel)
#Affichage Masque et image 
cv2.imshow('masque',imgmasque)
cv2.imshow('image',image)
imgc=mb.affichage(image,imgmasque,colors)
#Affichages des couches de masque 
cv2.imshow('Couches',imgc)
cv2.imwrite('masque.jpg',imgmasque)
cv2.imwrite('couches.jpg',imgc)

#%% Equivalent en une fonction : 
import cv2
import mabiblio as mb

n=10000 #Nombre de pixels
image=cv2.imread('immeuble.jpg') #Récupération de l'image
mb.affichageglobal(image, n)
    