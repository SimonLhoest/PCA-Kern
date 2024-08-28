# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:51:20 2022

@author: lhoes
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import copy
import matplotlib.patches as mpatches

image=cv2.imread('test.jpeg') #Récupération de l'image

def ACPimg(data_img):
    return ACP(data_img,q=2)

def Kmoyimg(data_img,qkayser):
    return Kmoy(data_img,qkayser)

# Regle de Kaiser
def Kaiser(C):
    eigv = np.linalg.eigvals(C)
    somme = 1/np.shape(C)[0] * np.sum(eigv)
    for i in range(len(eigv)):
        if eigv[i] >= somme:
            q = i # Nombre de composantes principales a etutier
    return q + 1

# ACP 
def ACP(X, q=0):
    l,c = np.shape(X)
    somme = np.zeros((1,c))
    for i in range(l):
        somme = somme + X[i,:]
    indg = 1/l * somme
    Xc = X - indg
    Xcr = np.zeros((l,c))
    for j in range(c):
        norme = np.linalg.norm(X[:,j], 2)
        for i in range(l):
           if norme > 1e-20:
               Xcr[i,j] = Xc[i,j]/ norme
           else:
               Xcr[i,j] = 0
    C = Xcr.T @ Xcr
    U, S, VT =np.linalg.svd(C)
    if q== 0:
        q = Kaiser(C)
    Uq = U[:,:q]
    Xq = Xcr @ Uq
    Y = (Xcr @ U[:,0]).reshape((l,1))
    if q > 1:
        for i in range(1,q):
            Y = np.block([Y, (Xcr @ U[:,i]).reshape(l,1)])
    return Xq, q

def ACPpond(X, W,q=0):
    l,c = np.shape(X)
    somme = np.zeros((1,c))
    for i in range(l):
        somme = somme + X[i,:]
    indg = 1/l * somme
    Xc = X - indg
    Xcr = np.zeros((l,c))
    for j in range(c):
        norme = np.linalg.norm(X[:,j], 2)
        for i in range(l):
           if norme > 1e-20:
               Xcr[i,j] = Xc[i,j]/ norme
           else:
               Xcr[i,j] = 0
    Cw = 1/np.sum(W) * (Xcr @ np.sqrt(W)).T @ Xcr @ np.sqrt(W)
    U, S, VT =np.linalg.svd(Cw)
    if q == 0:
        q = Kaiser(Cw)
    Uq = U[:,:q]
    Xq = Xcr @ Uq
    Y = (Xcr @ U[:,0]).reshape((l,1))
    if q > 1:
        for i in range(1,q):
            Y = np.block([Y, (Xcr @ U[:,i]).reshape(l,1)])
    return Xq,q


#Kmoyenne
def Kmoy(A, k, epsilon=10e-20) :
    # Pour une matrice de taille 2 minimum pour que l'algorithme fonctionne
    if len(A)<=2 : 
        print("Ensemble trop petit")
        return False,False
    else : 
        ligne,colonne=np.shape(A)
        Aavecind=np.concatenate((np.arange(0,ligne,1)[np.newaxis].T,A), axis=1)
        LL = [[] for x in range(0,k)]
        LLind = [[] for x in range(0,k)]
        # Tableau des indices aléatoires (un indice est unique!)
        ind=np.random.randint(0,ligne,k)  
        testunicite=len(ind)==len(np.unique(ind))
        # Matrice barycentre
        mu=A[ind]
        # Matrice de centrage
        cent=np.zeros(np.shape(mu))
        normtest=np.zeros((ligne,k))
        l=0 # Indices dans les futures listes
        # Nombre d'iterations faites
        compteur=0
        if testunicite==True : # Indices tous bien uniques
            # Iterations 
            while np.linalg.norm(cent-mu)>epsilon :
                cent=copy.deepcopy(mu)
                # Listes de listes
                LL = [[] for x in range(0,k)]
                LLind = [[] for x in range(0,k)]
                # Partitions avec A[i,:] plus proche de de mu(compteur)
                for r in range(0, k) :
                    for i in range(0,ligne) :
                        normtest[i,r]=np.linalg.norm(A[i,:]-mu[r,:])
                # Sauvegarde des partitions
                for i in range(0,ligne) :
                    l=np.where(normtest[i,:]==np.amin(normtest[i,:]))[0][0]
                    LL[l].append(A[i,:])
                    LLind[l].append((Aavecind[i,0]).astype('int'))
                # Nouveaux barycentres
                for r in range(0,k) :
                    s=0
                    for j in range(0,len(LL[r])) :
                        s=s+LL[r][j]
                    mu[r]=(1/len(LL[r]))*s
                compteur=compteur+1
        else : # Relance la fonction pour avoir des indices uniques
              Kmoy(A,k,epsilon)
        return LL  ,LLind

'''
FONCTIONS PARTIE 3
'''
def choixpts(img,nbpts):
    #Génération au hasard de nbpts points
    a,b,c = img.shape
    L=[(random.randint(1,a-1),random.randint(1,b-1)) for x in range(nbpts)]
    return L

def data_pixels(img,L):
    #Construction de data_img, comme indiqué dans le sujet
    a,b,c = img.shape
    M = np.zeros((len(L),8))
    for i in range (len(L)):
        M[i,:2]=np.array([L[i]]) #indice i j 
        M[i,2:5]=img[L[i]]  # couches de couleur 
        moy=img[L[i][0]-1:L[i][0]+2,L[i][1]-1:L[i][1]+2] #Les 8 pixels autour + le pixel
        M[i,5:]=np.array([np.mean(moy[:,:,0]), np.mean(moy[:,:,1]), np.mean(moy[:,:,2])]) #Moyenne des 9
    return M #data_img

def Masque(S,qkayser):
    M=np.zeros(image.shape)
    colors = np.zeros((3,qkayser))
    for i in range (qkayser):
        #Couleur au hasard pour chaque groupe 
        colors[:,i] = np.array([[random.randint(0,255)],
                         [random.randint(0,255)],
                         [random.randint(0,255)]]).reshape((3,))
    for i in range (S.shape[0]):
        #Assigne la couleur au groupe
        M[int(S[i][0]),int(S[i][1]),:] = colors[:,int(S[i][8])].reshape((3,))
              
    return M, colors #imgmasqueponctuel

def RemplissageMasque(imgmasqueponctuel):
    imgmasqueponctuel = imgmasqueponctuel.astype('uint8') #Doit convertir en uint8 pour les fonctions suivantes
    mask = cv2.cvtColor(imgmasqueponctuel, cv2.COLOR_BGR2GRAY)
    mask[mask>0]=255
    mask = 255*np.ones(np.shape(mask))-mask
    imgmasque = cv2.inpaint(imgmasqueponctuel,np.uint8(mask),3,cv2.INPAINT_NS)
    return imgmasque

def affichage(img,imgmasque,colors):
    #Affiche l'image pour chaque masque
    img2=img.copy()
    mask = (255*np.ones(img.shape)).astype('uint8')
    mask[imgmasque==colors[:,0]]=0
    img2[imgmasque!=colors[:,0]]=0
    mask = mask + img2
    imgs=mask
    for i in range(1,colors.shape[1]):
        img2=img.copy()
        mask = (255*np.ones(img.shape)).astype('uint8')
        mask[imgmasque==colors[:,i]]=0
        img2[imgmasque!=colors[:,i]]=0
        mask = mask + img2
        imgs=np.concatenate([imgs,mask],axis=1)
    return imgs
        
            
#%% Normal
data_img = data_pixels(image,choixpts(image,20000))
Xq,delamerde=ACPimg(data_img)
rien, qkayser = ACP(data_img)
L,LL = Kmoyimg(Xq,qkayser)
data_img = np.block([data_img,np.zeros((20000,1))]) 
 
#Ajoute le libélé du groupe à chaque ligne de data_img, pour ensuite faire le masque  
for i in range(qkayser):
    for j in range(len(LL[i])):
        data_img[LL[i][j],8]=i

imgmasqueponctuel,colors = Masque(data_img,qkayser)
imgmasque = RemplissageMasque(imgmasqueponctuel)
cv2.imshow('masque',imgmasque)
cv2.imshow('image',image)
imgc=affichage(image,imgmasque,colors)
cv2.imshow('Couches',imgc)

#%%Pondéré 
'W aléatoir pour utilisation avec ACP pondéré'
W=np.diag([random.randint(1,image.shape[1]/10) for x in range(8)])
