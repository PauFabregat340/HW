# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:43:14 2022

@author: Pau
"""

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

orig_size = (1294,1734)

dataset = gdal.Open(r'yukon.tif')
print(dataset.RasterCount)
# since there are 3 bands
# we store in 3 different variables
band1 = dataset.GetRasterBand(1) # Red channel
band2 = dataset.GetRasterBand(2) # Green channel
band3 = dataset.GetRasterBand(3) # Blue channel
band4 = dataset.GetRasterBand(4) # NIR
band5 = dataset.GetRasterBand(5) # Green channel
band6 = dataset.GetRasterBand(6) 
 # Blue channel
b1 = band1.ReadAsArray()
b2 = band2.ReadAsArray()
b3 = band3.ReadAsArray()
b4 = band4.ReadAsArray()
b5 = band5.ReadAsArray()
b6 = band6.ReadAsArray()

#img = np.dstack((b1, b2, b3))

fig, axs = plt.subplots(2,3)
axs[0, 0].imshow(b1)
axs[0, 1].imshow(b2)
axs[0, 2].imshow(b3)
axs[1, 0].imshow(b4)
axs[1, 1].imshow(b5)
axs[1, 2].imshow(b6)


## Flatten images
b1f = b1.flatten()
b2f = b2.flatten()
b3f = b3.flatten()
b4f = b4.flatten()
b5f = b5.flatten()
b6f = b6.flatten()

imgf = np.column_stack((b1f,b2f,b3f,b4f,b5f,b6f))

fig, axs = plt.subplots(2,3)
axs[0, 0].hist(imgf[:,0],bins=50)
axs[0, 1].hist(imgf[:,1],bins=50)
axs[0, 2].hist(imgf[:,2],bins=50)
axs[1, 0].hist(imgf[:,3],bins=50)
axs[1, 1].hist(imgf[:,4],bins=50)
axs[1, 2].hist(imgf[:,5],bins=50)



## Standardize
meanV = np.mean(imgf, axis=0)
stdV = np.std(imgf, axis=0)
Xst = (imgf-meanV) / stdV

cov = np.cov(np.transpose(Xst))
S, V, D = np.linalg.svd(cov)

score = 100 * ( V / np.sum(V) )

ratio = np.empty((6, 6))

for i in range(5):
    for j in range(5):
        ratio[i, j] = S[i, j] * np.sqrt(V[j]) / stdV[i]


cov_PCs = np.cov(S)     # Almost not correlated 

#imgf_PC = np.dot(imgf, S)
Xst_PC = np.dot(Xst, S)
#Xst_PC = np.dot(Xst, np.transpose(S))


## False color composite image

#img_false = np.dstack((imgf_PC[:,0].reshape(orig_size), imgf_PC[:,1].reshape(orig_size), imgf_PC[:,2].reshape(orig_size)))
Xst_false = np.dstack((Xst_PC[:,0].reshape(orig_size), Xst_PC[:,1].reshape(orig_size), Xst_PC[:,2].reshape(orig_size)))

f = plt.figure()
plt.imshow(Xst_false)
plt.savefig('False_composite_3_PCs.png')
plt.show()

var_3PCs = np.sum(score[:3]) / np.sum(score)

fig, axs = plt.subplots(2,3)
axs[0, 0].hist(Xst_PC[:,0],bins=50)
axs[0, 1].hist(Xst_PC[:,1],bins=50)
axs[0, 2].hist(Xst_PC[:,2],bins=50)
axs[1, 0].hist(Xst_PC[:,3],bins=50)
axs[1, 1].hist(Xst_PC[:,4],bins=50)
axs[1, 2].hist(Xst_PC[:,5],bins=50)


f = plt.figure()
plt.imshow(Xst_PC[:,1].reshape(orig_size))
plt.savefig('False_composite_PC2.png')
plt.show()