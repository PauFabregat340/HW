# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:04:33 2022

@author: Pau
"""
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np


def PCA_custom(Xflat):
    meanV = np.mean(Xflat, axis=0)
    stdV = np.std(Xflat, axis=0)
    Xst = (Xflat-meanV) / stdV

    cov = np.cov(np.transpose(Xst))
    S, V, D = np.linalg.svd(cov)
    
    PCs = np.multiply(S,V)
    Xst_PC = np.dot(Xst, PCs)
    return Xst_PC


L8pre = gdal.Open(r'L8pre.tif')
L8burn = gdal.Open(r'L8burn.tif')
orig_size = (422, 613)

print(L8pre.RasterCount)
print(L8burn.RasterCount)


## L8pre
band1 = ((L8pre.GetRasterBand(1)).ReadAsArray()).flatten() # Red channel
band2 = ((L8pre.GetRasterBand(2)).ReadAsArray()).flatten() # Red channel
band3 = ((L8pre.GetRasterBand(3)).ReadAsArray()).flatten() # Red channel
band4 = ((L8pre.GetRasterBand(4)).ReadAsArray()).flatten() # Red channel
## L8burn
band11 = ((L8burn.GetRasterBand(1)).ReadAsArray()).flatten() # Red channel
band22 = ((L8burn.GetRasterBand(2)).ReadAsArray()).flatten() # Red channel
band33 = ((L8burn.GetRasterBand(3)).ReadAsArray()).flatten() # Red channel
band44 = ((L8burn.GetRasterBand(4)).ReadAsArray()).flatten() # Red channel

imgf = np.column_stack((band1, band2, band3, band4, band11, band22, band33, band44))
imgpre = imgf[:,:4]
imgburn = imgf[:,4:]

c_pre = np.dstack((imgpre[:,0].reshape(orig_size), imgpre[:,1].reshape(orig_size), imgpre[:,2].reshape(orig_size)))
c_burn = np.dstack((imgburn[:,0].reshape(orig_size), imgburn[:,1].reshape(orig_size), imgburn[:,2].reshape(orig_size)))

f = plt.figure()
plt.imshow(c_pre)
plt.show()

f = plt.figure()
plt.imshow(c_burn)
plt.show()


fig, axs = plt.subplots(2,4)
axs[0, 0].imshow((L8pre.GetRasterBand(1)).ReadAsArray())
axs[0, 1].imshow((L8pre.GetRasterBand(2)).ReadAsArray())
axs[0, 2].imshow((L8pre.GetRasterBand(3)).ReadAsArray())
axs[0, 3].imshow((L8pre.GetRasterBand(4)).ReadAsArray())
axs[1, 0].imshow((L8burn.GetRasterBand(1)).ReadAsArray())
axs[1, 1].imshow((L8burn.GetRasterBand(2)).ReadAsArray())
axs[1, 2].imshow((L8burn.GetRasterBand(3)).ReadAsArray())
axs[1, 3].imshow((L8burn.GetRasterBand(4)).ReadAsArray())


'''
## Standardize
meanV = np.mean(imgf, axis=0)
stdV = np.std(imgf, axis=0)
Xst = (imgf-meanV) / stdV

cov = np.cov(np.transpose(Xst))
S, V, D = np.linalg.svd(cov)
S = np.transpose(S)
score = 100 * ( V / np.sum(V) )

ratio = np.empty((6, 6))

for i in range(5):
    for j in range(5):
        ratio[i, j] = S[i, j] * np.sqrt(V[j]) / stdV[i]


cov_PCs = np.cov(S)     # Almost not correlated 

Xst_PC = np.dot(Xst, S)
'''
Xst_PC = PCA_custom(imgf)


Xst_false = np.dstack((Xst_PC[:,0].reshape(orig_size), Xst_PC[:,1].reshape(orig_size), Xst_PC[:,2].reshape(orig_size)))

f = plt.figure()
plt.imshow(Xst_PC[:,0].reshape(orig_size))
plt.show()

f = plt.figure()
plt.imshow(Xst_PC[:,1].reshape(orig_size))
plt.show()

f = plt.figure()
plt.imshow(Xst_PC[:,2].reshape(orig_size))
plt.show()

f = plt.figure()
plt.imshow(Xst_PC[:,3].reshape(orig_size))
plt.show()

f = plt.figure()
plt.imshow(Xst_PC[:,4].reshape(orig_size))
plt.show()

f = plt.figure()
plt.imshow(Xst_PC[:,5].reshape(orig_size))
plt.show()


f = plt.figure()
plt.imshow(Xst_false)
plt.savefig('False_composite_3PCs_ex2.png')
plt.show()

#var_3PCs = np.sum(score[:3]) / np.sum(score)