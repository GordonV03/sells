# import seg as seg
# from skimage.transform import rescale, resize, downscale_local_mean
#
# rescaled_img = rescale(img, 1.0/4.0, anti_aliasing=True)
# resized_img = resize(img, (200,200))
# downscaled_img = downscale_local_mean(img, (4, 3))
# plt.imshow(img)
# print(img.shape, resized_img.shape, rescaled_img.shape, downscaled_img.shape)

# from skimage.filters import roberts, sobel, scharr, prewitt

# edge_roberts = roberts(img)
# edge_sobel = sobel(img)
# edge_scharr = scharr(img)
# edge_prewitt = prewitt(img)

# fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))
#
# ax = axes.ravel()
#
# ax[0].imshow(img, cmap=plt.cm.gray)
# ax[0].set_title('Original image')
#
# ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
# ax[1].set_title('Roberts Edge Detection')
#
# ax[2].imshow(edge_roberts, cmap=plt.cm.gray)
# ax[2].set_title('Sobel')
# ax[3].imshow(edge_roberts, cmap=plt.cm.gray)
# ax[3].set_title('Scharr')
#
# for a in ax:
#     a.axis('off')
#
# plt.tight_layout()
# plt.show()
from skimage import io
from matplotlib import pyplot as plt
from skimage.feature import  canny
from skimage import restoration
import skimage.segmentation as seg
import skimage.color as color
import numpy as np

img = io.imread("0003.png", as_gray=True)
edge_canny_2 = canny(img, sigma=2)
# edge_canny_2_5 = canny(img, sigma=2.5)
# edge_canny_3 = canny(img, sigma=3)

psf = np.ones((3,3)) / 9

deconvolved_2, _ = restoration.unsupervised_wiener(edge_canny_2, psf)
# deconvolved_2_5, _ = restoration.unsupervised_wiener(edge_canny_2_5, psf)
# deconvolved_3, _ = restoration.unsupervised_wiener(edge_canny_3, psf)

plt.imsave('deconvolved_2.jpg', deconvolved_2, cmap='gray')
# plt.imsave('deconvolved_2_5.jpg', deconvolved_2_5, cmap='gray')
# plt.imsave('deconvolved_3.jpg', deconvolved_3, cmap='gray')
print(np.unique(deconvolved_2).size)
plt.imshow(deconvolved_2, cmap="gray")
plt.show()
