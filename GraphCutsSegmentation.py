import cv2
import numpy as np
from matplotlib import pyplot as plt
import canny
import Canny_algo
import os

#== Parameters
CurDir = os.getcwd()
ImgAdd = CurDir + '\Images\\'
chef=ImgAdd+'chef.jpg'
fox=ImgAdd+'fox.jpg'
dog=ImgAdd+'dog.jpg'
dog2=ImgAdd+'dog2.jpg'   # none effect
BLUR = 21
MASK_COLOR = (1.0,1.0,1.0) # In BGR format


#-- select one of the picture above fox/dog/chef/dog2
img = cv2.imread(fox)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection
edges = Canny_algo.mycanny(gray)
# edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)    # this will work better


#-- Find contours in edges, sort by area
contour_info = []
_,contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # we take only the contours
for c in contours:
    contour_info.append((
        c,
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[1], reverse=True) # sort by area
max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))


# Smooth mask, then blur it
# mask = cv2.dilate(mask, None)
# mask = cv2.erode(mask, None)
mask =canny.GaussBlur(mask)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
plt.imshow(mask_stack)
plt.show()

#-- Blend masked img into MASK_COLOR background
mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
img         = img.astype('float32') / 255.0                 #  for easy blending

masked = (mask_stack * img) + ((1-mask_stack) *MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('picture')
plt.imshow(img[:,:,::-1])

plt.subplot(122)
plt.axis('off')
plt.title('cut segmention')
plt.imshow(masked[:,:,::-1])


plt.show()

