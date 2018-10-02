import skimage.io as io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import os

root_dir = "../../images"
IMAGE_LOCATION = os.path.join(root_dir, "cat.jpg")
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION))
img = img.astype(np.float32)

#
# show the image in RGB and BGR
#
plt.figure("RGB and BGR images")
ax = plt.subplot(1, 2, 1)
ax.set_title('Original image = RGB')
ax.imshow(img)
ax.axis('off')

imgBGR = img[:, :, (2, 1, 0)]
ax = plt.subplot(1, 2, 2)
ax.set_title('OpenCV, Caffe2 = BGR')
ax.imshow(imgBGR)
ax.axis('off')
plt.show()

#
# Rotated image
#
ROTATED_IMAGE = os.path.join(root_dir, "cell-tower.jpg")
imgRotated = skimage.img_as_float(skimage.io.imread(ROTATED_IMAGE))
imgRotated = imgRotated.astype(np.float32)
plt.figure("Rotated images")
ax = plt.subplot(1, 2, 1)
ax.set_title("Rotated image {}".format(imgRotated.shape))
ax.imshow(imgRotated)
ax.axis("off")

img = np.rot90(imgRotated, 3)
ax = plt.subplot(1, 2, 2)
ax.set_title("Normal image {}".format(img.shape))
ax.imshow(img)
ax.axis("off")
plt.show()

#
# Mirror image
#
MIRROR_IMAGE = os.path.join(root_dir, "mirror-image.jpg")
imgMirror = skimage.img_as_float(skimage.io.imread(MIRROR_IMAGE))
imgMirror = imgMirror.astype(np.float32)
plt.figure("Mirror images")
ax = plt.subplot(1, 2, 1)
ax.set_title("Mirror image")
ax.imshow(imgMirror)
ax.axis("off")

imgMirror = np.fliplr(imgMirror)
ax = plt.subplot(1, 2, 2)
ax.set_title("Normal image")
ax.imshow(imgMirror)
ax.axis("off")
plt.show()

#
# Resize image
#
scale = 0.01
h, w, _ = img.shape
scaled_img = skimage.transform.resize(img, (scale * h, scale * w))
plt.figure("Resized image")
ax = plt.subplot(1, 2, 1)
ax.set_title("Original Image {}".format(img.shape))
ax.imshow(img)
ax.axis("on")

ax = plt.subplot(1, 2, 2)
ax.set_title("Resized Image {}".format(scaled_img.shape))
ax.imshow(scaled_img)
ax.axis("on")

plt.tight_layout()
plt.show()

#
# Cropping image
#
IMAGE_LOCATION = os.path.join(root_dir, "cat.jpg")
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION))
img = img.astype(np.float32)
img256 = skimage.transform.resize(img, (256, 256))


def crop_center(img, cropx, cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


plt.figure("Cropping image")
imgCenter = crop_center(img, 224, 224)
ax = plt.subplot(1, 2, 1)
ax.set_title("Original")
ax.imshow(imgCenter)
ax.axis("on")

img256Center = crop_center(img256, 224, 224)
ax = plt.subplot(1, 2, 2)
ax.set_title("Squeezed")
ax.imshow(img256Center)
ax.axis("on")
plt.show()
