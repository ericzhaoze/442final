from PIL import Image, ImageFilter
import cv2
# import the necessary packages
import numpy as np
import argparse
import glob
 
size = 256, 256

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True,
# 	help="path to input dataset of images")
# args = vars(ap.parse_args())
 
# loop over the images
# for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# load the image, convert it to grayscale, and blur it slightly

im = Image.open("2.jpg")
im.thumbnail(size, Image.ANTIALIAS)
im.save("22.jpg")

image = cv2.imread("22.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# blurred = cv2.bilateralFilter(gray,9,75,75)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = 255 - cv2.Canny(blurred, 10, 200)
tight = 255 - cv2.Canny(blurred, 225, 250)
auto = 255 - auto_canny(blurred)

# show the images
cv2.imshow("Original", image)
cv2.imshow("Edges", wide)
cv2.waitKey(0)