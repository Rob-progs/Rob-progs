# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
from skimage.filters import threshold_local


########################################################################################################################
# https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
# order_points and four_point_transform from:
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


########################################################################################################################

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


########################################################################################################################
# https://www.murtazahassan.com/courses/opencv-projects/lesson/code-4/
def biggestContour(image, contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > (0.3 * image.shape[0] * image.shape[1]):
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


########################################################################################################################
# https://www.murtazahassan.com/courses/opencv-projects/lesson/code-4/
def drawRectangle(img, biggest, thickness):  # All 2s and 3s are switched
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[3][0][0], biggest[3][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[2][0][0], biggest[2][0][1]), (biggest[3][0][0], biggest[3][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[2][0][0], biggest[2][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img


########################################################################################################################

def auto_canny(image, sigma=0.5):  # Sigma changed to 0.5
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged, lower, upper


########################################################################################################################

def resize_image(image, resized_height):
    scale_percent = float(resized_height / image.shape[0] * 100)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


########################################################################################################################


# PROGRAM START
print("STEP 1: Load and resize image")
image = cv2.imread('C:/Users/robin/Desktop/ComputerVision/Noisy.jpg')
print('Original Dimensions : ', image.shape)
# image = sp_noise(image, 0.01)
# resizing
resized_height = 500.0
resized = resize_image(image, resized_height)

cv2.imshow("Image", resized)
print('Resized Dimensions : ', resized.shape)
ratio = image.shape[0] / resized_height
orig = image.copy()

# convert the image to grayscale, blur it, and find edges
# in the image
print("STEP 2: Convert image into grayscale and blur it.")
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow("Blured", blur)

print("STEP 5: Find edges with automatic canny edge detector.")
edged, lower_thres, upper_thres = auto_canny(blur)
# show the histogram and threshold values of the gray-scaled image
plt.hist(blur.ravel(), 256, [0, 256])
plt.axvline(lower_thres, color='r', linestyle='solid', linewidth=1)
plt.axvline(upper_thres, color='r', linestyle='solid', linewidth=1)
plt.show()
# show the detected edges of the image
print("Canny threshold values:", lower_thres, ", ", upper_thres)
cv2.imshow("Edged", edged)

## FIND ALL COUNTOURS
print("STEP 6: Find all contours")
imgContours = resized.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = resized.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)  # DRAW ALL DETECTED CONTOURS
cv2.imshow("Document contours", imgContours)

# FIND THE BIGGEST COUNTOUR
print("Step 7: Find the corner points of the biggest quad.")
biggest, maxArea = biggestContour(resized, contours)  # FIND THE BIGGEST CONTOUR
if biggest.size == 0:
    print("No document found. Close windows to exit program.")
    cv2.waitKey(0)
    exit()

imgBigContour = cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
imgBigContour = drawRectangle(imgBigContour, biggest, 2)
print("Corner points detected at 2D pixel positions:", biggest[0], biggest[1], biggest[2], biggest[3])
cv2.imshow("Outlined with Points", imgBigContour)

print("STEP 8: Apply a perspective transformation of the original full size image.")
# apply the four point transform to obtain a top-down
# view of the original image
points = np.float32(biggest)
warped = four_point_transform(orig, points.reshape(4, 2) * ratio)
output = warped.copy()

print("STEP 9: Grayscale and blur the transformed image to avoid information loss.")
# convert the warped image to grayscale, then blur it.
# to give it that 'black and white' paper effect
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_gray_blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)

print("STEP 10: Apply Binarization")
# Apply the sauvola thresholding algorithm
thresh_sauvola = threshold_sauvola(warped_gray_blur, window_size=25)
th = warped_gray_blur > thresh_sauvola

# convert type
th = np.where(th, 255, 0).astype(np.uint8)

# Other thresholding methods
# thresh_niblack = threshold_niblack(warped_gray_blur, window_size=25, k=0.8)
# th = warped_gray_blur > thresh_niblack
# th = np.where(th, 255,0).astype(np.uint8)

# th = cv2.threshold(warped_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)  # BINARIZATION WITH TRIANGLE THRESHOLDING

# th = cv2.threshold(warped_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # BINARIZATION WITH OTSUS THRESHOLDING

# th = cv2.adaptiveThreshold(warped_gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\cv2.THRESH_BINARY,11,10)

# T = threshold_local(warped_gray_blur, 11, offset = 10, method = "gaussian")
# th = (warped_gray_blur > T).astype("uint8") * 255

print("STEP 11: Downscale the scan to the wanted size")
output_pixel_height = 900
output = resize_image(output, output_pixel_height)
cv2.imshow("Original", output)
th = resize_image(th, output_pixel_height)
cv2.imshow("Scanned", th)

# Save image
filename = 'savedImage.jpg'
cv2.imwrite(filename, th)

cv2.waitKey(0)
