import os
import imutils
import cv2
from skimage.metrics import structural_similarity as ssim


try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')



# load the two input images
image_orig = cv2.imread("test/cam1/6.jpg") ## Reference image ##
image_mod = cv2.imread("test/cam1/7.jpg")
currentframe = 0

# resize for faster processing
resized_orig = cv2.resize(image_orig, (300, 200))
resized_mod = cv2.resize(image_mod, (300, 200))
gray_orig = cv2.cvtColor(resized_orig, cv2.COLOR_BGR2GRAY)
gray_mod = cv2.cvtColor(resized_mod, cv2.COLOR_BGR2GRAY)

(score, diff) =ssim(gray_orig, gray_mod, full=True)
diff = (diff * 255).astype("uint8")
print("Structural Similarity Index: {}".format(score))
# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff,0, 2, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    print(area)
    if area > 4500:

        cv2.rectangle(resized_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(resized_mod, (x, y), (x + w, y + h), (0, 0, 255), 2)
        name = './data/frame' + str(currentframe) + '.jpg'
        # print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name,image_mod)
        currentframe +=1


    # show the output images
cv2.imshow("Original", resized_orig)
cv2.imshow("Modified", resized_mod)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)