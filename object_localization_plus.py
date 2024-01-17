import cv2
import numpy as np

def show(image):
    cv2.imshow("temp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = ".\sample2.jpg"
img = cv2.imread(path)


#BGT 2 Gray
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Blur
blur_image = cv2.bilateralFilter(gray_image,9,75,75)

#Gray 2 canny with edge
canny_image = cv2.Canny(blur_image, 0, 200)


#Dilated edge image
kernel = np.ones((5, 5), np.uint8)
dilated_img = cv2.dilate(canny_image, kernel, iterations=1)
erode_img = cv2.erode(dilated_img, kernel, iterations=1)

#Fill the edge 
filled_image = np.zeros_like(gray_image)
contours, _ = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)
show(filled_image)

#-----------------------------------------------------------------------------------------------------------

thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
conv_image = cv2.erode(thresholded_image, (3,3), iterations=3)
show(conv_image)
gau_image =cv2.bilateralFilter(conv_image,9,75,75)
contours1, _ = cv2.findContours(gau_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#----------------------------------------------------------------------------------------------------------

re_image = np.logical_or(conv_image, filled_image)
re_image = re_image.astype(np.uint8) * 255
contours2, _ = cv2.findContours(re_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
re_image = cv2.bilateralFilter(re_image,9,75,75)
re_image = cv2.erode(re_image, (5,5), iterations=2)
cv2.drawContours(re_image, contours1, -1, 255, thickness=1)
cv2.drawContours(re_image, contours, -1, 255, thickness=1)

#-----------------------------------------------------------------------------------------------------------
original = img.copy()
cnts, _ = cv2.findContours(re_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
max_area = 0
max_contour = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > max_area:
        max_area = area
        max_contour = c
        print(area)
        print(img.shape[0]*img.shape[1])

if max_contour is not None:
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    ROI = original[y:y + h, x:x + w]
    cv2.imwrite("ROI.png", ROI)

show(img)

cv2.imwrite('result.jpg',img)
