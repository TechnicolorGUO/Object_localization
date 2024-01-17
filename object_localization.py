import cv2
import numpy as np

def show(image):
    cv2.imshow("temp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = ".\sample1.jpg"
img = cv2.imread(path)

#BGT 2 Gray
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Blur
blur_image = cv2.bilateralFilter(gray_image,9,75,75)

#Gray 2 canny with edge
canny_image = cv2.Canny(blur_image, 0, 200)

#Dilate and erode
kernel = np.ones((5, 5), np.uint8)
dilated_img = cv2.dilate(canny_image, kernel, iterations=1)
erode_img = cv2.erode(dilated_img, kernel, iterations=1)

#Fill the edge 
filled_image = np.zeros_like(gray_image)
contours, _ = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)

#Find the contours of the filled-edge image
original = img.copy()
cnts = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#Find the box with the largest area
max_area = 0
max_contour = None
for c in cnts:
    area = cv2.contourArea(c)
    if area > max_area:
        max_area = area
        max_contour = c

if max_contour is not None:
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    ROI = original[y:y + h, x:x + w]
    cv2.imwrite("rst.png", img)

show(img)