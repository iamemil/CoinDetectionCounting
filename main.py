import cv2
import numpy as np

# Load image
img = cv2.imread('input/coins.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Median blur to reduce noise
median_blur = cv2.medianBlur(gray, 5)

# Detect circles using HoughCircles function
circles = cv2.HoughCircles(median_blur, cv2.HOUGH_GRADIENT, 0.9, 50, param1=100, param2=72, minRadius=160, maxRadius=260)

# Convert circles to integers
circles = np.uint16(np.around(circles))

# Draw circles on original image
total = 0
for i in circles[0, :]:
    # Draw circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #print(i[2])
    # Calculate coin value based on diameter
    if i[2] > 220:
        total += 50
    elif i[2] >= 214:
        total += 200
    elif i[2] >= 202:
        total += 20
    elif i[2] >= 188:
        total += 10
    elif i[2] >= 183:
        total += 100
    elif i[2] >= 160:
        total += 5

"""
5ft - 21.2 mm 172
10ft - 24.8 mm 188
20ft - 26.3 mm 211
50ft - 27.4 mm 215
100ft - 23.8 mm 186
200ft - 28.3 mm 220
"""

# Display total amount and image
print('Total amount:', total, 'HUF')
cv2.imshow('Coins', img)
cv2.waitKey(0)
cv2.destroyAllWindows()