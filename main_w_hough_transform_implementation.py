import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def hough_circle(img_array, radius):
    h, w = img_array.shape
    accumulator = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            if img_array[y, x] > 0: # if pixel is not black
                for theta in range(360):
                    a = x - radius * math.cos(math.radians(theta))
                    b = y - radius * math.sin(math.radians(theta))
                    if a >= 0 and a < w and b >= 0 and b < h:
                        accumulator[int(b), int(a)] += 1
    return accumulator


def hough_circles(img_array, min_radius, max_radius, threshold):
    h, w = img_array.shape
    accumulator = np.zeros((h, w, max_radius-min_radius+1))

    for r in range(min_radius, max_radius+1):
        for y in range(h):
            for x in range(w):
                if img_array[y, x] > 0: # if pixel is not black
                    for theta in range(360):
                        a = x - r * math.cos(math.radians(theta))
                        b = y - r * math.sin(math.radians(theta))
                        if a >= 0 and a < w and b >= 0 and b < h:
                            accumulator[int(b), int(a), r-min_radius] += 1

    circles = []
    for r in range(min_radius, max_radius+1):
        for y in range(h):
            for x in range(w):
                if accumulator[y, x, r-min_radius] > threshold:
                    circles.append((x, y, r))

    return circles


def main():
    img = cv.imread('input/coins.jpg')
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 400, 550)
    plt.subplot(121), plt.imshow(gray, cmap='gray')
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.show()

    multiple = True

    if(multiple):
        img_array = np.array(edges)
        circles = hough_circles(img_array, 55, 80, 120)
        for x, y, r in circles:
            cv.circle(img, (x, y), r, (0, 255, 0), 2)

        cv.imshow('Detected circles',img)
        cv.waitKey(0)
    else:
        radius = 75  # 10ft-75, 5ft-60
        img_array = np.array(edges)
        accumulator = hough_circle(img_array, radius)

        y, x = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        print(x, y)
        # Draw circle on image
        img_with_circle = cv.circle(img, (x, y), radius, (0, 255, 0), 2)

        title = "Detected coin with center at (" + str(x) + ", " + str(y) + ")"
        # display the image with the circle
        cv.imshow(title, img_with_circle)
        cv.waitKey(0)

if __name__ == "__main__":
    main()