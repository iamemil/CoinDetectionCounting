"""
5ft - 21.2 mm 172
10ft - 24.8 mm 188
20ft - 26.3 mm 211
50ft - 27.4 mm 215
100ft - 23.8 mm 186
200ft - 28.3 mm 220

200
50
20
10
100
5
"""
"""
coin detection using opencv and python
task: draw a circle around each coin in an image
method:
- find the edge using Gaussian and Canny
- try to fit a circle to the edges by comparing circles of increasing size. Once passed threshold,
  assume that is the radius of coin and save the coordinates of the center
- draw the circles to the original image
"""

import cv2
import numpy as np
import math

coins = cv2.imread('input/coins.jpg', 1)

# defining minimal and maximal radius, specified to the coins.jpg
min_r = 300
max_r = 480

def edge_detect_coins():
    """
    import the coins.jpg image and detect the edges of the coins.
    """

    coins_height, coins_width, coins_channel = coins.shape

    # optimisation by decreasing the size of image, resulting in 4x faster run time
    coins_resized = cv2.resize(coins, (int(coins_width), int(coins_height)))

    # blur to optimise edge finding
    coins_blurred = cv2.GaussianBlur(coins_resized, (5, 5), cv2.BORDER_DEFAULT)

    # used Canny to find the edge
    coins_edge = cv2.Canny(coins_blurred, 127, 255)

    cv2.imwrite("output/coins_blurred.jpg", coins_blurred)
    cv2.imwrite("output/coins_edge.jpg", coins_edge)

    return coins_edge


def coin_center_detect():
    """
    aim is to find the edges, find the radius of the coin and save the coordinates of the centers.
    """

    # image with edges of coins detected
    coins_edge = edge_detect_coins()

    # obtain the image size
    max_height, max_width = coins_edge.shape

    edge_threshold = 0.9  # how many pixels need to pass to be considered a coin edge
    intensity_threshold = 255 * 0.123  # the min value of pixel intensity to be considered edge
    next_circle_step = 100  # the amount of pixels to move to start comparing again
    coin_detection = []

    # draw circles
    for radius in range(min_r, max_r):
        img_circle = np.zeros((radius * 2, radius * 2, 1), np.uint8)
        circle = cv2.circle(img_circle, (radius, radius), radius, 255)

        circumference = 2 * math.pi * radius

        circle_pixels = []

        for y in range(len(circle)):
            for x in range(len(circle[y])):
                if circle[x][y] == 255:
                    circle_pixels.append((x, y))

        print(('radius', radius))

        # move circle through image
        for start_y in range(0, max_height - 2 * radius, next_circle_step):
            for start_x in range(0, max_width - 2 * radius, next_circle_step):
                count = 0

                # cycle through the coordinates of circle
                for (x, y) in circle_pixels:
                    image_y = start_y + y
                    image_x = start_x + x

                    if coins_edge[image_y][image_x] >= intensity_threshold:
                        count += 1

                if count > 50:
                    percentage = round(count / circumference * 100, 2)
                    coor_x = start_x + radius
                    coor_y = start_y + radius
                    print(('candidate', coor_x, coor_y, radius, percentage))

                if (count / circumference) > edge_threshold:
                    coor_x = start_x + radius
                    coor_y = start_y + radius
                    coin_detection.append((coor_x, coor_y, radius))  # center
                    print(('-----------------', start_x + radius, start_y + radius, radius))

    return coin_detection


def circle_coins():

    coins_circled = coin_center_detect()
    coins_copy = coins.copy()
    for detected_circle in coins_circled:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(coins_copy, (x_coor*2, y_coor*2), detected_radius*2, (0, 0, 255), 1)

    cv2.imwrite("output/coin_detection/coins_detected.jpg", coins_detected)

    # cv2.imwrite("output_image/coins_resized.jpg", coins_resized)


def hough_circle_detection():
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        img,  # source image
        cv2.HOUGH_GRADIENT,  # type of detection
        1,
        40,
        param1=50,
        param2=30,
        minRadius=min_r*2,  # minimal radius
        maxRadius=max_r*2,  # max radius
    )

    coins_copy = coins.copy()

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(coins_copy, (int(x_coor), int(y_coor)), int(detected_radius), (0, 0, 255), 1)

    cv2.imwrite("output/coins_detected_Hough.jpg", coins_detected)


def compare_circle_detection():
    circle_coins()
    hough_circle_detection()

if __name__ == "__main__":
    compare_circle_detection()