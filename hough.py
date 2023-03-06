from PIL import Image, ImageDraw
import numpy as np
import math

# Load the JPEG image and convert it to grayscale
img = Image.open('input/5ft_small.jpeg').convert('L')

# Convert the grayscale image to a numpy array
img_array = np.array(img)

# Define the Hough Circle Transform function
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

# Apply the Hough Circle Transform with radius 20
accumulator = hough_circle(img_array, 100)

# Find the circle with the maximum votes in the accumulator
y, x = np.unravel_index(np.argmax(accumulator), accumulator.shape)

# Draw the circle on the original image
draw = ImageDraw.Draw(img)
draw.ellipse((x-20, y-20, x+20, y+20), outline='red')

# Save the resulting image
img.save('result.jpg')
