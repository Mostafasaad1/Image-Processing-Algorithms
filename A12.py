import cv2
import numpy as np


def gaussian_blur(image, kernel_size, sigma):
    # Create a Gaussian kernel using the given kernel size and standard deviation
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    # Convolve the image with the Gaussian kernel
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


# Define a function that takes an image and two threshold values as input
def canny_edge_detection(inmage, low_threshold, high_threshold):

    image = cv2.cvtColor(inmage, cv2.COLOR_BGR2GRAY)
    
    blurred_image = gaussian_blur(image,5, 0)

    
    # Use cv2.CV_32F to avoid overflow
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate the magnitude and angle of the gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    
    # Initialize an output image with zeros
    output_image = np.zeros_like(magnitude)

    # Get the dimensions of the image
    height, width = magnitude.shape

    # Loop over each pixel in the image
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Get the gradient angle of the current pixel
            theta = angle[i, j]

            # Round the angle to one of four directions: 0, 45, 90, or 135 degrees
            if (theta < 22.5 or theta > 157.5):
                theta = 0
            elif (22.5 <= theta < 67.5):
                theta = 45
            elif (67.5 <= theta < 112.5):
                theta = 90
            else:
                theta = 135

            # Compare the magnitude of the current pixel with its neighbors along the gradient direction
            # If it is a local maximum, keep it. Otherwise, suppress it.
            if (theta == 0):
                if (magnitude[i, j] >= magnitude[i, j-1] and magnitude[i, j] >= magnitude[i, j+1]):
                    output_image[i, j] = magnitude[i, j]
            elif (theta == 45):
                if (magnitude[i, j] >= magnitude[i-1, j+1] and magnitude[i, j] >= magnitude[i+1, j-1]):
                    output_image[i, j] = magnitude[i, j]
            elif (theta == 90):
                if (magnitude[i, j] >= magnitude[i-1, j] and magnitude[i, j] >= magnitude[i+1, j]):
                    output_image[i, j] = magnitude[i, j]
            else:
                if (magnitude[i, j] >= magnitude[i-1, j-1] and magnitude[i, j] >= magnitude[i+1, j+1]):
                    output_image[i, j] = magnitude[i, j]

    
    # Define two threshold values: high and low
    high_threshold = high_threshold * output_image.max()
    low_threshold = low_threshold * high_threshold

    # Initialize two arrays to store strong and weak pixels
    strong_pixels = np.array(np.where(output_image >= high_threshold)).T
    weak_pixels = np.array(np.where((output_image <= high_threshold) & (output_image >= low_threshold))).T

    # Set the values of strong and weak pixels in the output image
    output_image[strong_pixels[:,0], strong_pixels[:,1]] = 255
    output_image[weak_pixels[:,0], weak_pixels[:,1]] = 50
    
    # Define a function that checks if a weak pixel is connected to a strong pixel
    def is_connected(i ,j):
        # Get the neighboring pixels of the current pixel
        neighbors = output_image[i-1:i+2,j-1:j+2]

        # Check if any of them is a strong pixel
        if np.any(neighbors == 255):
            return True
        else:
            return False

    # Loop over each weak pixel in the output image
    for i,j in weak_pixels:
        # If it is connected to a strong pixel, mark it as an edge. Otherwise, discard it.
        if is_connected(i, j):
            output_image[i, j] = 255
        else:
            output_image[i, j] = 0

    # Return the final output image
    return output_image


## Testing 
image = cv2.imread('images/image-1.png')
edges_custom = cv2.cvtColor(canny_edge_detection(image, 0.01,0.2),cv2.COLOR_GRAY2BGR)
edges_cv2 = cv2.cvtColor(cv2.Canny(image, 100, 200),cv2.COLOR_GRAY2BGR)
cv2.putText(img=edges_custom, text='manual canny', org=(0, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
cv2.putText(img=edges_cv2, text='cv2 canny', org=(0, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
imm = np.hstack((edges_custom, edges_cv2))
cv2.imshow("Final",imm)
cv2.waitKey(0)
cv2.destroyAllWindows()




