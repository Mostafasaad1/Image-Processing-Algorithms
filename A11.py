import cv2
import numpy as np

def gaussian_blur(image, kernel_size, sigma):
    # Create a Gaussian kernel using the given kernel size and standard deviation
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    # Convolve the image with the Gaussian kernel
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image



# Load the image
image = np.asanyarray(cv2.imread('images/image-1.png'))
# Apply a Gaussian blur filter to the image
blurred_image = np.asanyarray(gaussian_blur(image, 5, 1))
blurred_image_cv2 = cv2.GaussianBlur(image, (5, 5), 1)
difference = cv2.absdiff(blurred_image, blurred_image_cv2)


cv2.putText(img=image, text='original', org=(0, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
cv2.putText(img=blurred_image, text='manual blur', org=(0, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
cv2.putText(img=blurred_image_cv2, text='cv2 blur', org=(0, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
cv2.putText(img=difference, text='blurring difference', org=(0, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)

cv2.imshow('Image', np.vstack((np.concatenate((image, blurred_image), axis=1), np.concatenate((blurred_image_cv2, difference), axis=1))))

cv2.waitKey(0)
cv2.destroyAllWindows()

