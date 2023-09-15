import numpy as np
import cv2



def histo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the grayscale image
    hist = np.zeros(256)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            hist[gray[i,j]] += 1

    # Compute the cumulative distribution function of the histogram
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]

    # Normalize the cdf by dividing by the total number of pixels and multiplying by 255
    cdf = cdf / (gray.shape[0] * gray.shape[1]) * 255
    cdf = cdf.astype(np.uint8)

    # Apply the mapping to each pixel in the grayscale image
    equ = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            equ[i,j] = cdf[gray[i,j]]

    # Convert the histogram equalized image back to color, if needed
    if img.shape[2] == 3:
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
    return equ 


# Read the input image and convert it to grayscale
img1 =  cv2.imread('images/image-1.png')
out1 = histo(img1) 
imm1 = np.hstack((img1, out1))

img2 =  cv2.imread('images/image-2.png')
out2 = histo(img2) 
imm2 = np.hstack((img2, out2))

# Display the original and equalized images
cv2.imshow("IMAGE 1", imm1)
cv2.imshow("IMAGE 2", imm2)

cv2.waitKey(0)
cv2.destroyAllWindows()
