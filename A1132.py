import cv2
import numpy as np

def morph(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and smooth edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding method to obtain a binary image
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a structuring element for morphology operations
    kernel = np.ones((3, 3), np.uint8)

    # Apply a morphological opening operation to remove small white regions
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Apply a morphological closing operation to fill small black gaps
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing



# Read the input image and convert it to grayscale
img =  cv2.imread('images/image-3.jpeg')
out = morph(img) 
out1 = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
imm = np.hstack((img, out1))
# Display the original and equalized images
cv2.imshow("OUTPUT", imm)
cv2.waitKey(0)
cv2.destroyAllWindows()
