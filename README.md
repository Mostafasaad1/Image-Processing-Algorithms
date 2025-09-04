# AI Bootcamp Final Project: Image Processing Algorithms

This repository contains a collection of Python scripts demonstrating fundamental image processing algorithms implemented from scratch, as well as comparisons with their OpenCV counterparts. This project was developed as part of an AI bootcamp final assignment.

## Project Structure

The project is organized into several Python scripts, each focusing on a specific image processing technique:

- `A11.py`: Custom Gaussian Blur Implementation
- `A12.py`: Custom Canny Edge Detection Implementation
- `A1131.py`: Custom Histogram Equalization Implementation
- `A1132.py`: Custom Morphological Operations (Opening and Closing) Implementation

## Setup and Installation

To run these scripts, you will need Python 3.x and the OpenCV library. It is recommended to use a virtual environment.

1. Clone the repository:

   ```bash
   git clone https://github.com/Mostafasaad1/Image-Processing-Algorithms.git
   cd Image-Processing-Algorithms
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required libraries:

   ```bash
   pip install opencv-python numpy
   ```

## Usage

Each script can be run independently. Ensure you have the necessary image files in a directory named `images` in the same location as the scripts. Example images used in the scripts are `image-1.png`, `image-2.png`, and `image-3.jpeg`.

### `A11.py`: Gaussian Blur

This script demonstrates a custom implementation of Gaussian blur and compares it with OpenCV's `GaussianBlur` function. It visualizes the original image, the custom blurred image, the OpenCV blurred image, and the difference between the two blurred outputs.

To run:

```bash
python A11.py
```

### `A12.py`: Canny Edge Detection

This script provides a step-by-step implementation of the Canny edge detection algorithm, including Gaussian smoothing, gradient calculation, non-maximum suppression, and hysteresis thresholding. It compares the custom Canny output with OpenCV's `Canny` function.

To run:

```bash
python A12.py
```

### `A1131.py`: Histogram Equalization

This script implements histogram equalization from scratch. It applies the equalization to two different images and displays the original and processed versions side-by-side.

To run:

```bash
python A1131.py
```

### `A1132.py`: Morphological Operations

This script demonstrates custom implementations of morphological opening and closing operations. It applies these operations to an image after thresholding and displays the original and processed images.

To run:

```bash
python A1132.py
```

## Contributing

Feel free to fork this repository, submit pull requests, or open issues for any improvements or bug fixes.

## License

This project is open-source and available under the MIT License
