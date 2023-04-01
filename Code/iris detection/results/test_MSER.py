import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

class MSER:
    def __init__(self, delta=2, min_area=0, max_area=1000000, max_variation=1, min_diversity=0):
        self.delta = delta
        self.min_area = min_area
        self.max_area = max_area
        self.max_variation = max_variation
        self.min_diversity = min_diversity
        self.components = None
        self.sizes = None
        self.links = None
        self.num_ccs = 0
        self.regions = []
        self.h = None
        self.w = None

    def process_image(self, image):
        # Convert to grayscale if the image has multiple channels
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Initialize some variables
        self.h, self.w = gray_image.shape
        h = self.h
        w = self.w
        print(f"h {self.h}, and w:{self.w}")

        self.components = np.zeros((256, h * w), dtype=np.int32)
        self.sizes = np.zeros((256, h * w), dtype=np.int32)
        self.links = np.zeros((256, h * w), dtype=np.int32)

        # Step 1: Linearize the image data
        linear_image = np.zeros((256, h * w), dtype=np.uint8)
        for i in range(0, 256):
            mask = (gray_image == i)
            linear_image[i] = np.where(mask, 1, 0).flatten()

        # Step 2: Compute the component tree
        self._compute_component_tree(linear_image)

        # Step 3: Extract MSERs
        self._extract_mser()

        return self.regions

    def _compute_component_tree(self, linear_image):
        h, w = linear_image.shape[::-1]
        print(f"h {h}, and w:{w}, len {len(self.regions)}")

        self.components = np.zeros((256, h * w), dtype=np.int32)
        self.sizes = np.zeros((256, h * w), dtype=np.int32)
        self.links = np.zeros((256, h * w), dtype=np.int32)

        for level in range(1, 256):
            # Initialize the first component with the previous level
            self.components[level] = self.components[level - 1]
            self.sizes[level] = self.sizes[level - 1]
            self.links[level] = np.arange(h * w)

            # For each pixel in the level
            for idx in np.argwhere(linear_image[level - 1]).flatten():
                # Get the parent pixel
                parent = self.links[level - 1][idx]

                # If the parent has not been processed yet, continue
                if self.components[level - 1][parent] == -1:
                    continue

                # Merge the components
                self.components[level][idx] = self.num_ccs
                self.sizes[level][idx] = self.sizes[level - 1][parent] + 1

                # Remove the parent from the previous level
                self.components[level - 1][parent] = -1

                # Increase the number of connected components
                self.num_ccs += 1
                print(f'self num_ccs = {self.num_ccs}, idx {idx}, level {level}')
                print(f'parent = {parent}, self.components {self.components}, self.links {self.links}')

                if level > 190:
                    self.visualize_component_tree(level-160)

    def _extract_mser(self):
        for q in range(self.num_ccs):
            for level in range(256 - self.delta, -1, -1):
                if self.sizes[level][q] > 0:
                    if self.sizes[level - self.delta][q] > 0:
                        size_ratio = float(self.sizes[level][q]) / float(self.sizes[level - self.delta][q])

                        if size_ratio < self.max_variation and self.sizes[level][q] > self.min_area and self.sizes[level][q] < self.max_area:
                            if len(self.regions) == 0 or self.regions[-1][0] != q or self.regions[-1][1] / size_ratio > 1.0 - self.min_diversity:
                                x, y = np.unravel_index(np.argwhere(self.components[level] == q).flatten(), (self.h, self.w))
                                self.regions.append((q, size_ratio, np.column_stack((x, y))))
                            else:
                                x, y = np.unravel_index(np.argwhere(self.components[level] == q).flatten(), (self.h, self.w))
                                self.regions[-1] = (q, size_ratio, np.column_stack((x, y)))

        # Print some statistics about the detected MSERs
        print(f"Number of MSERs detected: {len(self.regions)}")
        if len(self.regions) > 0:
            print("Example MSERs:")
            for i in range(min(5, len(self.regions))):
                region = self.regions[i]
                print(f"\tMSER {i+1}: size_ratio={region[1]:.2f}, area={self.sizes[region[0]][0]}, num_pixels={len(region[2])}")


    def visualize_component_tree(self, level):
        component_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.h):
            for j in range(self.w):
                if self.components[level][i * self.w + j] == -1:
                    component_image[i, j] = [0, 0, 0]
                else:
                    color = self.components[level][i * self.w + j] * (255 // self.num_ccs)
                    component_image[i, j] = [color, color, color]
        plt.imshow(component_image)
        plt.show()
    
    
if __name__ == '__main__':
    image = cv2.imread('test_roi.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize MSER detector
    mser = MSER()

    # Detect regions
    regions = mser.process_image(image)

    # Draw regions on image
    image_copy = image.copy()
    for region in regions:
        x, y, w, h = cv2.boundingRect(region[2])
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show image with detected regions
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.show()                                