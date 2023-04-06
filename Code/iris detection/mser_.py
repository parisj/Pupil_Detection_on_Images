import numpy as np 
import cv2
import matplotlib.pyplot as plt


def threshold_image(th_img, img):
    for i in range(0,256):
        mask = (img == i)
        th_img[i] = np.where(mask,1,0)
        print(f'i: {i}, th_img[i]: {th_img[i]}')
        cv2.imshow('mask', th_img[i].astype(np.uint8)*255)
        cv2.waitKey(1)
    return th_img


    

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
        
    def process_mser(self, img):
        if len(img.shape) > 2:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: 
            gray_image = img 
        
        self.w, self.h = gray_image.shape
        print(f'self.h {self.h}, self.w {self.w}')
        
        self.components = np.zeros((256, self.w, self.h), dtype = np.int32)
        self.sizes = np.zeros((256, self.w, self.h), dtype = np.int32)
        self.links = np.zeros((256, self.w * self.h), dtype = np.int32)
        th_base = np.zeros((256, self.w, self.h), dtype = np.int32)
        th_img = threshold_image(th_base, img)
        self._compute_component_tree(th_img)
        self._extract_mser()
        
        return self.regions
        

    
    def _compute_component_tree(self, th_img):
        print(f'self.w:{self.w}, self.h: {self.h}')

        self.links[0] = np.arange(self.w * self.h)
        for level in range(1, 256):
            # Initialize the first component with the previous level
            self.components[level] = self.components[level - 1]
            self.sizes[level] = self.sizes[level - 1]
            self.links[level] = np.arange(self.h * self.w)


            # For each pixel in the level
            for i in range(self.w):
                for j in range(self.h):
                    if th_img[level, i, j]:
                        # Get the parent pixel
                        parent = self.links[level - 1][(j * self.w) + i]
                        print(f'parent: {parent}')
                        # If the parent has not been processed yet, continue
                        if self.components[level - 1][parent%self.w][parent//self.w] == -1:
                            continue

                        # Merge the components
                        self.components[level][i, j] = self.num_ccs
                        self.sizes[level][i, j] = self.sizes[level - 1][parent%self.w][parent//self.w] + 1

                        # Remove the parent from the previous level
                        self.components[level - 1][parent%self.w][parent//self.w] = -1

                        # Increase the number of connected components
                        self.num_ccs += 1
        print(f"self.components: {self.components}")

    def _extract_mser(self):
        # For each level in the component tree

        for level in range(1, 256):
            # Get the pixels of the level
            level_pixels = np.where(self.components[level] is not -1)
            print(f'self.components[levels]: {self.components[level]}')
            print(f'level_pixels: {level_pixels},')
            # For each pixel of the level
            # level pixels has the same form as self.components (x,y)
            print(f"level_pixels: {level_pixels}")
            print(f"level_pixels shape: {len(level_pixels)}")
            print(f"level_pixels[0] shape: {len(level_pixels[0])}")

            for x , y in zip(level_pixels[0], level_pixels[1]):

                # Get the connected components of the pixel
                components = self.get_components(x, y, level)

                # For each connected component
                for c in components:
                    # Compute the area variation and diversity
                    area = self.sizes[level][y, x]
                    area_c = self.sizes[level][c[1], c[0]]
                    var = area_c / area
                    div = area_c / self.get_area(c)

                    # If the variation and diversity are within the thresholds
                    if var <= self.max_variation and div >= self.min_diversity:
                        # Add the region
                        self.regions.append(c)

    def get_components(self, x, y, level):
        components = []

        # Get the parent of the pixel
        parent = self.links[level - 1][y * self.w + x]
        print(f'parent: {parent}')

        # If the parent has not been processed yet, return empty components
        if parent < 0 or parent >= self.w * self.h or self.components[level - 1][parent % self.w][parent // self.w] == -1:            
            return components

        # Add the parent to the components
        components.append((x, y))

        # For each neighbor of the parent
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            xx = x + dx
            yy = y + dy

            # If the neighbor is in bounds
            if xx >= 0 and xx < self.w and yy >= 0 and yy < self.h:
                # Get the neighbor's parent
                neighbor_parent = self.links[level - 1][yy * self.w + xx]

                # If the neighbor's parent is the same as the pixel's parent
                if neighbor_parent == parent:
                    # Add the neighbor to the components
                    components.append((xx, yy))
        print(f'components: {components}')
        return components

    def get_area(self, components):
        print(f'components: {components}')
        components = [(c[0], c[1]) for c in components if (c[0], c[1]) != (0, 0)]
        print(f'components: {components}')

        if len(components) == 0:
            return 0

        min_x = min(components, key=lambda c: c[0])[0]
        max_x = max(components, key=lambda c: c[0])[0]
        min_y = min(components, key=lambda c: c[1])[1]
        max_y = max(components, key=lambda c: c[1])[1]

        return (max_x - min_x + 1) * (max_y - min_y + 1)



        

if __name__ == '__main__':
    img = cv2.imread('test_roi.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mser_obj = MSER()
    mser_obj.process_mser(img)
    
    test = np.array([[0,1,2],[1,2,2],[0,0,1]])
    test_th = np.zeros((256,3,3))
    print(f'shape test: {test.shape}, test_th: {test_th.shape}')
