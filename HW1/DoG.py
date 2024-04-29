import numpy as np
import cv2

class Difference_of_Gaussian(object):
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        first = [image] + [cv2.GaussianBlur(image, (0, 0), self.sigma**i) for i in range(1, self.num_guassian_images_per_octave)]
        
        adjust = cv2.resize(first[-1], (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_NEAREST)
        
        second = [adjust] + [cv2.GaussianBlur(adjust, (0, 0), self.sigma**i) for i in range(1, self.num_guassian_images_per_octave)]
        
        gaussian_images = [first, second]
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        
        dog_images = []
        for i in range(self.num_octaves):
            gaussian_images_i = gaussian_images[i]
            dog_images_i = []
            
            for j in range(self.num_DoG_images_per_octave):
                dogs = cv2.subtract(gaussian_images_i[j], gaussian_images_i[j+1])
                dog_images_i.append(dogs)
                
                Max = max(dogs.flatten())
                Min = min(dogs.flatten())
                
                norm = (dogs - Min)*255 / (Max - Min)
                cv2.imwrite(f"testdata/DoG{i+1}-{j+1}.png", norm)
                
            dog_images.append(dog_images_i)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = np.array([], dtype='int64').reshape((0, 2))
        for i in range(self.num_octaves):
            dogs = np.array(dog_images[i])
            

            temp = np.array([np.roll(dogs,(x,y,z),axis=(2,1,0)) for z in range(-1,2) for y in range(-1,2) for x in range(-1,2)])
            
            change = np.logical_and(np.absolute(dogs) >= self.threshold, 
                      np.logical_or(np.min(temp, axis=0) == dogs, 
                                    np.max(temp, axis=0) == dogs))
            
            for j in range(1, self.num_DoG_images_per_octave-1):
                m = change[j]
                x_temp, y_temp = np.meshgrid(np.arange(m.shape[1]), np.arange(m.shape[0]))
                stacked_points = np.stack([y_temp[m], x_temp[m]]).T
                
                if i:
                    keypoints_temp = stacked_points * 2
                else:
                    keypoints_temp = stacked_points
                    
                keypoints = np.concatenate([keypoints,keypoints_temp])
            

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        
        keypoints = np.unique(np.array(keypoints), axis = 0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        
        return keypoints

