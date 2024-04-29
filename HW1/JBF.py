import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        s = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)
        
        r = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)
        
        weight, result = np.zeros(padded_img.shape), np.zeros(padded_img.shape)
        
        for x in range(-self.pad_w, self.pad_w+1):
            for y in range(-self.pad_w, self.pad_w+1):

                temp = r[np.abs(np.roll(padded_guidance, [y,x], axis=[0,1])-padded_guidance)]
                
                if temp.ndim==2:
                    temp_r = temp
                else:
                    temp_r = np.prod(temp,axis=2)
                
                temp_s = s[np.abs(x)] * s[np.abs(y)]
                temp_t = temp_s * temp_r
                padded_img_roll = np.roll(padded_img, [y,x], axis=[0,1])
                
                for channel in range(padded_img.ndim):
                    result[:,:,channel] += padded_img_roll[:,:,channel] * temp_t
                    weight[:,:,channel] += temp_t
                    
        output = (result/weight)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w,:]
        
        return np.clip(output, 0, 255).astype(np.uint8)