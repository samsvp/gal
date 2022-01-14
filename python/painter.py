import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from typing import List, Tuple

from genetic_algorithm import GeneticAlgorithm


class Painter:

    def __init__(self, img_path: str, 
            brush_ranges=[[0.1, 0.3], [0.3,0.7]], seed=0) -> None:
        self.original_img = cv2.imread(img_path)
        self.img_grey = cv2.cvtColor(self.original_img,cv2.COLOR_BGR2GRAY)
        self.img_grads = self._img_gradient(self.img_grey)
        self.dna = None
        self.seed = seed
        self.brushes_range = brush_ranges
        self.sampling_mask = None
        
        #start with an empty black img
        self.img_buffer = [np.zeros((self.img_grey.shape[0], self.img_grey.shape[1]), np.uint8)]


    def generate(self, stages=10, generations=100, 
            brush_strokes_count=10, show_progress_imgs=False):
        for s in range(stages):
            #initialize new DNA
            sampling_mask = self.sampling_mask \
                if self.sampling_mask is not None \
                else self.create_sampling_mask(s, stages)
            
            self.dna = GeneticAlgorithm(self.img_grey.shape, 
                             self.img_grads, 
                             self.calc_brush_range(s, stages), 
                             canvas=self.img_buffer[-1], 
                             sampling_mask=sampling_mask)

            self.dna.init_random(self.img_grey, brush_strokes_count,  self.seed + time.time() + s)
            #evolve DNA
            for g in range(generations):
                self.dna.evolve_dna_seq(self.img_grey, self.seed + time.time() + g)
                print("Stage ", s+1, ". Generation ", g+1, "/", generations)
                if show_progress_imgs:
                    clear_output(wait=True)
                    #plt.imshow(sampling_mask, cmap='gray')
                    plt.imshow(self.dna.cached_image, cmap='gray')
                    plt.show()
            self.img_buffer.append(self.dna.cached_image)
        return self.dna.cached_image


    def _img_gradient(self, img: np.ndarray) -> Tuple[float, float]:
        #convert to 0 to 1 float representation
        img = (img / 255.0).astype(np.float32)
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #normalize magnitudes
        mag /= np.max(mag)
        #lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle

    
    def calc_brush_range(self, stage, total_stages):
        return self._calc_brush_size(
            self.brushes_range, stage, total_stages)
        

    def set_brush_range(self, ranges: List[int]) -> None:
        self.brushes_range = ranges
        

    def set_sampling_mask(self, img_path: str) -> None:
        self.sampling_mask = cv2.cvtColor(
            cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


    def create_sampling_mask(self, s: int, stages: int) -> np.ndarray:
        percent = 0.2
        start_stage = int(stages*percent)
        sampling_mask = None
        if s >= start_stage:
            t = (1.0 - (s-start_stage)/max(stages-start_stage-1,1)) * 0.25 + 0.005
            sampling_mask = self.calc_sampling_mask(t)
        return sampling_mask

    
    def _calc_brush_size(self, b_ranges: List[List[float]],
            stage: int, total_stages: int) -> List[float]:
        sizes = []
        for b_range in b_ranges:
            bmin = b_range[0]
            bmax = b_range[1]
            t = stage/max(total_stages-1, 1)
            sizes.append((bmax-bmin)*(-t*t+1)+bmin)
        return sizes

    
    def calc_sampling_mask(self, blur_percent: float) -> float:
        # Calculate gradient 
        gx = cv2.Sobel(self.img_grey, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(self.img_grey, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #calculate blur level
        w = self.img_grey.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
        #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0/mag.max()
        return mag*scale
        