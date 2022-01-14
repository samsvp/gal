import cv2
import random
import numpy as np

from typing import Tuple


class GeneticAlgorithm:

    def __init__(self, bound: Tuple[int, int], img_gradient: np.ndarray, 
            brushstrokes_range: Tuple[int, int], canvas: np.ndarray, 
            sampling_mask=None) -> None:
        
        self.dna_seq = []
        self.bound = bound

        self.min_size = brushstrokes_range[0]
        self.max_size = brushstrokes_range[1]

        self.brush_size = 300
        self.padding = int(self.brush_size * self.max_size) // 2 + 5

        self.canvas = canvas

        # gradient
        self.img_mag = img_gradient[0]
        self.img_angles = img_gradient[1]

        self.sampling_mask = sampling_mask

        # cache
        self.cached_img = None
        self.cached_error = None

        self.brush = cv2.imread("../brushes/1.jpg")
        

    def init_random(self, target_image: np.ndarray, count: int, seed=0) -> None:
        #initialize random DNA sequence
        for i in range(count):
            color = random.randrange(0, 255)
            size = random.random()*(self.max_size-self.min_size) + self.min_size
            posY, posX = self.gen_new_positions()
            '''
            start with the angle from image gradient
            based on magnitude of that angle direction, adjust the random angle offset.
            So in places of high magnitude, we are more likely to follow the angle with our brushstroke.
            In places of low magnitude, we can have a more random brushstroke direction.
            '''
            local_mag = self.img_mag[posY][posX]
            local_angle = self.img_angles[posY][posX] + 90 #perpendicular to the dir
            rotation = random.randrange(-180, 180)*(1-local_mag) + local_angle
            self.dna_seq.append([color, posY, posX, size, rotation])
        #calculate cache error and image
        self.cached_error, self.cached_image = self.calc_total_error(target_image)
    

    def gen_new_positions(self) -> Tuple[int, int]:
        if self.sampling_mask is not None:
            pos = self.sample_from_img(self.sampling_mask)
            print(pos)
            posX = pos[1][0]
            posY = pos[0][0]
        else:
            posY = int(random.randrange(0, self.bound[0]))
            posX = int(random.randrange(0, self.bound[1]))
        return (posY, posX)


    def sample_from_img(self, img: np.ndarray) -> np.ndarray:
        #possible positions to sample
        pos = np.indices(dimensions=img.shape)
        pos = pos.reshape(2, pos.shape[1]*pos.shape[2])
        img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
        return pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]

            
    def calc_total_error(self, in_img: np.ndarray):
        return self._calc_error(self.dna_seq, in_img)


    def _calc_error(self, dna_seq, in_img: np.ndarray) :
        my_img = self.draw_all(dna_seq)
        #compare the DNA to img and calc fitness only in the ROI
        diff1 = cv2.subtract(in_img, my_img) #values are too low
        diff2 = cv2.subtract(my_img, in_img) #values are too high
        total_iff = cv2.add(diff1, diff2)
        total_iff = np.sum(total_iff)
        return (total_iff, my_img)
    
    
    def draw_all(self, dna_seq):
        in_img = np.zeros(self.bound[:2], np.uint8) \
            if self.canvas is None else self.canvas

        p = self.padding
        in_img = cv2.copyMakeBorder(in_img, p, p, p, p, 
            cv2.BORDER_CONSTANT, value=[0,0,0])

        for dna in dna_seq:
            self.draw_dna(dna, in_img)

        #remove padding
        y = in_img.shape[0]
        x = in_img.shape[1]
        return in_img[p:(y-p), p:(x-p)] 


    def draw_dna(self, dna, in_img: np.ndarray) -> np.ndarray:
        #get DNA data
        color = dna[0]
        posX = int(dna[2]) + self.padding #add padding since indices have shifted
        posY = int(dna[1]) + self.padding
        size = dna[3]
        rotation = dna[4]
        
        brush_img = cv2.resize(self.brush, None,
            fx=size, fy=size, interpolation=cv2.INTER_CUBIC)

        brush_img = self._rotate_img(brush_img, rotation)

        brush_img = cv2.cvtColor(brush_img,cv2.COLOR_BGR2GRAY)
        rows, cols = brush_img.shape

        canvas = np.copy(brush_img)
        canvas[:, :] = color

        #find ROI
        y_min = int(posY - rows/2)
        y_max = int(posY + (rows - rows/2))
        x_min = int(posX - cols/2)
        x_max = int(posX + (cols - cols/2))
        
        foreground = canvas[0:rows, 0:cols].astype(float)
        background = in_img[y_min:y_max,x_min:x_max].astype(float) #get ROI
        
        alpha = brush_img.astype(float)/255.0
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(np.clip((1.0 - alpha), 0.0, 1.0), background)
        out_image = (np.clip(cv2.add(foreground, background), 0.0, 255.0)).astype(np.uint8)
        in_img[y_min:y_max, x_min:x_max] = out_image
        return in_img
        

    def _rotate_img(self, img: np.ndarray, angle: float):
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols,rows))
        return dst

    
    def _evolve_dna(self, index, in_img, seed):
        #create a copy of the list and get its child  
        dna_seq_copy = np.copy(self.dna_seq)           
        child = dna_seq_copy[index]
        
        #mutate the child
        #select which items to mutate
        random.seed(seed + index)
        index_options = [0,1,2,3,4]
        change_indices = []
        change_count = random.randrange(1, len(index_options)+1)
        for _ in range(change_count):
            indexToTake = random.randrange(0, len(index_options))
            #move it the change list
            change_indices.append(index_options.pop(indexToTake))

        np.sort(change_indices)
        change_indices = change_indices[::-1]

        for change_index in change_indices:
            if change_index == 0:# if color
                child[0] = int(random.randrange(0, 255))
            elif change_index == 1 or change_index == 2:#if pos Y or X
                child[1], child[2] = self.gen_new_positions()
            elif change_index == 3: #if size
                child[3] = random.random()*(self.max_size-self.min_size) + self.min_size
            elif change_index == 4: #if rotation
                local_mag = self.img_mag[int(child[1])][int(child[2])]
                local_angle = self.img_angles[int(child[1])][int(child[2])] + 90 #perpendicular
                child[4] = random.randrange(-180, 180)*(1-local_mag) + local_angle

        #if child performs better replace parent
        child_error, child_img = self._calc_error(dna_seq_copy, in_img)
        if  child_error < self.cached_error:
            self.dna_seq[index] = child[:]
            self.cached_image = child_img
            self.cached_error = child_error
    

    def evolve_dna_seq(self, in_img: np.ndarray, seed: int) -> None:
        for i in range(len(self.dna_seq)):
            self._evolve_dna(i, in_img, seed)
        
        