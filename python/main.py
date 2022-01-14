# %%
import cv2
import time
from painter import Painter

gen = Painter('../imgs/example.jpg', seed=time.time())
out = gen.generate(100, 20, 10, 1)


# %%
#load a custom mask and set a smaller brush size for finer details
gen.sampling_mask = cv2.cvtColor(cv2.imread("../imgs/mask.jpg"), cv2.COLOR_BGR2GRAY)
gen.brushes_range = [[0.05, 0.1], [0.1, 0.2]]

#keep drawing on top of our previous result
out = gen.generate(40, 30, 10, 1)

for i in range(len(gen.img_buffer)):
    cv2.imwrite("out", f"{i:06d}.png", gen.img_buffer[i])
# %%
