# %%
import cv2
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

from typing import Tuple


def load_target(filename: str) -> np.ndarray:
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


def load_brush(filename: str) -> np.ndarray:
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    rows, columns = img.shape[:2]
    scale = 0.2
    img = cv2.resize(img, (int(scale * columns),
        int(scale * rows)))
    img[:,:,:3] -= 100
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def pick_points(img: np.ndarray, max_points_x: int,
        max_points_y: int) -> \
            Tuple[np.ndarray, np.ndarray]:
    """
    Returns which points(x,y) in the image to paint
    """
    rows, columns = img.shape[:2]

    # don't ask
    step_y = rows // max_points_x
    step_x = columns // max_points_y

    points_y = np.arange(rows, step=step_y)
    points_x = np.arange(columns, step=step_x)

    return tuple(np.meshgrid(points_y, points_x))


def brushstroke(img: np.ndarray, brush: np.ndarray,
        color: np.ndarray, point: Tuple[int, int]) -> np.ndarray:
    """
    """
    brush_x, brush_y = brush.shape[:2]
    x, y = point
    c_brush = np.copy(brush)
    c_brush[...,:3] = (c_brush[...,:3] / 255 * color[...,:3]).astype(np.uint8)
    try:
        patch = img[x:x+brush_x, y:y+brush_y, :]
        mask = patch == 0
        patch[mask] = c_brush[mask]
        img[x:x+brush_x, y:y+brush_y, :] = patch
    except:
#        print(x+brush_x, y+brush_y, img.shape)
        pass
    return img


def create_img(imgs_path: str, brush_path: str,
        max_points_x: int=100, max_points_y: int=100) -> \
            np.ndarray:
    """
    """
    brush = load_brush(brush_path)
    target = load_target(imgs_path)

    rows, cols, depth = target.shape
    img = np.zeros((rows, cols, depth), 
        dtype=np.uint8)

    points_y, points_x = pick_points(target, 
        max_points_x, max_points_y)

    colors = target[(points_y, points_x)]

    for xs,ys,cs in zip(points_x, points_y, colors):
        for x, y, c in zip(xs, ys, cs):
            img = brushstroke(img, brush, c, (y, x))

    return img

if __name__ == "__main__":
    imgs_path = "imgs/example.jpg"
    brush_path = "brushes/3.png"

    target = load_target(imgs_path)
    img = create_img(imgs_path, brush_path,
        max_points_x=100, max_points_y=150)
    brush = load_brush(brush_path)

    cv2.imshow("window_name", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    plt.imshow(img)
    plt.show()
    plt.imshow(target)
    plt.show()


# %%
