# %%
import argparse
import numpy as np
import skimage

from typing import List, Tuple


def get_angles(genes: np.ndarray) -> np.ndarray:
    angles = genes[3,:].reshape(-1)
    return 2 * np.pi * angles - np.pi


def get_scale(genes: np.ndarray, 
        og_size: Tuple[int, int],
        target_size: Tuple[int, int]) -> np.ndarray:
    r = target_size[0] / og_size[0]
    scales = genes[2,:].reshape(-1)
    return (0.7 * scales + 0.3) * r


def get_xy(genes: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    x = genes[1,:].reshape(-1)
    y = genes[0,:].reshape(-1)
    return (x, y)


def load_data(filepath: str) -> \
        Tuple[List[int], List[str], np.ndarray]:
    """
    Loads the gene data from file
    """
    with open(filepath) as f:
        data = f.read().split("end")

    dna_dims = [int(d) for d in 
        data[0].split(":")[-1].split(" ")]
    og_dims = [int(d) for d in 
        data[1].split(":")[-1].split(" ")]
    scale = float(data[2].split(":")[-1])
    img_files = [d.replace("\t","") for d in 
        data[3].split(":")[-1].split("\n") if d]
    flat_genes = np.array([float(d) for d in 
        data[4].split(":")[-1].split("\n") if d])
    genes = flat_genes.reshape(dna_dims[:2][::-1])
    return (og_dims, scale, img_files, genes)


def resize(img: np.ndarray, s: float) -> np.ndarray:
    """
    Resizes image by the given scale percentage
    """
    size = (int(s * img.shape[0]), int(s * img.shape[1]))
    r_img = (skimage.transform.resize(
        img / 255, size, anti_aliasing=True) * 255).astype(int)
    return r_img


def rotate(img: np.ndarray, rad_angle: float) -> np.ndarray:
    """
    Rotates the image by the given angle
    """
    angle = np.rad2deg(rad_angle)
    r = skimage.transform.rotate(img / 255, angle, 1)
    return (r * 255).astype(int)


def create_img(genes: np.ndarray, img_files: List[str],
        target_size: Tuple[int, int], og_scale: float) -> np.ndarray:
    """
    Creates the image using the given metadata
    """
    angles = get_angles(genes)
    scales = get_scale(genes, og_dims, target_size)
    x, y = get_xy(genes)

    res_img = np.zeros(
        (*target_size[:2], 4), dtype=int)

    objs = [skimage.io.imread(img_file) 
        for img_file in img_files]

    for i in range(x.shape[0]):
        def transform(_obj: np.ndarray, scale: float,
                angle: float) -> np.ndarray:
            """
            Rotates and resizes the image    
            """
            obj = resize(rotate(_obj, angle),
                scale * og_scale)
            return obj

        obj = transform(objs[i], scales[i], angles[i])

        s_x = int(x[i] * res_img.shape[1])
        s_y = int(y[i] * res_img.shape[0])
        e_x = s_x + obj.shape[1]
        e_y = s_y + obj.shape[0]

        img = res_img[s_y:e_y,s_x:e_x,:]
        
        # adds the image together
        mask = obj[:,:,-1].astype(bool)
        try:
            img[mask] = obj[mask]
            res_img[s_y:e_y,s_x:e_x,:] = img
        except:
            print(f"Image {i} skipped due to image bounds")
    
    return res_img


# %%
if __name__ == "__main__":
    # example usage: 
    # python3 make_img.py --path ../build/out_genes.txt -r 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", 
        type=str, help="File to read metadata from")
    parser.add_argument("-s", "--save_path", default="out.png",
        type=str, help="Path to save image")
    parser.add_argument("-r", "--resize", default=1,
        type=float, help="Resize the original target image")

    args = parser.parse_args()
    
    print(f"{args}")

    og_dims, scale, img_files, genes = load_data(args.path)
    target_size = (args.resize * np.array(og_dims)).astype(int)
    img = create_img(genes, img_files, target_size, scale)
    skimage.io.imsave(args.save_path, img / 255)
