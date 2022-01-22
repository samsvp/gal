import sys
import cv2
import numpy as np


def color_transfer(content: np.ndarray, style: np.ndarray) -> np.ndarray:
    """
    Color transfer with histogram matching using interpolation
    """
    transfered = np.copy(content)
    # for each channel of the content, match the cum_histogram with the style's one
    for i in range(0, content.shape[2]):
        content_channel = content[:, :, i].flatten()
        style_channel = style[:, :, i].flatten()
        # calculate histogram for both content and style
        content_values, content_indices, content_counts = np.unique(content_channel, return_inverse=True, return_counts=True)
        style_values, style_counts = np.unique(style_channel, return_counts=True)
        # calculate cummulative histogram
        content_cumhist = np.cumsum(content_counts)
        style_cumhist = np.cumsum(style_counts)
        # normalize it
        content_cumhist = content_cumhist / np.max(content_cumhist)
        style_cumhist = style_cumhist / np.max(style_cumhist)
        # match using interpolation
        matched = np.interp(content_cumhist, style_cumhist, style_values)
        transfered[:, :, i] = matched[content_indices].reshape(content[:, :, i].shape)
    return transfered


def color_transfer_lab(content: np.ndarray, style: np.ndarray) -> np.ndarray:
    """
    Color transfer through converting an image to the LAB color space, changing
    the mean and variance there, then converting the image back into RGB
    """
    # convert images to LAB space
    style_lab = cv2.cvtColor(style, cv2.COLOR_RGB2LAB)  # color.rgb2lab(style)
    content_lab = cv2.cvtColor(content, cv2.COLOR_RGB2LAB)  # color.rgb2lab(content)
    # calculate mean
    content_mu = np.mean(content_lab, axis=tuple(range(2)))
    style_mu = np.mean(style_lab, axis=tuple(range(2)))
    # calculate standard deviation
    content_std = np.std(content_lab, axis=tuple(range(2)))
    style_std = np.std(style_lab, axis=tuple(range(2)))
    # transfer
    content_lab = (content_lab - content_mu) * (content_std / style_std) + style_mu
    content_lab = np.clip(content_lab, 0, 255)
    # convert back to RGB)
    content_rgb = cv2.cvtColor(content_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)  # color.lab2rgb(content_lab)
    content_rgb = np.clip(content_rgb, 0, 255)
    return content_rgb


if __name__ == "__main__":
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    # 1 for color_transfer_lab, 0 for color_transfer
    function = sys.argv[3] if len(sys.argv) == 4 else "0"

    content = cv2.cvtColor(cv2.imread(content_path), cv2.COLOR_BGR2RGB)
    style = cv2.cvtColor(cv2.imread(style_path), cv2.COLOR_BGR2RGB)

    img = color_transfer_lab(content, style) \
        if function == "1" else color_transfer(content, style)

    cv2.imwrite("tmp.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
