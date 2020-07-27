# Resize image

import os
import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import fatal_error


def auto_crop_edit(img, obj, padding_t=0, padding_b=0, padding_l=0, padding_r=0 color='black'):
    """Resize image.

    Inputs:
    img       = RGB or grayscale image data
    obj       = contours
    padding_l = padding to the left of the cropped image
    padding_r = padding to the right of the cropped image
    padding_t = padding to the top of the cropped image
    padding_b = padding to the bottom of the cropped image
    color     = either 'black', 'white', or 'image'

    Returns:
    cropped   = cropped image

    :param img: numpy.ndarray
    :param obj: list
    :param padding_x: int
    :param padding_y: int
    :param color: str
    :return cropped: numpy.ndarray
    """

    params.device += 1
    img_copy = np.copy(img)
    img_copy2 = np.copy(img)

    # Get the height and width of the reference image
    height, width = np.shape(img)[:2]

    x, y, w, h = cv2.boundingRect(obj)
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)

    crop_img = img[y:y + h, x:x + w]

    offsett = int(np.rint(padding_t))
    offsetb = int(np.rint(padding_b))
    offsetl = int(np.rint(padding_l))
    offsetr = int(np.rint(padding_r))

    if color.upper() == 'BLACK':
        colorval = (0, 0, 0)
        cropped = cv2.copyMakeBorder(crop_img, offsett, offsetb, offsetl, offsetr, cv2.BORDER_CONSTANT, value=colorval)
    elif color.upper() == 'WHITE':
        colorval = (255, 255, 255)
        cropped = cv2.copyMakeBorder(crop_img, offsett, offsetb, offsetl, offsetr, cv2.BORDER_CONSTANT, value=colorval)
    elif color.upper() == 'IMAGE':
        # Check whether the ROI is correctly bounded inside the image
        if x - offsetl < 0 or y - offsett < 0 or x + w + offsetl > width or y + h + offsett > height:
            cropped = img_copy2[y - offsett:y + h + offsett, x - offsetl:x + w + offsetl]
        else:
            # If padding is the image, crop the image with a buffer rather than cropping and adding a buffer
            cropped = img_copy2[y:y + h, x:x + w]
    else:
        fatal_error('Color was provided but ' + str(color) + ' is not "white", "black", or "image"!')

    if params.debug == 'print':
        print_image(img_copy, os.path.join(params.debug_outdir, str(params.device) + "_crop_area.png"))
        print_image(cropped, os.path.join(params.debug_outdir, str(params.device) + "_auto_cropped.png"))
    elif params.debug == 'plot':
        if len(np.shape(img_copy)) == 3:
            plot_image(img_copy)
            plot_image(cropped)
        else:
            plot_image(img_copy, cmap='gray')
            plot_image(cropped, cmap='gray')

    return cropped
