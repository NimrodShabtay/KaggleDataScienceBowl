import pre_process
import cv2 as cv
import numpy as np


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

for i in range(70, 120, 10):
    img = pre_process.first_patient_pixels[i]
    kernel = np.ones((15,15),np.int16)
    # lungs are white
    image = cv.convertScaleAbs(img)
    cv.imshow('ImgOriginal', image)
    threshold = cv.mean(image)
    th, img_th = cv.threshold(image, threshold[0], 255, cv.THRESH_OTSU)
    # TODO: still needs to handle non-complete shapes - maybe using closing.
    opening = cv.morphologyEx(img_th, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    cv.imshow('ImgClosing', closing)
    cv.waitKey(0)
    # Find the largest contour and extract it
    im, contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contours.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))
    # Create a mask from the largest contour
    mask = np.zeros_like(closing)
    #improve this section
    contrList = []
    for j in range(len(contours)):
        if set(contours[j][0][0]).intersection([0,0]):
            continue
        else:
            contrList.append(contours[j])
        cv.fillPoly(mask, contrList, 1)
    finalImg = image * mask

    cv.imshow('final', finalImg)
    cv.waitKey(0)
