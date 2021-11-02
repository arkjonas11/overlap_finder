import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys

def open_img(image):
    img = cv.imread(image,cv.IMREAD_GRAYSCALE)
    return img

if __name__ == "__main__":
    print("Opening photos...")
    img, img2 = open_img(sys.argv[1]), open_img(sys.argv[2])
    
    print("Photos opened. \nMatching points...")
    sift = cv.SIFT_create()
    kp, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    diff = 0
    for m,n in matches:
        if m.distance < 0.60*n.distance:
            good.append([m])
            diff += kp[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]

    diff_mean = diff/len(good)
    print("\033[1;32;49mOVERLAP ->" , '%.1f'%((770-diff_mean)/770*100),"% \033[0;0m")

    if (len(sys.argv) > 3) and (sys.argv[3] == "true"):
        img3 = cv.drawMatchesKnn(img,kp,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()

