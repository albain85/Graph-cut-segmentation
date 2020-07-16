
import cv2
import canny as cl
import numpy as np

def mycanny(img):
    lowThreshold = 0
    highThreshold = 0

    # Gaussian Blur
    imgGB = cl.GaussBlur(img)

    imgGB = imgGB - np.min(imgGB)
    imgGB = imgGB / np.max(imgGB)
    imgGB = np.round(imgGB * 255)

    # Phase and Gradient
    GradX, GradY = cl.SobelMasking(imgGB)

    IGrad = (GradX ** 2 + GradY ** 2) ** 0.5

    PhaseMat = cl.PhaseMatrixCalculation(GradX, GradY)

    # Non-Maximum suppression
    cPhaseMat = cl.NonMaximumSuppressionStep1(PhaseMat.copy())

    nmsIGrad = cl.NonMaximumSuppressionStep2(IGrad, cPhaseMat)


    edgeMap = cl.DoubleThresholding(nmsIGrad, lowThreshold, highThreshold)

    EdgeBinMap = cl.TrackEdgesByHysteresis(edgeMap)

    CannyGradX = GradX * EdgeBinMap
    CannyGradY = GradY * EdgeBinMap

    CannyIGrad = (CannyGradX ** 2 + CannyGradY ** 2) ** 0.5

    CannyIGrad = CannyIGrad - np.min(CannyIGrad, axis=(0,1))
    CannyIGrad = CannyIGrad / np.max(CannyIGrad, axis=(0,1)) * 255
    CannyIGrad = np.uint8(CannyIGrad)


    return CannyIGrad

