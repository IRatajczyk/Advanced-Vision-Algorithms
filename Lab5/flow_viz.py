from numpy import ndarray, zeros, pi, float64, uint8
from cv2 import cartToPolar, normalize, COLOR_HSV2BGR, cvtColor, NORM_MINMAX
from matplotlib.pyplot import imshow, show


def visualize_optical_flow(
        u: ndarray,
        v: ndarray,
        ) -> None:
        hsv = zeros((u.shape[0], u.shape[1], 3), dtype= uint8)
        mag, ang = cartToPolar(u.astype(float64), v.astype(float64))
        hsv[:,:,0] = ang*180/pi/2
        hsv[:,:,1] = 255
        hsv[:,:,2] = normalize(mag,None,0,255,NORM_MINMAX)
        rgb = cvtColor(hsv, COLOR_HSV2BGR)
        imshow(rgb)
        show()

