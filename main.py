
import cv2

import numpy as np

print(cv2.__version__)

def showimage():
    image = cv2.imread("image/lenna.png", cv2.IMREAD_COLOR)
    cv2.imshow("lenna", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def copyimage():
    image = cv2.imread("image/lenna.png",cv2.IMREAD_COLOR)
    cv2.imwrite("image/lenna_copy.png",image)
    image_copy = cv2.imread("image/lenna_copy.png",cv2.IMREAD_COLOR)
    cv2.imshow("lenna_copy",image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showvideo():
    try:
        print("start video")
        video = cv2.VideoCapture("video/sample.mp4")
    except:
        print("fail to play video")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("video reading end")
            break
        cv2.imshow("video",frame)
        k = cv2.waitKey(1)&0xff
        if k == 27:
            break
    video.release()
    cv2.destroyAllWindows()

def resizeimage():
    image = cv2.imread("image/lenna.png",cv2.IMREAD_COLOR)

    image_rs = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("original", image)
    cv2.imshow("fx=0.5. fy=0.5 resized image", image_rs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cropimage():
    image = cv2.imread("image/lenna.png",cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    image_cr = image.copy()
    image_cr = image[int(0.25*width):int(0.75*width),int(0.25*height):int(0.75*height)]
    cv2.imshow("original", image)
    cv2.imshow("cropped image", image_cr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cvtimage_bgr2rgb():
    image = cv2.imread("image/lenna.png",cv2.IMREAD_COLOR)

    image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("original", image)
    cv2.imshow("converted image", image_cvt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cvtimage_rgb2gray():
    image = cv2.imread("image/lenna.png",cv2.IMREAD_COLOR)

    image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", image)
    cv2.imshow("converted image", image_cvt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blurimage():
    image = cv2.imread("image/lenna.png", cv2.IMREAD_COLOR)
    kernel = np.ones((5,5), np.float32)/25
    image_bl = cv2.filter2D(image, -1, kernel)

    cv2.imshow("original", image)
    cv2.imshow("blurred image", image_bl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edgeimage():
    image = cv2.imread("image/lenna.png", cv2.IMREAD_COLOR)
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype = np.float32)
    image_ed = cv2.filter2D(image, -1, kernel)

    cv2.imshow("original", image)
    cv2.imshow("blurred image", image_ed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveimage():
    image = cv2.imread("image/lenna.png", cv2.IMREAD_COLOR)

    image_rs = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite("image/lenna_resize.png", image_rs)

    height, width = image.shape[:2]
    image_cr = image.copy()
    image_cr = image[int(0.25 * width):int(0.75 * width), int(0.25 * height):int(0.75 * height)]
    cv2.imwrite("image/lenna_crop.png", image_cr)

    image_cvt_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("image/lenna_rgb.png", image_cvt_bgr)

    image_cvt_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("image/lenna_gray.png", image_cvt_gray)

    kernel_blur = np.ones((5, 5), np.float32) / 25
    image_bl = cv2.filter2D(image, -1, kernel_blur)
    cv2.imwrite("image/lenna_blur.png", image_bl)

    kernel_edge = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype = np.float32)
    image_ed = cv2.filter2D(image, -1, kernel_edge)
    cv2.imwrite("image/lenna_edge.png", image_ed)


# showimage()
# copyimage()
# showvideo()
# resizeimage()
# cropimage()
# cvtimage_bgr2rgb()
# cvtimage_rgb2gray()
# blurimage()
# edgeimage()

saveimage()

