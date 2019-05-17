import pytesseract
import numpy as np
import cv2
import math
from itertools import groupby
from PIL import Image as im
from PIL import Image
import tempfile
import io
import os

def otsu_Binarization(image):
    image_blured = cv2.GaussianBlur(image, (3, 3), 0)
    #image_blured = image
    _, image_rez = cv2.threshold(image_blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #image_rez = cv2.GaussianBlur(image_rez, (3, 3), 0)
    return image_rez
#Functie care realizeaza repararea literelor discontinue, incearca sa le repare. Imaginea trebuie sa fie binarizata
def morphological_close(image):
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, k)

    k1 = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(closing, k1, iterations=1)
    return erosion

def image_De_skewing(image):

    image_original = image
    height, width = image.shape[0:2]
    cv2.bitwise_not(image,image)

    src = image
    minLineLength = width/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src,1,np.pi/180,100,minLineLength,maxLineGap)

    angle = 0.0
    #print(lines)
    nb_lines = len(lines)

    for line in lines:
        angle += math.atan2(line[0][3] * 1.0 - line[0][1] * 1.0, line[0][2] * 1.0 - line[0][0] * 1.0);

    angle /= nb_lines * 1.0

    result_angle = angle* 180.0 / np.pi

    img = image
    non_zero_pixels = cv2.findNonZero(img)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = img.shape
    rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    sizex = np.int0(wh[0])
    sizey = np.int0(wh[1])
    print(theta)
    if theta > -45 :
        temp = sizex
        sizex= sizey
        sizey= temp
    return cv2.bitwise_not(cv2.getRectSubPix(rotated, (sizey, sizex), center))

def apply_OCR(image):
    # print(image.shape)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    # config = ('-l eng --oem 1 --psm 3')

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    config = ('--tessdata-dir "tessdata" -l ron --oem 1 --psm 3')
    text = pytesseract.image_to_string(image, lang='ron')
    return text
def set_image_dpi(image):
    im = Image.fromarray(image)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    print("Aici e totul: ",im_resized.size)
    im_resized.save(temp_filename, dpi=(400, 400))
    image = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def parser(text):
    symbols = np.array(['[',']','"','`','{','}','/','?','!','>','<',';','~','_','#','*','»','“','‘','o','"','_','”','¢','“','|','£','é','§','(',')','°'])
    text=list(text)
    scoase = 0
    for i in range(len(text)):
        if(text[i-scoase] in symbols):
            text.pop(i-scoase)
            scoase = scoase + 1
    text = ''.join(text)

    word_list = text.split()
    scoase = 0

    #Eliminare cuvinte garbage:
    for i in range(len(word_list)):
        #Cuvintele cu mai putin de 3 litere sunt eliminate:
        #print(i-scoase, " ", len(word_list))
        if len(word_list[i-scoase])<3 and word_list[i-scoase]!="x":
            word_list.pop(i-scoase)
            scoase = scoase + 1
        elif len(word_list[i - scoase]) >45:
            word_list.pop(i - scoase)#cel mai lung cuvant din limba romana are 44 de litere, cuvinte mai lungi de 45 de litere nu sunt bune
            scoase = scoase + 1
        elif any((char for char, group in groupby(word_list[i - scoase]) if sum(1 for _ in group) >= 4)):
            word_list.pop(i-scoase)
            scoase = scoase + 1


    for word in word_list:
        print(word)
    return

#imaginea trebuie convertita la grayscale
def scan_receipt(image):
    # increase DPI of image
    """
    im.save("test_img.jpg"t, dpi=(300,300) )
    image = cv2.imread("test_img.jpg", )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    height, width = image.shape
    print(height, width)

    # image = image_De_skewing(image)
    # cv2.imshow('',image)
    # cv2.waitKey()
    """
    cv2.imshow('',image)
    cv2.waitKey()
    image = otsu_Binarization(image)
    cv2.imshow('',image)
    cv2.waitKey()
    """

    # text = apply_OCR(image)
    # result = parser(text)
    image = set_image_dpi(image)
    print(image.shape)

    if height > 1100:
        image = cv2.resize(image, None, fx=5., fy=5., interpolation=cv2.INTER_CUBIC)
    elif height < 300:
        image = cv2.resize(image, None, fx=12., fy=12., interpolation=cv2.INTER_CUBIC)
    elif height < 500:
        image = cv2.resize(image, None, fx=10., fy=12., interpolation=cv2.INTER_CUBIC)
    elif height < 650:
        image = cv2.resize(image, None, fx=13., fy=13., interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, None, fx=10., fy=10., interpolation=cv2.INTER_CUBIC)

    image = otsu_Binarization(image)
    image = morphological_close(image)
    print(len(image), len(image[0]))

    text = apply_OCR(image)
    result = parser(text)

    return result, image


#MAIN-UL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Aici incepe partea principala din proiect
imPath = "D:\\python projects\\ProiectImagine\\crop_scan_3.jpg"

image = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
imagInit = image
result, image = scan_receipt(image)

print(result)
cv2.imwrite( "Gray_Image.jpg", image)

