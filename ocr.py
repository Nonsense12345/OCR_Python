import cv2
import tr
from PIL import Image


def extract_text_from_image(img):
    cnrr = tr.CRNN()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    txts = ""

    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
    
        if w > 1 and h > 2 and h < 80:  
            roi = gray[y:y+h,x:x+w].copy()
            chars, scores = cnrr.run(roi)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            txts = txts + "".join(chars) + "\n"
    return txts


