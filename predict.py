from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os
import shutil


# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that
# were given. The evaluation code may give unexpected results if this convention is not followed.

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILENAME)


def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

def decaptcha(filenames):
    captcha_image_files = filenames
    numChars = 3 * np.ones((len(filenames),))
    count = 0
    codes = []
    for image_file in captcha_image_files:
        image = cv2.imread(image_file)
        cv2.imshow("Output", image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low = np.array([0, 100, 250])
        high = np.array([179, 255, 255])
        mask_fore = cv2.inRange(hsv, low, high)
        image = cv2.bitwise_not(mask_fore)

        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = []

        temp=0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w / h > 1.25:

                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h)) 
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        output = cv2.merge([image] * 3)
        predictions = []


        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            letter_image = resize_to_fit(letter_image, 20, 20)
        
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)
        
            prediction = model.predict(letter_image)

            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)
            temp = temp + 1
    
        numChars[count] = temp
        count = count + 1
        captcha_text = "".join(predictions)
        print("CAPTCHA text is: {}".format(captcha_text))
        codes.append(captcha_text)

    
    return (numChars, codes)
