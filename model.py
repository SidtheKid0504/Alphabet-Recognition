import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

y = pd.read_csv('https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv')['labels']
X = np.load('image (1).npz')['arr_0']

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=9, train_size=10725, test_size=2500)
train_X_scaled = train_X / 255
test_X_scaled = test_X / 255

lr = LogisticRegression(solver='saga', multi_class='multinomial')

lr.fit(train_X_scaled, train_y)

y_preds = lr.predict(test_X_scaled)
acc = accuracy_score(test_y, y_preds)
print(str(acc*100) + '%')

cap = cv2.VideoCapture(0)

while(True):
  try:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    im_pil = Image.fromarray(roi)

    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

cap.release()
cv2.destroyAllWindows()