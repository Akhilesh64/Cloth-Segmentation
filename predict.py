import os, cv2
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
import scipy.io

# Read the labels from the matlab label file and save it as a dictionary
mat = scipy.io.loadmat('/content/clothing-co-parsing/label_list.mat')
labels = {0:'background'}

for i in range(1,59):
  labels[i]=mat['label_list'][0][i][0]

test = '/content/Data/test'
model = load_model('/content/drive/MyDrive/task2_model.h5', compile = False)

if not os.path.exists('results'):
  os.makedirs('./results/')

#Run model predictions on the test folder provided
for file in tqdm(sorted(os.listdir(test))):
  img = cv2.imread(os.path.join(test, file))
  img = cv2.resize(img,(256,256))
  im = img/255.0
  im = np.expand_dims(im, axis= 0)
  preds = model.predict(im)
  preds = np.argmax(preds[0], axis= -1)
  preds[preds==41] = 0
  preds[preds==19] = 0
  classes = np.unique(preds)
  label = []
  for i in classes[1:]:
      label.append(labels[i])
  preds[preds>0]=255
  img[preds==0]=0
  cv2.putText(img, ' '.join(label), (2, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
  cv2.imwrite(os.path.join('results', file),img)