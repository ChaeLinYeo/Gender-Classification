from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

import tensorflow as tf

seed = 5
tf.set_random_seed(seed)
np.random.seed(seed)

# caltech_dir = '/Users/macbook/Desktop/genderdataset/test'
caltech_dir = '/Users/macbook/Desktop/selfy'

image_w = 32
image_h = 32

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)

X = np.array(X)
X = X.astype(float) / 255
model = load_model('/Users/macbook/Desktop/modeltest/test.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0
for i in prediction:
    if i >= 0.5: print("해당 " + filenames[cnt].split("/")[6] + "  이미지는 여자로 추정됩니다.")
    else : print("해당 " + filenames[cnt].split("/")[6] + "  이미지는 남자로 추정됩니다.")
    cnt += 1