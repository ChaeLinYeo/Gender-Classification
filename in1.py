# import numpy  as np
# from PIL import Image
# import os

# path = '/Users/macbook/Desktop/man'
# count = 0
# for item in os.listdir(path):
#     print (item)
#     imgpath = path +'/' + item
#     img = Image.open(imgpath)
#     arr = np.array(img)
#     print (arr)
#     print ('--------------')
#     img = Image.fromarray(arr)
#     print (count)
#     img.save(str(count) +  "output.jpg")
#     count = count +1

from keras.utils import np_utils
from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split


#이미지를 학습시키기 전에 배열화 시켜주기.


#변수에 학습시킬 데이터의 위치 주소 담기
imgpath = '/Users/macbook/Desktop/genderdataset/train'
#학습시킬 데이터의 카테고리
categories = ['man', 'woman']
np_classes = len(categories)

#데이터셋 사진 크기를 32x32 크기로 변환한다.
img_w = 32
img_h = 32
pixel = img_h * img_w * 3

#케라스에서 학습을 진행하기 위해 이미지와 카테고리를 나눈다.
train_data = []
train_cat = []

#카테고리별로 돌면서... imgpath의 경로로부터 이미지를 읽는다.
for index, category in enumerate(categories):
    imgpath_detail = imgpath + "/" + category
    files = glob.glob(imgpath_detail+"/*.jpg")


#이미지 변환을 해준다. 각 이미지를 가지고와서 RGB 형태로 변환해준 뒤 resize해준다.
#그리고 그 값을 numpy 배열로 바꾸고 배열에 추가해준다.
#동시에 카테고리 값도 넣어준다 (Y)
    for idx, cat in enumerate(files):
        try:
            img = Image.open(cat)
            img = img.convert("RGB")
            img = img.resize((img_w, img_h))
            data = np.asarray(img)
            #Y는 0 아니면 1이니까 index값으로 넣는다.
            train_data.append(data)
            train_cat.append(index)
            if idx % 100 == 0:
                print(category, " : ", cat)
        except:
            print(category, str(i)+" 번째에서 에러 ")
X = np.array(train_data)
Y = np.array(train_cat)

#훈련 데이터, 테스트 데이터 (검증 데이터)로 나눠준다.
#test_size : 테스트 분할에 포함할 데이터 세트의 비율. 기본이 0.25이고 여기선 0.2로 설정했다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#훈련 데이터, 테스트 데이터로 나눈 것을 numpy 형태로 저장한다.
xy = (X_train, X_test, Y_train, Y_test)

np.save("/Users/macbook/Desktop/nptest/test.npy", xy)
