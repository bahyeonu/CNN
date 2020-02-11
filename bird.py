# coding: utf-8

import pickle
import os
import cv2
import numpy as np
from numpy import array
from random import shuffle
from PIL import Image

key_file = {
    'train_img': 'train_image.npy',
    'train_label': 'train_label.npy',
    'test_img': 'test_image.npy',
    'test_label': 'test_label.npy'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/bird.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 56, 56)
img_size = 3136


def _create():
    train_dir = os.path.dirname(os.path.abspath(__file__)) + '/images'
    test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test'
    train_dir_list = array(os.listdir(train_dir))
    test_dir_list = array(os.listdir(test_dir))

    train_image = []
    train_label = []
    test_image = []
    test_label = []

    def _create_npy(dir, list, image, label):
        for index in range(len(list)):
            path = os.path.join(dir, list[index]) + '/'
            img_list = os.listdir(path)

            for img in img_list:
                img_path = os.path.join(path, img)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, dsize=(56, 56), interpolation=cv2.INTER_AREA)
                image.append([np.array(img)])
                label.append(index)

        temp = [[x, y] for x, y in zip(image, label)]
        shuffle(temp)
        image = [arr[0] for arr in temp]
        label = [arr[1] for arr in temp]

        image = np.reshape(image, (-1, 3136))

        image = np.array(image).astype(np.uint8)
        label = np.array(label).astype(np.uint8)

        return image, label

    train_image, train_label = _create_npy(train_dir, train_dir_list, train_image, train_label)
    np.save("train_image.npy", train_image)
    np.save("train_label.npy", train_label)

    test_image, test_label = _create_npy(test_dir, test_dir_list, test_image, test_label)
    np.save("test_image.npy", test_image)
    np.save("test_label.npy", test_label)


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
            labels = np.load(file_name)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
            data = np.load(file_name)
    print("Done")

    return data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_bird():
    _create()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_bird(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기

    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label :
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다.

    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):
        init_bird()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 56, 56)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def binal(numpy_array, threshold=150):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def switch(value):
    return{
        0: 'Black_footed_Albatross',
        1: 'Laysan_Albatross',
        2: 'Groove_billed_Ani',
        3: 'Red_winged_Blackbird',
        4: 'Rusty_Blackbird',
        5: 'Bobolink',
        6: 'Indigo_Bunting',
        7: 'Eastern_Towhee',
        8: 'Pelagic_Cormorant',
        9: 'Bronzed_Cowbird'
    }.get(value, -1)


if __name__ == '__main__':
    init_bird()