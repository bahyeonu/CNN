# coding: utf-8
import sys, os

sys.path.append(os.pardir)
import numpy as np
from functions import *
from bird import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def train(self, x_train, t_train, x_test, t_test):
        iters_num = 10000  # 반복 횟수를 적절히 설정한다.
        train_size = x_train.shape[0]
        batch_size = 100  # 미니배치 크기
        learning_rate = 0.1

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        # 1에폭당 반복 수
        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters_num):
            # 미니배치 획득
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 기울기 계산
            grad = self.gradient(x_batch, t_batch)

            # 가중치 및 바이어스 갱신
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.params[key] -= learning_rate * grad[key]

            # 학습 경과 기록
            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            # 1에폭당 정확도 계산
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        return

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return mean_squared_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


if __name__ == '__main__':
    # 신경망 만들기
    network = TwoLayerNet(input_size=3136, hidden_size=50, output_size=10)

    # MNIST 데이터 읽어오기
    (x_train, t_train), (x_test, t_test) = load_bird(normalize=False, one_hot_label=True)

    # 숫자 인식 훈련
    network.train(x_train, t_train, x_test, t_test)

    # 숫자 인식 (테스트 데이터 중에서)
    select = np.random.randint(len(x_test))
    img = x_test[select]
    label = t_test[select]
    num = network.predict(img)  # 숫자 인식
    print("정답:", switch(np.argmax(label)))
    print("예측:", switch(np.argmax(num)))
    img_show(img.reshape(56, 56))

    # 그림 파일에서 숫자 인식
    while True:
        f = input("파일이름:")
        if f == "--":
            break

        img = Image.open(f)
        img.thumbnail((32, 32))
        img = img.resize((56, 56))
        img = img.convert("L")
        img = np.invert(img)
        im2arr = np.array(img)
        im2arr = binal(im2arr)
        im2arr = im2arr.flatten()

        num = network.predict(im2arr)
        print("예측:", switch(np.argmax(num)))
        img_show(im2arr.reshape(56, 56))