import numpy as np
import random
from sklearn import datasets

iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

# Подсчет софтмакса
def softmax(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)

# Нелинейная ункция активации
def relu(n):
    return np.maximum(0, n)

# Функция ошибки
def s_cross_entropy(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

# Перевод маркера в нужный формат
def to_v(y, dim):
    y_f = np.zeros((len(y), O_DIM))
    for j, i in enumerate(y):
        y_f[j, i] = 1
    return y_f

# Производная функции активации
def relu_pr(t):
    return (t>=0).astype(float)

# Предсказание
def predict(x):
    t1 = x @ W1 + B1
    h1 = relu(t1)
    t2 = h1 @ W2 + B2
    z = softmax(t2)
    return z

# Подсчет точности
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    ac = correct / len(dataset)
    return ac

# Гиперпараметры
IM_DIM = 4
H_DIM = 15
O_DIM = 3

BATCH = 60
ALPHA = 0.0001
EPOCHE = 1000


# Веса и смещения
W1 = np.random.randn(IM_DIM,H_DIM)
B1 = np.random.randn(H_DIM)
W2 = np.random.randn(H_DIM,O_DIM)
B2 = np.random.randn(O_DIM)

loss_arr = []

for _ in range(EPOCHE):

    random.shuffle(dataset)
    
    for i in range(len(dataset)// BATCH):

        batch_x, batch_y = zip(*dataset[i*BATCH : i*BATCH+BATCH])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)
        
        # Прямое распространение
        t1 = x @ W1 + B1
        h1 = relu(t1)
        t2 = h1 @ W2 + B2
        z = softmax(t2)
        E = np.sum(s_cross_entropy(z, y))

        # Обратное распространение
        y_v = to_v(y, O_DIM)
        dE_dt2 = z - y_v
        dE_dW2 = h1.T @ dE_dt2
        dE_dB2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_pr(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_dB1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Обновление весов при обратном распространении
        W1 = W1 - ALPHA * dE_dW1
        B1 = B1 - ALPHA * dE_dB1
        W2 = W2 - ALPHA * dE_dW2
        B2 = B2 - ALPHA * dE_dB2

        loss_arr.append(E)


acc = calc_accuracy()
print('Точность:', acc)

import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()
