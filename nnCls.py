# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:46:08 2018

@author: usman
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
feature_set, labels = datasets.make_moons(50, noise=0.10)
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap=plt.cm.winter)
labels = labels.reshape(50, 1)


class Network:
    def __init__(self):
        self.wh = np.random.rand(len(feature_set[0]), 4)  # self.bh = np.random.rand(len(feature_set[0]), 1)
        self.wo = np.random.rand(4, 1)  # self.bo = np.random.rand(1, 1)
        self.lr = 0.5
        self.avrg = 0.05
        self.error_sum = 100
        
    def train(self):
        while self.error_sum > self.avrg: 
            self.forward()
            self.backward(self.ao, labels)
        
    def test(self):
        for k, v in [(0.97475324,  0.10398468), (1.96357787,  0.25012318), (-0.5,  0.25), (1,  -0.5), (1,  0.25), (-1,  0.25)]:
            single_point = np.array([k,  v])
            h = self.sigmoid(np.dot(single_point, self.wh))
            result = self.sigmoid(np.dot(h, self.wo))
            print(k, v, ' ------> ', result)
            
    @staticmethod
    def show():
        plt.show()
        
    def forward(self):
        self.zh = np.dot(feature_set, self.wh)  # + self.bh
        self.ah = self.sigmoid(self.zh)
        self.zo = np.dot(self.ah, self.wo)  # + self.bo
        self.ao = self.sigmoid(self.zo)
            
            
    def backward(self, ao, labels):
        # phase 1
        error_out = ((1 / 2) * (np.power((ao - labels), 2)))
        self.error_sum = error_out.sum()
        dcost_dao = ao - labels
        dao_dzo = self.sigmoid_der(self.zo)
        dzo_dwo = self.ah
        dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)
        # phase 2
        dcost_dzo = dcost_dao * dao_dzo
        dzo_dah = self.wo
        dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
        dah_dzh = self.sigmoid_der(self.zh)
        dzh_dwh = feature_set
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        # adjust weights
        self.wh -= self.lr * dcost_wh
        self.wo -= self.lr * dcost_wo
       
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
       
    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
    
if __name__ == '__main__':
    net = Network()
    net.train()
    net.test()
    net.show()
    
    
    
    
    
    
    
    
    
    
# def sigmoid(x):
#     return 1/(1+np.exp(-x))


# def sigmoid_der(x):
#     return sigmoid(x) * (1-sigmoid(x))


# wh = np.random.rand(len(feature_set[0]), 4)
# wo = np.random.rand(4, 1)
# lr = 0.5

# for epoch in range(50000):
#     # feedforward
#     zh = np.dot(feature_set, wh)
#     ah = sigmoid(zh)

#     zo = np.dot(ah, wo)
#     ao = sigmoid(zo)

#     # Phase1 =======================

#     error_out = ((1 / 2) * (np.power((ao - labels), 2)))
#     print(error_out.sum())

#     dcost_dao = ao - labels
#     dao_dzo = sigmoid_der(zo)
#     dzo_dwo = ah

#     dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

#     # Phase 2 =======================

#     # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
#     # dcost_dah = dcost_dzo * dzo_dah
    
#     dcost_dzo = dcost_dao * dao_dzo
#     dzo_dah = wo
#     dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
#     dah_dzh = sigmoid_der(zh)
#     dzh_dwh = feature_set
    
#     dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

#     # Update Weights ================

#     wh -= lr * dcost_wh
#     wo -= lr * dcost_wo
    
# single_point = np.array([0.97475324,  0.10398468])
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)


# single_point = np.array([1.96357787,  0.25012318])
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)
    
# single_point = np.array([1.7339206,  -0.11933111])
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)

# single_point = np.array([-1,  0.25])
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)

# single_point = np.array([1,  0.25])
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)

# single_point = np.array([1,  -0.5])
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)

# single_point = np.array([-0.5,  0.25])  
# h = sigmoid(np.dot(single_point, wh))
# result = sigmoid(np.dot(h, wo))
# print(result)

# plt.show()



