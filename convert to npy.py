import numpy as np

size = np.array([
    100,
    128,
    128
])

data = []
for x in open("Dat phantoms/efg3cut.dat"):
    x = float(x)
    if x == 0.01: #Мягкие ткани
        data.append(0.019) #133
    elif x == 0.015: #Лёгкие
        data.append(0.015) #107
    elif x == 0.15: #печень
        data.append(0.15) #1020
    elif x == 0.2: #толстая кишка
        data.append(0.3) 
    elif x == 0.24: #сердце
        data.append(0.11) #735
    elif x == 0.5: #желчный пузырь
        data.append(3.)
    else:
        data.append(0.)
    # if x == 0.04:
    #     data.append(1)
    # elif x == 0.15:
    #     data.append(2)
    # elif x == 0.28:
    #     data.append(3)
    # else:
    #     data.append(0)
    # data.append(x)
# data = np.array(data, dtype=np.uint8).reshape(size)
data = np.array(data).reshape(size)
data = np.rot90(data, axes=(0, 2))
data = np.rot90(data, axes=(0, 1))
data = data[::-1]
print(np.unique(data))
np.save('Phantoms/efg3cut.npy', data)