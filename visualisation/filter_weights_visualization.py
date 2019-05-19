from model.network.classification import InhibitionClassificationCNN, BaseClassificationCNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from util.filter_ordering import two_opt


#mymodel = InhibitionClassificationCNN(inhibition_strategy="recurrent")
mymodel = BaseClassificationCNN()
#mymodel.load_state_dict(torch.load('../saved_models/InhibitionClassificationCNN_19.model'))
mymodel.load_state_dict(torch.load('../saved_models/BaseClassificationCNN_19.model'))
filters = mymodel.features[0].weight.data.numpy()
#print('original', filters[4])
sorted_filters = two_opt(filters)
#print('sorted', sorted_filters[1])
for i in range(len(filters)):
    for j in range(len(filters)):
        if np.allclose(filters[i], sorted_filters[j]):
            print(i+1, j+1)
fig, axs = plt.subplots(2, 3)
fig.suptitle('unsorted Baseline')
fig2, axs2 = plt.subplots(2, 3)
fig2.suptitle('sorted Baseline')
f = 0
for row in range(2):
    for col in range(3):
        img = np.swapaxes(filters[f], 0, 2)
        img = np.ndarray.astype(np.interp(img, (img.min(), img.max()), (0, 1)), dtype=float)
        img2 = np.swapaxes(sorted_filters[f], 0, 2)
        img2 = np.ndarray.astype(np.interp(img2, (img2.min(), img2.max()), (0, 1)), dtype=float)
        if row==1 and col == 1:
            print(img.shape)
            #print(sorted_filters[f])
            #print(np.swapaxes(sorted_filters[f], 0, 2))
        axs[row, col].imshow(img, cmap="gray",)
        axs2[row, col].imshow(img2, cmap="gray")
        f += 1

#plt.show()
