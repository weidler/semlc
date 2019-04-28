import torch
from torch.optim import Adam

from model.network.inhibition import InhibitionClassificationCNN
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from misc_functions import preprocess_image, recreate_image


class FilterVisualization:

    def __init__(self, model):
        self.model = model
        self.layers = model.get_layers_for_visualization()

    def visualize_filters_by_layer(self, num_layer: int, steps: int = 30):
        selected_layer: nn.Module = self.layers[num_layer]
        kernel_size = selected_layer.kernel_size
        random_image = np.uint8(np.random.uniform(150, 180, (kernel_size[0], kernel_size[1], 3)))
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        num_filters = selected_layer.out_channels
        print(num_filters)
        fig, axs = plt.subplots(num_filters)
        for selected_filter in range(num_filters):
            for i in range(1, steps+1):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                x = processed_image
                for index, layer in enumerate(self.layers):
                    x = layer(x)
                    print(index, layer)
                    if index == num_layer:
                        break
                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                conv_output = x[0, selected_filter]
                loss = -torch.mean(conv_output)
                # print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(str(loss.data.numpy())))
                # Backward
                loss.backward()
                # Update image
                optimizer.step()
                # Recreate image
                created_image = recreate_image(processed_image)
                axs[selected_filter].imshow(created_image)
        plt.show()


if __name__ == "__main__":
    mymodel = InhibitionClassificationCNN()
    # print(model.get_layers_for_visualization()[0])
    vis = FilterVisualization(mymodel)
    vis.visualize_filters_by_layer(0)
