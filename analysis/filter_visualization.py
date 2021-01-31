"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""

import copy
import os

import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm

from analysis.util import load_model_by_id

from torchvision.transforms.functional import resize, center_crop, rgb_to_grayscale, gaussian_blur


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])

    if im_as_var.shape[0] == 3:
        for c, _ in enumerate(recreated_im):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]

    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0

    if recreated_im.shape[0] == 3:
        recreated_im = np.uint8(np.round(recreated_im * 255))

    recreated_im = recreated_im.transpose(1, 2, 0)
    return recreated_im


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_filter = selected_filter
        self.conv_output = torch.tensor(0)

        self.upscaling_steps = 6
        self.upscaling_factor = 1.2

        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model.get_final_block1_layer().register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self, resize_crop=False, blur=False):
        # Hook the selected layer
        self.hook_layer()
        size = 32

        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (size, size, self.model.input_channels)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)

        for i in range(self.upscaling_steps):
            for j in range(1, 21):
                optimizer.zero_grad()

                self.model.extract_features(processed_image)
                loss = - torch.mean(self.conv_output)
                loss.backward()
                optimizer.step()

            if resize_crop:
                processed_image = center_crop(
                    resize(processed_image, [int(size * self.upscaling_factor), int(size * self.upscaling_factor)],
                           interpolation=PIL.Image.BICUBIC),
                    [size, size])

            if blur and not i == self.upscaling_steps - 1:
                processed_image = Variable(gaussian_blur(processed_image, [3, 3], sigma=[0.4, 0.4]), requires_grad=True)

        created_image = recreate_image(processed_image)
        return torch.from_numpy(created_image.transpose(2, 0, 1))


if __name__ == '__main__':
    # pretrained_model = load_model_by_id("1611709222457144")  # none
    pretrained_model = load_model_by_id("1612027536819388")  # semlc

    images = []
    for filter_pos in tqdm(range(4)):
        layer_vis = CNNLayerVisualization(pretrained_model, filter_pos)

        # Layer visualization with pytorch hooks
        images.append(layer_vis.visualise_layer_with_hooks(blur=False))

    plt.imshow(torchvision.utils.make_grid(images).numpy().transpose(1, 2, 0), cmap="gray")
    plt.show()