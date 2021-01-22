import os

from matplotlib.colors import ListedColormap
import seaborn as sns


class Config:
    pass


CONFIG = Config()
CONFIG.PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG.DATA_DIR = os.path.join(CONFIG.PROJECT_PATH, "../data")
CONFIG.MODEL_DIR = "experiments/static/saved_models/"
CONFIG.AVAILABLE_TEST_SETTINGS = ["default", "occlusion-r", "occlusion-v", "occlusion-h", "sprinkle-contrast",
                                  "sprinkle-blur"]

CONFIG.COLORMAP = list(ListedColormap(sns.color_palette("deep")).colors)
CONFIG.COLORMAP_HEX = ['#%02x%02x%02x' % tuple(int(c * 255) for c in color) for color in CONFIG.COLORMAP]

RGB_TO_GREYSCALE_WEIGHTS = (0.2989, 0.5870, 0.1140)