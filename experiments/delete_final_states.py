import os
import re
import shutil
import sys

import simplejson as json

from config import CONFIG

exp_dir = "../" + CONFIG.MODEL_DIR
experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

for path in experiment_paths:
    eid_m = re.match("[0-9]+", str(path.split("/")[-1]))

    if eid_m:

        final_path = os.path.join(path, "final.parameters")
        if os.path.isfile(final_path):
            os.remove(final_path)
            print(f"Deleted {final_path}")
