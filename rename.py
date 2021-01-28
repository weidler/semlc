import os
import re
import sys

import simplejson as json

from config import CONFIG

exp_dir = CONFIG.MODEL_DIR
experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

for path in experiment_paths:
    eid_m = re.match("[0-9]+", str(path.split("/")[-1]))

    if eid_m:
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        if not (meta.get("group") == sys.argv[1]):
            continue

        meta["group"] = sys.argv[2]

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
