import numpy as np
import json

grid_size = 7
range_width = [3, 10]
range_damp = [0.1, 0.2]

widths = np.linspace(start=range_width[0], stop=range_width[1], num=grid_size)
damps = np.linspace(start=range_damp[0], stop=range_damp[1], num=grid_size)

CONFIG = {}
index = 0
for w in widths:
    for d in damps:
        CONFIG.update({
            index: {
                'widths': [w],
                'damps': [d]
            }
        })
        index += 1

print(CONFIG)
with open("hp_params.json", 'w') as f:
    json.dump(CONFIG, f, indent=4)
