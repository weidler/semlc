network = "capsnet"
dataset = "mnist"
strategy = "semlc"

widths = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5]

CONFIG = {}
index = 0

with open(f"run_hp_fine_opt_{network}_{strategy}.sh", 'a') as f:
    for w in widths:
        f.write(f"sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w {w} --group hpo-{network}-{w} --data mnist' -i $1\n")
