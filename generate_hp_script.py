network = "capsnet"
dataset = "mnist"
strategy = "semlc"

widths = [1, 3, 6, 12, 24, 32]

CONFIG = {}
index = 0

with open(f"run_hp_optim_{network}_{strategy}.sh", 'a') as f:
    for w in widths:
        f.write(f"sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w {w} -d 0.2 --group hpo-{network}-{w} --data mnist' -i $1 -g\n")
