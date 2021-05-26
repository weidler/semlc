network = "cornetz"
dataset = "cifar10"
strategy = "semlc"

epsp_widths = [2, 2.5, 3.0, 3.5, 4.0]
ipsp_width_adds = [1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 10.0, 15.0]
ratio = [2, 3, 4]
damping = [0.05, 0.1]
n_options = len(epsp_widths) * len(ipsp_width_adds) * len(ratio) * len(damping)

print(f"{n_options} will be tested.")

CONFIG = {}
index = 0

with open(f"run_hp_opt_{network}_{strategy}.sh", 'w') as f:
    for w1 in epsp_widths:
        for w2_add in ipsp_width_adds:
            for r in ratio:
                for d in damping:
                    w2 = w1 + w2_add
                    f.write(f"sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w {w1} {w2} -r {r} -d {d} --group "
                            f"hpo-{network}-{'-'.join([str(w1), str(w2), str(r), str(d)])} --data cifar10' -i $1\n")
