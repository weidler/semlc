sh submit.sh -p 'python3 run.py capsnet none -e 180 --auto-group --data cifar10-bw' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet lrn -e 180 --auto-group --data cifar10-bw' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 180 -w 12 --auto-group --data cifar10-bw' -i $1 -g