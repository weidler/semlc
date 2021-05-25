sh submit.sh -p 'python3 run.py cornet-z none -e 250 --auto-group --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py cornet-z parametric-semlc -e 250 --auto-group --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py cornet-z gaussian-semlc -e 250 --auto-group --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py cornet-z semlc -e 250 -w 12 --auto-group --data mnist' -i $1 -g