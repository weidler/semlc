sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w 1 -d 0.2 --group hpo-capsnet-1 --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w 3 -d 0.2 --group hpo-capsnet-3 --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w 6 -d 0.2 --group hpo-capsnet-6 --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w 12 -d 0.2 --group hpo-capsnet-12 --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w 24 -d 0.2 --group hpo-capsnet-24 --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 100 -w 32 -d 0.2 --group hpo-capsnet-32 --data mnist' -i $1 -g
