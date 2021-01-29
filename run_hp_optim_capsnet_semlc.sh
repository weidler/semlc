sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w 1 --group hpo-capsnet-1 --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w 3 --group hpo-capsnet-3 --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w 6 --group hpo-capsnet-6 --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w 12 --group hpo-capsnet-12 --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w 24 --group hpo-capsnet-24 --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet semlc -e 80 -w 32 --group hpo-capsnet-32 --data mnist' -i $1
