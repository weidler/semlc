sh submit.sh -p 'python3 run.py capsnet none -e 250 --auto-group --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet lrn -e 250 --auto-group --data mnist' -i $1 -g
sh submit.sh -p 'python3 run.py capsnet semlc -e 250 --auto-group --data mnist' -i $1 -g