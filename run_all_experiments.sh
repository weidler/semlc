# Baseline
sh submit.sh -p 'python3 -O run.py shallow none --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py simple none --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py alexnet none --auto-group' -i $1

# LRN
sh submit.sh -p 'python3 -O run.py shallow lrn --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py simple lrn --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py alexnet lrn --auto-group' -i $1

# SemLC
sh submit.sh -p 'python3 -O run.py shallow semlc --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py simple semlc --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py alexnet semlc --auto-group' -i $1

# ADAPTIVE SemLC
sh submit.sh -p 'python3 -O run.py shallow adaptive-semlc --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py simple adaptive-semlc --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py alexnet adaptive-semlc --auto-group' -i $1

# PARAMETRIC SemLC
sh submit.sh -p 'python3 -O run.py shallow parametric-semlc --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py simple parametric-semlc --auto-group' -i $1
sh submit.sh -p 'python3 -O run.py alexnet parametric-semlc --auto-group' -i $1

# Capsule Networks
sh submit.sh -p 'python3 run.py capsnet none -e 1 --auto-group --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet lrn -e 1 --auto-group --data mnist' -i $1
sh submit.sh -p 'python3 run.py capsnet semlc -e 1 --auto-group --data mnist' -i $1
