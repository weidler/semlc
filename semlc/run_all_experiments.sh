ITERATIONS=1
DATA="cifar10"

while getopts ":i:d:" opt; do
  case $opt in
    i) ITERATIONS="$OPTARG"
    ;;
    d) DATA="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Baseline
sh submit.sh -p "python3 -O run.py shallow none --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py simple none --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py alexnet none --auto-group --data $DATA" -i $ITERATIONS

# LRN
sh submit.sh -p "python3 -O run.py shallow lrn --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py simple lrn --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py alexnet lrn --auto-group --data $DATA" -i $ITERATIONS

# Gaussian
sh submit.sh -p "python3 -O run.py shallow gaussian-semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py simple gaussian-semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py alexnet gaussian-semlc --auto-group --data $DATA" -i $ITERATIONS

# SemLC
sh submit.sh -p "python3 -O run.py shallow semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py simple semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py alexnet semlc --auto-group --data $DATA" -i $ITERATIONS

# ADAPTIVE SemLC
sh submit.sh -p "python3 -O run.py shallow adaptive-semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py simple adaptive-semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py alexnet adaptive-semlc --auto-group --data $DATA" -i $ITERATIONS

# PARAMETRIC SemLC
sh submit.sh -p "python3 -O run.py shallow parametric-semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py simple parametric-semlc --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py alexnet parametric-semlc --auto-group --data $DATA" -i $ITERATIONS
