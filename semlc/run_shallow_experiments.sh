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

sh submit.sh -p "python3 -O run.py shallow none -w 2 4.5 -r 2 -d 0.1 --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py shallow lrn -w 2 4.5 -r 2 -d 0.1 --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py shallow gaussian-semlc -w 2 4.5 -r 2 -d 0.1 --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py shallow semlc -w 2 4.5 -r 2 -d 0.1 --auto-group --data $DATA" -i $ITERATIONS
sh submit.sh -p "python3 -O run.py shallow parametric-semlc -w 2 4.5 -r 2 -d 0.1 --auto-group --data $DATA" -i $ITERATIONS
