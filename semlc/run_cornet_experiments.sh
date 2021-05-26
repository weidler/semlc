VARIANT=$1

if [ "$VARIANT" != "z" ] && [ "$VARIANT" != "s" ] ; then
  echo "illegal variant"
  exit 1
fi

sh submit.sh -p "python3 run.py cornet-$VARIANT none -w 2 4.5 -e 250 --auto-group --data cifar10" -i $2
sh submit.sh -p "python3 run.py cornet-$VARIANT parametric-semlc -w 2 4.5 -e 250 --auto-group --data cifar10" -i $2
sh submit.sh -p "python3 run.py cornet-$VARIANT gaussian-semlc -w 2 4.5 -e 250 --auto-group --data cifar10" -i $2
sh submit.sh -p "python3 run.py cornet-$VARIANT semlc -e 250 -w 2 4.5 --auto-group --data cifar10" -i $2