#NUM=$1
NUM=50
ITER=10

for i in $(seq 0 $(($NUM-1)));
do
    head -n -1 deploy.sh > deploy_${i}.sh;
    echo "python main.py -p ${i}" >> deploy_${i}.sh;

    for k in $(seq 1 $ITER);
    do
        echo "sbatch deploy_${i}.sh" >> start.sh;
    done
done