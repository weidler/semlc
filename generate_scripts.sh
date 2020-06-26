#NUM=$1
NUM=48
ITER=5

# clean start.sh
true > start.sh

# generate
for i in $(seq 0 $(($NUM-1)));
do
    head -n -1 deploycpu.sh > deploy_${i}.sh;
    echo "python main.py CLC frozen -p ${i}" >> deploy_${i}.sh;

    for k in $(seq 1 $ITER);
    do
        echo "sbatch deploy_${i}.sh" >> start.sh;
    done
done