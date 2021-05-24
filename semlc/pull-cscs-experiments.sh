ssh daint 'rm ~/semlc-experiments.zip'
ssh daint 'zip -r ~/semlc-experiments.zip $SCRATCH/semlc/semlc/experiments/static/saved_models/'
scp daint:~/semlc-experiments.zip .

unzip -n semlc-experiments.zip -d experiments/static/saved_models/
cp -r experiments/static/saved_models/scratch/snx3000/bp000299/semlc/semlc/experiments/static/saved_models/* experiments/static/saved_models
rm -r experiments/static/saved_models/scratch
rm semlc-experiments.zip