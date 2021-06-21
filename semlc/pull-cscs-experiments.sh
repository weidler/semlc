ssh daint 'rm ~/semlc-experiments.zip'
#ssh daint 'zip -r ~/semlc-experiments.zip $SCRATCH/semlc/semlc/experiments/static/saved_models/'
ssh daint 'zip -r ~/semlc-hpoptims.zip $SCRATCH/semlc/semlc/experiments/static/hpoptims/'
#scp daint:~/semlc-experiments.zip .
scp daint:~/semlc-hpoptims.zip .

#unzip -n semlc-experiments.zip -d experiments/static/saved_models/
unzip -n semlc-hpoptims.zip -d experiments/static/hpoptims/
#cp -r experiments/static/saved_models/scratch/snx3000/bp000299/semlc/semlc/experiments/static/saved_models/* experiments/static/saved_models
cp -r experiments/static/hpoptims/scratch/snx3000/bp000299/semlc/semlc/experiments/static/hpoptims/* experiments/static/hpoptims
#rm -r experiments/static/saved_models/scratch
rm -r experiments/static/hpoptims/scratch
#rm semlc-experiments.zip
rm semlc-hpoptims.zip