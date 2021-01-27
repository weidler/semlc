#!/usr/bin/env bash

ITERATIONS=1
GPU=false

while getopts ":i:p:g:" opt; do
  case $opt in
    i) ITERATIONS="$OPTARG"
    ;;
    p) SCRIPT="$OPTARG"
    ;;
    g) GPU=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "Submitting $ITERATIONS jobs."
if [ "$GPU" = true ] ; then
  echo "Using GPU Server."
else
  echo "Using Multicore Server."
fi

if (($ITERATIONS >= 0 && $ITERATIONS <= 120)); then
  for ((i = 0 ; i < $ITERATIONS ; i++)); do
    if test -z "$SCRIPT"
    then
      if [ "$GPU" = true ] ; then
        sbatch deploy-gpu.sh
      else
        sbatch deploy.sh
      fi
    else
      if [ "$GPU" = true ] ; then
        sbatch --export=ALL,SCRIPTCOMMAND="$SCRIPT" deploy-gpu.sh
      else
        sbatch --export=ALL,SCRIPTCOMMAND="$SCRIPT" deploy.sh
      fi
    fi
  done
else
  echo "Dont do more than 120 (or less than 0)"
fi
