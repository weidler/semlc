#!/usr/bin/env bash

ITERATIONS=1

while getopts ":i:p:" opt; do
  case $opt in
    i) ITERATIONS="$OPTARG"
    ;;
    p) SCRIPT="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "Submitting $ITERATIONS jobs."
if (($ITERATIONS >= 0 && $ITERATIONS <= 120)); then
  for ((i = 0 ; i < $ITERATIONS ; i++)); do
    if test -z "$SCRIPT"
    then
      sbatch deploy.sh
    else
      sbatch --export=ALL,SCRIPTCOMMAND="$SCRIPT" deploy.sh
    fi
  done
else
  echo "Dont do more than 120 (or less than 0)"
fi
