#!/usr/bin/env bash

while getopts ":p:" opt; do
  case $opt in
    p) SCRIPT="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if (($1 >= 0 && $1 <= 120)); then
  for ((i = 0 ; i < $1 ; i++)); do
    sbatch --export=ALL,SCRIPTCOMMAND="$SCRIPT" deploy.sh
  done
else
  echo "Dont do more than 120 (or less than 0)"
fi
