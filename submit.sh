#!/usr/bin/env bash

if (($1 >= 0 && $1 <= 120)); then
  for ((i = 0 ; i < $1 ; i++)); do
    sbatch deploy.sh
  done
else
  echo "Dont do more than 120 (or less than 0)"
fi
