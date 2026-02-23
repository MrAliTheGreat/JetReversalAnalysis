#!/bin/bash

# Parameters
# $1: log file path

sum=$(cat $1 | grep "Val Stream R2" | cut -d ":" -f 3 | paste -sd+ | bc)
echo "scale=10; $sum / $(cat $1 | grep 'Val Stream R2' | wc -l)" | bc
