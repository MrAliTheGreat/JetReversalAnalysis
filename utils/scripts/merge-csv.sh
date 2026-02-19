#!/bin/bash

# Parameters:
# $1 and $2: CSVs targeted for merge
# $3: Output CSV 

(head -n 1 $1 && tail -n +2 $1 && tail -n +2 $2 | shuf) > $3