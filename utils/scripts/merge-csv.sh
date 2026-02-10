#!/bin/bash

(head -n 1 /mnt/abahari/reversals_dataset_pos2neg_10000-stats.csv && tail -n +2 /mnt/abahari/reversals_dataset_pos2neg_10000-stats.csv && tail -n +2 /mnt/abahari/reversals_dataset_neg2pos_10000-stats.csv | shuf) > /mnt/abahari/reversals_dataset_20000-stats.csv