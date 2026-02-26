#!/bin/bash

# Execute from main folder
# Check all num_windows when providing different window lengths!


python utils/scripts/split-dataset.py --input-dataset /mnt/abahari/stream_dataset_100.csv
python utils/scripts/split-dataset.py --input-dataset /mnt/abahari/reversals_dataset_pos2neg_10000.csv
python utils/scripts/split-dataset.py --input-dataset /mnt/abahari/reversals_dataset_neg2pos_10000.csv
echo "+ Split stream, reversal pos2neg, and reversal neg2pos csv datasets"

##### STATS #####
# 147000 windows each 100 50 5
# 54000 windows each 2 50 5
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_pos2neg_10000-stats.csv --output-zarr /mnt/abahari/reversals_dataset_pos2neg-stats.zarr --num-windows 54000 --num-timesteps 1000 --mode reversal --normalize n
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_neg2pos_10000-stats.csv --output-zarr /mnt/abahari/reversals_dataset_neg2pos-stats.zarr --num-windows 54000 --num-timesteps 1000 --mode reversal --normalize n
echo "+ Generated windows for pos2neg and neg2pos reversal stats datasets and saved to Zarr files"

python utils/scripts/merge-zarr-in-ram.py --datasets /mnt/abahari/reversals_dataset_neg2pos-stats.zarr/ /mnt/abahari/reversals_dataset_pos2neg-stats.zarr/ --output-zarr /mnt/abahari/reversals_dataset-stats.zarr
echo "+ Merged pos2neg and neg2pos reversal stats datasets"

rm -r /mnt/abahari/reversals_dataset_neg2pos-stats.zarr/ /mnt/abahari/reversals_dataset_pos2neg-stats.zarr/
echo "+ Removed pos2neg and neg2pos reversal stats datasets"

# 294000 windows 100 50 5
# 108000 windows 2 50 5
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/stream_dataset_100-stats.csv --output-zarr /mnt/abahari/stream_dataset_100-stats.zarr --num-windows 108000 --num-timesteps 100000 --mode stream --normalize n
echo "+ Generated windows for stream stats dataset and saved to Zarr file"

python utils/scripts/merge-zarr-in-ram.py --datasets /mnt/abahari/reversals_dataset-stats.zarr/ /mnt/abahari/stream_dataset_100-stats.zarr/ --output-zarr /mnt/abahari/dataset-stats.zarr
echo "+ Merged stream and reversal stats Zarr datasets"

# Stats Zarr dataset ready for calculating mean and std!
rm -r /mnt/abahari/reversals_dataset-stats.zarr/ /mnt/abahari/stream_dataset_100-stats.zarr/
echo "+ Removed stream and reversal stats Zarr datasets"

python utils/scripts/get-stats-windowed.py --input-dataset /mnt/abahari/dataset-stats.zarr --output-csv /mnt/abahari/stats5050.csv
echo "+ Created stats"

rm -r /mnt/abahari/dataset-stats.zarr
echo "+ Removed stats dataset"

rm /mnt/abahari/reversals_dataset_pos2neg_10000-stats.csv /mnt/abahari/reversals_dataset_neg2pos_10000-stats.csv /mnt/abahari/stream_dataset_100-stats.csv
echo "+ Clean up stats csv files"
#################



##### TRAIN #####
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_pos2neg_10000-train.csv --output-zarr /mnt/abahari/reversals_dataset_pos2neg-train.zarr --num-windows 39000 --num-timesteps 1000 --mode reversal --normalize y
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_neg2pos_10000-train.csv --output-zarr /mnt/abahari/reversals_dataset_neg2pos-train.zarr --num-windows 39000 --num-timesteps 1000 --mode reversal --normalize y
echo "+ Generated windows for pos2neg and neg2pos reversal train datasets and saved to Zarr files"

python utils/scripts/merge-zarr-in-ram.py --datasets /mnt/abahari/reversals_dataset_neg2pos-train.zarr/ /mnt/abahari/reversals_dataset_pos2neg-train.zarr/ --output-zarr /mnt/abahari/reversals_dataset-train.zarr
echo "+ Merged pos2neg and neg2pos reversal train datasets"

rm -r /mnt/abahari/reversals_dataset_neg2pos-train.zarr/ /mnt/abahari/reversals_dataset_pos2neg-train.zarr/
echo "+ Removed pos2neg and neg2pos reversal train datasets"

python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/stream_dataset_100-train.csv --output-zarr /mnt/abahari/stream_dataset_100-train.zarr --num-windows 78000 --num-timesteps 100000 --mode stream --normalize y
echo "+ Generated windows for stream train dataset and saved to Zarr file"

python utils/scripts/merge-zarr-in-ram.py --datasets /mnt/abahari/reversals_dataset-train.zarr/ /mnt/abahari/stream_dataset_100-train.zarr/ --output-zarr /mnt/abahari/dataset_5050-train.zarr
echo "+ Merged stream and reversal train Zarr datasets"

rm -r /mnt/abahari/reversals_dataset-train.zarr/ /mnt/abahari/stream_dataset_100-train.zarr/
echo "+ Removed stream and reversal train Zarr datasets"

rm /mnt/abahari/reversals_dataset_pos2neg_10000-train.csv /mnt/abahari/reversals_dataset_neg2pos_10000-train.csv /mnt/abahari/stream_dataset_100-train.csv
echo "+ Clean up train csv files"
#################



##### VALIDATION #####
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_pos2neg_10000-val.csv --output-zarr /mnt/abahari/reversals_dataset_pos2neg-val.zarr --num-windows 10000 --num-timesteps 1000 --mode reversal --normalize y
python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_neg2pos_10000-val.csv --output-zarr /mnt/abahari/reversals_dataset_neg2pos-val.zarr --num-windows 10000 --num-timesteps 1000 --mode reversal --normalize y
echo "+ Generated windows for pos2neg and neg2pos reversal validation datasets and saved to Zarr files"

python utils/scripts/merge-zarr-in-ram.py --datasets /mnt/abahari/reversals_dataset_neg2pos-val.zarr/ /mnt/abahari/reversals_dataset_pos2neg-val.zarr/ --output-zarr /mnt/abahari/dataset-nearReversal-val.zarr
echo "+ Merged pos2neg and neg2pos reversal validation datasets"

rm -r /mnt/abahari/reversals_dataset_neg2pos-val.zarr/ /mnt/abahari/reversals_dataset_pos2neg-val.zarr/
echo "+ Removed pos2neg and neg2pos reversal validation datasets"

python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/stream_dataset_100-val.csv --output-zarr /mnt/abahari/dataset-stream-val.zarr --num-windows 20000 --num-timesteps 100000 --mode stream --normalize y
echo "+ Generated windows for stream train dataset and saved to Zarr file"

rm /mnt/abahari/reversals_dataset_pos2neg_10000-val.csv /mnt/abahari/reversals_dataset_neg2pos_10000-val.csv /mnt/abahari/stream_dataset_100-val.csv
echo "+ Clean up validation csv files"
######################



##### TEST #####
# Used in inference for visualization
utils/scripts/merge-csv.sh /mnt/abahari/reversals_dataset_pos2neg_10000-test.csv /mnt/abahari/reversals_dataset_neg2pos_10000-test.csv /mnt/abahari/reversals_dataset_10000-test.csv
echo "+ Merged pos2neg and neg2pos reversal test csv datasets"

python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/reversals_dataset_10000-test.csv --output-zarr /mnt/abahari/dataset-reversal-windowed-test.zarr --num-windows -1 --num-timesteps 1000 --mode visualization --normalize y
python utils/scripts/generate-full-time-series.py --input-dataset /mnt/abahari/reversals_dataset_10000-test.csv --output-zarr /mnt/abahari/dataset-reversal-full-test.zarr --num-timesteps 1000
echo "+ Generated windowed and full time-series reversal test dataset"

python utils/scripts/generate-windowed-dataset.py --input-dataset /mnt/abahari/stream_dataset_100-test.csv --output-zarr /mnt/abahari/dataset-stream-windowed-test.zarr --num-windows -1 --num-timesteps 100000 --mode visualization --normalize y
python utils/scripts/generate-full-time-series.py --input-dataset /mnt/abahari/stream_dataset_100-test.csv --output-zarr /mnt/abahari/dataset-stream-full-test.zarr --num-timesteps 100000
echo "+ Generated windowed and full time-series stream test dataset"

rm /mnt/abahari/reversals_dataset_pos2neg_10000-test.csv /mnt/abahari/reversals_dataset_neg2pos_10000-test.csv /mnt/abahari/reversals_dataset_10000-test.csv /mnt/abahari/stream_dataset_100-test.csv
echo "+ Clean up test csv files"
################


