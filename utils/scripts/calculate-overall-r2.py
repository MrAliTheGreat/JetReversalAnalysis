import numpy as np
import argparse

parser = argparse.ArgumentParser(description = "Log file information")
parser.add_argument("--log-path", required = True, help = "Log file path")
parser.add_argument("--mode", required = True, help = "Reversal or Stream")
args = parser.parse_args()

values = []
with open(args.log_path, "r") as f:
    for line in f:
        if(f"Val {args.mode} R2" in line):
            parts = line.strip().split(": ")
            values.append(float(parts[-1]))

values = np.array(values)

print(f"Mean: {values.mean()}")
print(f"STD: {values.std()}")
