#!/usr/bin/env python3

import argparse
from random import randint

parser = argparse.ArgumentParser(
    description="Split file into test and train files. Names of created files are: test_{filename} and train_{filename} \n\n CAUTION! \nUse only for small files (max. 10MB)"
)
parser.add_argument(
    "--test_size", default=0.2, help="Part of file chosen as test data. [0-1]"
)
parser.add_argument("filename", help="File containing data")
args = parser.parse_args()

if args.test_size < 0.0 or args.test_size > 1.0:
    print("Wrong test size")
    exit()

f_test = open("test_" + args.filename, "w")
f_train = open("train_" + args.filename, "w")

with open(args.filename) as fin:
    lines = fin.readlines()
    test_size = int(len(lines) * args.test_size)
    for i in range(test_size):
        f_test.write(lines.pop(randint(0, len(lines) - 1)))
    for line in lines:
        f_train.write(line)

f_test.close()
f_train.close()
