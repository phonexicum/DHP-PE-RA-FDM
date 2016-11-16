#! /usr/bin/python

import sys
from os import listdir, path
import json
import math

if len(sys.argv) < 2:
    print ("Usage: ./generate_gnuplot.py output_dir")
    sys.exit(1)

target_dir = path.normpath(sys.argv[1])
target_files = [path.join(target_dir, f) for f in listdir(target_dir) if path.isfile(path.join(target_dir, f))]

data = []

for file in target_files:
    with open(file, "r") as data_file:
        print (file)
        data.extend(json.load(data_file))


def srtFunc(x, y):

    if x["x"] < y["x"]:
        return 1
    elif x["x"] == y["x"]:
        if x["y"] < y["y"]:
            return 1
        elif x["y"] == y["y"]:
            return 0
        else:
            return -1
    else:
        return -1


data.sort(cmp=srtFunc, reverse=False)
# sorted(data, key=itemgetter("x", "y"))

maximum = 0.0
minimum = 1000000000.0

with open(target_dir + ".dat", "w") as gnuplot_file:
    gnuplot_file.write("# x y u\n")

    prev_value_p = -1
    for value in data:
        if prev_value_p != value["x"]:
            prev_value_p = value["x"]
            gnuplot_file.write("\n")

        if value["x"] > 0.0 and value["y"] > 0.0:

            # Real value
            val = value["u"]
            #
            # Absolute error
            # val = value["u"] - math.log(1 + value['x'] * value['y'])
            #
            # Relative error
            # val = abs(value["u"] - math.log(1 + value['x'] * value['y'])) / value["u"]

            maximum = max(maximum, val)
            minimum = min(minimum, val)

            gnuplot_file.write(str(value["x"]) + " " + str(value["y"]) + " " + str(val) + "\n")

print ""
print maximum
print minimum
