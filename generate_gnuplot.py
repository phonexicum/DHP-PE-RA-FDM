#! /usr/bin/python

import sys
from os import listdir, path
import json

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

with open(target_dir + ".dat", "w") as gnuplot_file:
    gnuplot_file.write("# x y u\n")

    prev_value_p = -1
    for value in data:
        if prev_value_p != value["x"]:
            prev_value_p = value["x"]
            gnuplot_file.write("\n")

        gnuplot_file.write(str(value["x"]) + " " + str(value["y"]) + " " + str(value["u"]) + "\n")
