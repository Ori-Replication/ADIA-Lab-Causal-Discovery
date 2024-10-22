#!/usr/bin/env python

import csv

si = 0
li = 1
di = 2
hi = 3

fp = open("abalone.dat", "w")

with open("abalone.data", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")

    for row in reader:
        sex, length, diameter, height = row[si], row[li], row[di], row[hi]
        if sex == "M":
            sex = 1
        elif sex == "F":
            sex = 2
        else:
            sex = 3
        fp.write("%d %s %s %s\n" % (sex, length, diameter, height))

fp.close()

