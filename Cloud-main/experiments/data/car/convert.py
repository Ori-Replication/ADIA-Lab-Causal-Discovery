#!/usr/bin/env python

import csv

si = 0
li = 1
di = 2
hi = 3

fp = open("car.dat", "w")

with open("car.data", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")

    next_alphabet = 1
    val_to_alphabet = dict()

    for row in reader:
        fimi_row = []
        for val in row:
            if val_to_alphabet.has_key(val):
                alphabet = val_to_alphabet[val]
            else:
                alphabet = next_alphabet
                val_to_alphabet[val] = next_alphabet
                next_alphabet += 1
            fimi_row.append(alphabet)
        fimi_row_str = " ".join(str(v) for v in fimi_row)
        fp.write(fimi_row_str + "\n")

fp.close()
