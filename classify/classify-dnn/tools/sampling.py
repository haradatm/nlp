#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('filename')
parser.add_argument('lines')
args = parser.parse_args()

filename = args.filename
lines = int(args.lines)
#print "file  = %s" % filename
#print "lines = %d" % lines

f = open(filename, "r")
l = f.readlines()
s = random.sample(l, lines)
print("".join(s))
