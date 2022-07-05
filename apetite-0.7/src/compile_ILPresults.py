#!/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
from graph_compare import compile_ILPresults

if __name__=="__main__":
    try:
        if len(sys.argv)>2:
            compile_ILPresults(sys.argv[1],suffix=sys.argv[2])
        else:
            compile_ILPresults(sys.argv[1])
    except:
        print "usage : compile_ILPresults.py directory [solutions file suffix]"
