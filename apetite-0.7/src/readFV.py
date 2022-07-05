#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

"""reads and converts feature values dumped by trainer, for a posteriori analysis with weka"""


test="rel_equals e1_str=pledge e1_class=I_ACTION e1_tns=NONE e1_asp=NONE e1_pol=POS e2_str=demand e2_class=I_ACTION e2_tns=NONE e2_asp=NONE e2_pol=POS class_pair=I_ACTION_&_I_ACTION tns_pair=NONE_&_NONE asp_pair=NONE_&_NONE pol_pair=POS_&_POS same_tns=True same_asp=True"


import sys
def readFV(filename):
    """reads and converts feature values dumped by trainer, for a posteriori analysis with weka"""
    res=[]
    for line in open(filename):
        line=line.strip()
        if line!="":
            all=line.split()
            #print '-------'
            #print line
            classValue=all[0]
            fv=[x.split("=")[1] for x in all[1:]]
            res.append((classValue,fv))
    return res

def writeFV(fvlist,file=sys.stdout):
    """dumps features in csv format"""
    output=[",".join(fv+[classValue]) for (classValue,fv) in fvlist]
    print >> file, "\n".join(output)
        
    
if __name__=="__main__":
    filein=sys.argv[1]
    try:
        fileout=open(sys.argv[2])
    except:
        fileout=sys.stdout
    writeFV(readFV(filein),fileout)
