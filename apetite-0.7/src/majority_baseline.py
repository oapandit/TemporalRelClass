#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


'''

Script for computing majority relation accuracy

'''

import sys
from TimeMLDocument import Document
from TimeMLDocument import normalize_relation 


def baseline(timeml_dir, condense=False, file_ext='tml.xml'):
    predictions = []
    # create list of files
    docs = [os.path.join(timeml_dir,f) for f in os.listdir(timeml_dir)
            if f.endswith(file_ext) and not f.startswith('.')]
    # iterate through documents counting relations
    rel_ct = {}
    total = 0
    for (i,d) in enumerate(docs):
        os.write(1, "%s" %"\b"*len(str(i))+str(i)) 
        doc = Document( d )
        relations = doc.get_event_event_relations()
        for (n1,n2) in relations:
            rel = relations[(n1,n2)]
            if condense:
                (n1,n2), rel = normalize_relation((n1,n2),rel)
            rel_ct[rel] = rel_ct.get(rel,0) + 1
            total += 1
    os.write(1,' documents \n')
    # find most frequent relation
    most_freq_rel = None
    highest_ct = 0
    print "-"*37
    print "| %20s | %10s |" %("Relation", "Freq. ct.")
    print "-"*37
    for (rel,ct) in rel_ct.items():
        print "| %20s | %10s |" %(rel,ct)
        if ct > highest_ct:
            most_freq_rel = rel
            highest_ct = ct
    print "-"*37
    print "| %20s | %10s |" %("TOTAL",total)
    print "-"*37
    # compute acc for always picking most frequent
    acc = highest_ct/float(total)
    print "\nBaseline Acc (always predict '%s'): %s (%s/%s)" \
          %(most_freq_rel,round(acc,3),highest_ct,total)



if __name__ == '__main__':

    import os
    import optparse
    
    usage = "usage: %prog [options] TimeML directory (e.g., TimeBank1.1/docs)"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-c", "--condense_relations", action="store_true", default=False, help="use 6 instead of 13 relations")

    (options, args) = parser.parse_args()

    timeml_dir = args[0]
    baseline( timeml_dir, condense=options.condense_relations )
