#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

'''

Simple script for checking that output graphs are consistent or not. The input format is:

[trel_before] EI2155 EI2156
[trel_before] EI2161 EI2160
[trel_equals] EI2165 EI2166
[trel_beforei] EI2149 EI2148
[trel_equals] EI2150 EI2152
[trel_before] EI2155 EI2151
[trel_beforei] EI2165 EI2164
[trel_equals] EI2161 EI2162
[trel_before] EI2150 EI2153
[trel_before] EI2150 EI2151
[trel_before] EI2148 EI2154
[trel_beforei] EI2160 EI2159
[trel_beforei] EI2154 EI2153
[trel_before] EI2155 EI2153
[trel_equals] EI2168 EI2170
[trel_equals] EI2163 EI2162
[trel_equals] EI2158 EI2159
[trel_beforei] EI2163 EI2164
[trel_equals] EI2170 EI2171



'''

import sys
import os
import optparse
from apetite.graph.Graph import AllenGraph, Edge
from apetite.graph.Relation import Relation

usage = "usage: %prog [options] dir"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-x", "--file_ext", type=str, default='.pred', help="file extension for file selection (default: '.pred')")
parser.add_option("-f", "--filter", action="store_true", default=False, help="filter out relations that are not in the gold")
(options, args) = parser.parse_args()

result_dir = args[0]    
pred_files = dict([(f[:-len(options.file_ext)],os.path.join( result_dir,f )) for f in os.listdir(result_dir)
                   if f.endswith(options.file_ext)])
gold_files = dict([(f[:-5],os.path.join( result_dir,f )) for f in os.listdir(result_dir)
                   if f.endswith(".gold")])

# if set(pred_files.keys()) <> set(gold_files.keys()):
#     sys.exit("Error: %s predictions files vs. %s gold files" %(len(pred_files),len(gold_files)))

print "%s files with extension %s" %(len(pred_files),options.file_ext)


print "%35s | %5s | %5s" %("Document","Cons.","#events")
ct = 0
for f in pred_files:
    # process predictions
    predictions = {}
    for l in file(pred_files[f]):
        l = l.strip()
        rel,e1,e2 = l.split()
        predictions[(e1,e2)] = rel
    # process gold
    if options.filter:
        gold = {}
        for l in file(gold_files[f]):
            l = l.strip()
            e1,e2,rel = l.split()
            gold[(e1,e2)] = rel
    # process predictions graph
    pg = AllenGraph()
    events = {}
    for (e1,e2) in predictions:
        if options.filter:
            if not (e1,e2) in gold:
                continue
        pg.addNode(e1)
        pg.addNode(e2)
        rel = predictions[e1,e2]
        pg.addEdge( Edge(e1,e2,Relation(rel)) )
        events[e1] = 1
        events[e2] = 1
    cons = 0
    if pg.saturate():
        cons = 1
        ct += 1
    print "%35s | %5s | %5s" %(f,cons,len(events))

print "%s/%s (%s) consistent graphs" %(ct,len(pred_files),ct/float(len(pred_files)))

