#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
#
#
# TODO:
#   x- s�parer eval des mesures pour faire graphes directo dans tableur (au lieu de cette
#     bouillie)
#     test avec cut
#     cut -d" " -f1,3,...  autocomp.stats 
#   x- ajouter "pas" de degradation (pour pas tout faire)
#     ex: 10 "pas" va de 90% des relations a 0%
#   x- degrader mesures en changeant des relations au hasard
#   x- comparer en affaiblissnt l'annotation de d�part
#   - ajout DICE coeff ?
# TODO perf:
#    - replace deepcopy of random graphs with regeneration (seems faster with real timeml docs)
#    - do not copy nodes, only edges
# BUG:
#    solved?- prec & recall sur pts decale de 1 ou 2 unites sur l'expe "remove", ex sur wsj_0161

"""
Evaluations of different measures' behaviour
on a text or texts in a directory, compared to variations of annotations:

- standard vs various baselines
- standard vs "degraded" standard:
    * standard with relations removed
    * standard with relations changed

usage:
  - one file: auto_compare.py file.timeml 
  - all files:  auto_compare.py

  ex: g�n�ration des fichiers de comparaison (prend tous les xml du
 repertoire et genere les fichiers autocomp1 et 2)

 > python $PATHTO/autocompare.py 2> logerr > log

"""
import sys,os 
import re
import random
#apetite_path=os.environ.get("APETITEROOT",".")
#sys.path.append(apetite_path+"/Scripts")
import apetite

from apetite.TimeMLDocument import Document
import apetite.read_timeml as timeml
from apetite.graph_compare import make_stats, generic_measure, transform2pt, get_kernel, read_graphe, table_row_labels
#from apetite.graph_compare import *

from apetite.graph.Graph import allen_algebra, Relation
from apetite.utils import make_graph
import copy
#from timeml_compare import  normalise, make_graph

    
# quelques variantes sur un meme graphe pour
# tester la robustesse des mesures
# en enlevant n relations sur le graphe non satur�
# g0 est le graphe satur� r�f�rent, g est le non-satur�
# pb: le non-satur� est ensuite satur� ...
# TODO: une vraie copie
def variantes(g0,g,n=1,g_noyau=None,gpt0=None,gmin0=None):
    #from pickle import dumps, loads
    #saved=dumps(g)
    edges=[x for x in g.edges().values() if x.node1()<x.node2()]# if arc_type(x,"event","event")]
    print >> sys.stderr, "nb edges before saturation", len(edges)
    list_e=random.sample(edges,min(n,len(edges)))
    for x in list_e:
        g.del_edge(x)
    g.saturate()
    if gpt0 is None:
        gpt0,gmin0=transform2pt(g0)
        print >> sys.stderr, "gpt computed again?"
    if g_noyau is None:
        g_noyau=get_kernel(g0)
        print >> sys.stderr, "kernel computed again?"
    res=make_stats(g0,g,g_noyau=g_noyau,gpt1=gpt0,gmin1=gmin0)
    #g=loads(saved)
    return res


# quelques variantes sur un meme graphe pour
# tester la robustesse des mesures
# en changeant n relations sur le graphe non satur�
# g0 est le graphe satur� r�f�rent, g est le non-satur�
# 
def disturb_variantes(g0,g,n=1,g_noyau=None,gpt0=None,gmin0=None):
    #from pickle import dumps, loads
    #saved=dumps(g)
    edges=[x for x in g.edges().values() if x.node1()<x.node2()]# if arc_type(x,"event","event")]
    print >> sys.stderr, "nb edges before saturation", len(edges)
    consistent=False
    tries=10
    while not(consistent) and tries>0:
        list_e=random.sample(edges,min(n,len(edges)))
        for x in list_e:
            newrel=random.sample(["overlaps","overlapsi",
                                  "before","after","includes","is_included"],1)[0]
            newrel=Relation(newrel)
            g.change_edge(x,newrel)
        consistent=g.saturate()
        tries=tries-1
    if gpt0 is None:
        gpt0,gmin0=transform2pt(g0)
        print >> sys.stderr, "gpt computed again?"
    if g_noyau is None:
        g_noyau=get_kernel(g0)
        print >> sys.stderr, "kernel computed again?"
    res=make_stats(g0,g,g_noyau=g_noyau,gpt1=gpt0,gmin1=gmin0)
    #g=loads(saved)
    return res


# auto-compare annotation with same annotation degraded by removing one relation
# at random at a time from the originally annotated set
# repeat <sample> times the operation to average the result

# results -> cf. timeml_compare/make_stats
# TODO:
# done  - make steps every 10% of relations for graph>10
#         pour moins de 10 on garde tout
def remove_relations(g1,data,sample=5,outfile="autocomp.stats",steps=10,g1_noyau=None,gpt1=None,g1min=None):
    # g1bis must be at all time the unsaturated copy of the original annotation
    # or everything is meaningless
    g1bis=clone_graphe(g1,data) 
    # don't assume a fixed set of measures
    total=None
    cmpstats=open(outfile,"w")
    cmpstats.write(" ".join(table_row_labels)+"\n")
    edge_number=len(g1bis.edges())/2
    edge_nb_sature=len(g1.edges())/2
    
    for k in range(edge_number+1)[::(edge_number/min(edge_number,steps))]:
        total=None
        for n in range(sample):
            # g1bis must be at all time the unsaturated copy of the original annotation
            g1bis=clone_graphe(g1,data) 
            print "removed: %d, total edge nb (sat.): %d,  annotation edge  nb : %d"%(k,edge_nb_sature, edge_number)
            results = variantes(g1,g1bis,k,g_noyau=g1_noyau,gpt0=gpt1,gmin0=g1min)
            if total==None:
                total=[(0.,0.) for x in results]
            total=[(x1+y1,x2+y2) for ((x1,x2),(y1,y2)) in zip(total,results)]
        total=[(x1/sample,x2/sample) for x1,x2 in total]
        stats_tmp=[]
        for (x,y) in total:
            if y==0:
                stats_tmp.append((x,y,0))
            else:
                stats_tmp.append((x,y,float(x)/y))
        cmpstats.write(" ".join(["%f %f %f"%x for x in stats_tmp])+"\n")
    cmpstats.close()


# regenerate the relations from the initial document,
# to go back from saturations.
# if data is empty, then copy the graph itself. (random graphs: is this safe ?)
def clone_graphe(g1,data):
    if data=={}:
        g1bis=copy.deepcopy(g1)
    else:
        #data.graph=data.make_graph(data.relations)
        g1bis=make_graph(data.get_relations(),algebra=allen_algebra,index=False)
    return g1bis
# auto-compare annotation with same annotation degraded by changing one relation
# at random at a time from the originally annotated set
# repeat <sample> times the operation to average the result
# TODO:
#   - annotation must remain consistent
#     -> disturb until inconsistent
def disturb_relations(g1,data,sample=5,outfile="autocomp.stats",steps=10,g1_noyau=None,gpt1=None,g1min=None):
    g1bis=clone_graphe(g1,data)   
    # don't assume a fixed set of measures
    total=None
    cmpstats=open(outfile,"w")
    cmpstats.write(" ".join(table_row_labels)+"\n")
    edge_number=len(g1bis.edges())/2
    # on enleve par paquet de 10%
    for k in range(edge_number+1)[::edge_number/min(edge_number,steps)]:
        total=None
        for n in range(sample):
            g1bis=clone_graphe(g1,data)   
            print "disturbed: %d, total edge nb (sat.): %d, annotation edge nb : %d"%(k,len(g1.edges())/2, len(g1bis.edges())/2)
            results = disturb_variantes(g1,g1bis,k,g_noyau=g1_noyau,gpt0=gpt1,gmin0=g1min)
            if total==None:
                total=[(0.,0.) for x in results]
            total=[(x1+y1,x2+y2) for ((x1,x2),(y1,y2)) in zip(total,results)]
        total=[(x1/sample,x2/sample) for x1,x2 in total]
        stats_tmp=[]
        for (x,y) in total:
            if y==0:
                stats_tmp.append((x,y,0))
            else:
                stats_tmp.append((x,y,float(x)/y))
        cmpstats.write(" ".join(["%f %f %f"%x for x in stats_tmp])+"\n")
    cmpstats.close()



if __name__=="__main__":

    import os, glob, time
    import os.path
    #import psyco
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-r", "--random",
                      dest="random",action="store_true",default=False,
                      help=" expe on random graphs")   
    parser.add_option("-d", "--disturb",
                      action="store_true",default=False,
                      help="do also disturb experiment (default: do only remove)")
    parser.add_option("-n", "--nb-sample",
                      dest="sample",type="int",default=1,
                      help="do each experiment point n times and average (default 1)")
    parser.add_option("-s", "--scratch",
                      dest="redo",action="store_true",default=False,
                      help="redo everything from scracth (default: dont redo existing computations")
    parser.add_option("-t", "--time_time", action="store_true",
                      default=False, help="use Time-Time and Event-Time reference relations (default: False)")
    #parser.add_option("-b", "--bethard_relations", action="store_true", 
    #                  default=False, help="use additional relations from Bethard (default: False)")
    
    (options, args) = parser.parse_args()
    #psyco.full()
    # whether to start from scratch or pick up at the last problem file
    REDO=options.redo
    SAMPLE=options.sample
    
    #bethard_relations = {}
    #if options.bethard_relations:
    #    bethard_relations = extract_bethard_relations()



    t0=time.clock()
    newt=t0
    if len(args)==0:# no argument, so take all timeml files in current dir
        if options.random:
            list_filenames=glob.glob("random*")
            list_filenames=[x for x in list_filenames if not(x.endswith(".csv"))]
        else:
            list_filenames=glob.glob("*.tml.xml")
            list_filenames.extend(glob.glob("*.tml"))
        
    # or, argument= do it on a file or set of files
    else:
        list_filenames=glob.glob(args[0])
        print >> sys.stderr, "targets:", list_filenames
    for filename in list_filenames:
        if os.path.isfile(filename+".autocomp1.csv") and not(REDO):
            print >> sys.stderr, "not redoing file",filename
        else:
            print >> sys.stderr, "doing file %s "%filename,
            #base1=open(filename).read()
            #try:
            if True:
                if options.random:
                    g1=read_graphe(filename,allen_algebra)
                    data1bis={}
                else:
                    doc=Document(filename)
                    g1=doc.get_graph(isodate=options.time_time)
                    data1bis=doc
                    #base1=open(filename).read()
                    #data1,t1=timeml.extract_from_string(base1)
                    #data1bis=normalise(data1)
                    #data1bis=data1
                    #g1=make_graph(data1bis)
                print >> sys.stderr, "with %d relations"%(0.5*len(g1.edges()))
                if options.random:
                    human_coherent=True
                else:
                    human_coherent=g1.saturate()
                if not(human_coherent):
                    print "incoherence in the human annotation"
                else:
                    gpt1,g1min=transform2pt(g1)
                    g1_noyau=get_kernel(g1)
                    #beurk
                    remove_relations(g1,data1bis,g1_noyau=g1_noyau,gpt1=gpt1,g1min=g1min,sample=SAMPLE,outfile=filename+".autocomp1.csv")
                    if options.disturb:
                        disturb_relations(g1,data1bis,g1_noyau=g1_noyau,gpt1=gpt1,g1min=g1min,sample=SAMPLE,outfile=filename+".autocomp2.csv")
                delta=time.clock()-newt
                newt=time.clock()
                print >> sys.stderr, "file %s done, with %d relations in %fs"%(filename,0.5*len(g1.edges()),delta)
            #except:
            else:    
                print >> sys.stderr, "pb reading file ", filename

# test noyau du graphe (obsolete)
# OK: et on peut voir que g2 pas les m�mes relations que annotation (g1=make_graph(data1bis))
"""
sys.argv.append("../Tests/wsj_0006_orig.tml.xml")
base1=open(sys.argv[1]).read()
data1,t1=timeml.extract_from_string(base1)
data1bis=normalise(data1)
g1=make_graph(data1bis)
human_coherent=g1.saturate()
from graph_compare import get_kernel
g2=get_kernel(g1)
g2.saturate()
from timeml_compare import make_stats
make_stats(g1,g2)
"""
#result
#FINAllen, finesse allen: 4.384615/9.000000
# COHAllen, coherence allen: 4.000000/4.000000
# FINAR, finesse annotation relations: 4.714286/9.000000
# COHAR, coherence annotation relations: 4.000000/4.000000
# PREC, prec. single ann. relations: 4.000000/4.000000
# RECA, recall single ann. relations: 4.000000/9.000000
# PRECG, prec. single ann. relations: 4.000000/4.000000
# RECAG, recall single ann. relations: 4.714286/9.000000
# JAQ_AR, jaquard sur annotation relations: 4.714286/9.000000
# FINAREE, finesse event-event annotation relations: 4.714286/9.000000
# COHAREE, coherence event-event annotation relations: 9.000000/9.000000
# COMPATAR, : 9.000000/9.000000
# PRECEE, event-event prec. single ann. relations: 2.000000/2.000000
# RECAEE, event-event recall single ann. relations: 2.000000/7.000000
#[(0.69230769230769229, 9.0), (0, 0), (1.2857142857142854, 9.0), (0, 0), (0, 0), (0.0, 9.0), (0, 0), (1.2857142857142854, 9.0), (1.2857142857142854, 9.0), (1.2857142857142854, 9.0), (7.0, 7.0), (9.0, 9.0), (0, 0), (0.0, 7.0)]
