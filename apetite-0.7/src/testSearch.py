#!/bin/env python
# -*- coding: iso-8859-1 -*-
#
"""
tests de performances des recherche de graphe coherents
soit par methodes locales, algo greedy, ou a*


todo:
    - tester baseline ds l'ordre du texte
    - prevoir exp�s crossvalidation sur tous les folds
    - update des procedures d'evaluation (eval_old/Evaluation)
    - linker changed: test what is broken
"""

from TimeMLDocument import Document
from model import MegamClassifier
from optimisation import *
from linker import Linker, BaseLinker, LocalSearchLinker
from evaluation_old import ResultSink
from random import shuffle
from graph_compare import generic_measure, arc_type

def select_pairs(doc,method="gold",extend=True):
    """
    various methods to select attachemnts of event-pairs to test
    global temporal graph

    extend: true if t-t/e-t pairs are added to the selection
    
    method:
    - gold: stick to the oracle ... select the pairs from the gold annotation
    - successive: only pairs from adjacent sentences
    - random: select n event pairs, where n is nb if event instances in the document
    """
    result=[]
    if extend:
        result.extend(doc.get_all_time_time_relations().keys()+doc.get_event_time_relations().keys())
    if method=="gold":
        result.extend( doc.get_event_event_relations().keys())
    elif method=="random":
        le1=[x for x in doc.event_instances]
        shuffle(le1)
        le2=[x for x in doc.event_instances]
        shuffle(le2)
        result.extend( zip(le1,le2))
    elif method=="successive":
        evts=[(doc.event_instances[x].event.extent,x) for x in doc.event_instances]
        evts.sort()
        evts=[eid for (extent,eid) in evts]
        result.extend( zip(evts,evts[1:]))
    else:
        print >> sys.stderr, "warning, unknown method for event pair selection"
    return result
    
    
def alarmHandler(signum, frame):
    raise TimeOutError("astar timed out")



if __name__=="__main__":
    # exemple test (local classif est coherente)
    #python testSearch.py ../data/folds/0/test/wsj_1011_orig.tml.xml ../data/models/0.megam
    # exemple ou local classif est incoherente
    
    import sys, time, glob, os.path, signal
    import optparse
    from ilp import TimeOutError
    
    usage = "usage: %prog [options] [file]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-i", "--iterations", type=int, default=1, help="number of local search iterations (default 1)")
    parser.add_option("-m", "--pair_selection_method", dest="select_method",type=str, default="gold",
                      help="pair selection methods (default is oracle : stick to gold standard)")
    parser.add_option("-p", "--point-graph", dest="point", default=False, action="store_true",
                      help="do everything on interval endpoints instead o intervals (not implemented yet)")
    
    parser.add_option("-c", "--ignore-consistent", dest="ignore_consistent", default=False, action="store_true",
                      help="only test texts for which the baseline is inconsistent (imply method=gold)")
    parser.add_option("-f", "--fold", dest="fold", type=int,default=5,
                      help="use OTC fold number fold")
    parser.add_option("-s", "--saturation_method", dest="saturation_method",type=str, default="default",
                      help="method used for saturation (not implem)")
    parser.add_option("-d", "--directory", dest="directory",type=str, default=None,
                      help="directory on which to launch script (every *tml.xml file")
    parser.add_option("-l", "--classifier-model", dest="model",type=str, default=None,
                      help="model used for classification")



    (options, args) = parser.parse_args()

    FOLD=options.fold
    
    if len(args)>0:
        files=[args[0]]
    elif options.directory:
        files=glob.glob('%s/*.tml.xml'%options.directory)
        modelfile=options.model
    else:
        files=glob.glob('../data/folds_OTC/%d/test/*.tml.xml'%FOLD)
        modelfile="../data/models_OTC/%d.megam"%FOLD

    classifier=MegamClassifier(paramfile=modelfile)
    print "file\tgold\tbcons\tbaseacc\tacc"
    total_base=0
    total_heur=0
    total_rel=0
    for onefile in files:
        print os.path.basename(onefile),
        doc=Document(onefile)
        gold=doc.get_event_event_relations()
        tt=doc.get_all_time_time_relations()
        et=doc.get_event_time_relations()
        rel_nb=len(gold)
        total_rel+=rel_nb
        print "%d"%rel_nb,
        # Local search
        #print >> sys.stderr, "starting local search"

        #filter=set(gold.keys()+tt.keys()+et.keys())
        filter=set(select_pairs(doc,method=options.select_method))

        # Baseline
        # TODO (FIXME): linker now init without doc
        #
        base_linker = BaseLinker(doc,classifier)
        base_linker.local_probabilities.update(tt)
        base_linker.local_probabilities.update(et)
        
        base_sink = ResultSink()
        base_linker.link()
        base_predictions=base_linker.get_predictions()
        base_sink.update( base_predictions, gold )
        #filter=set(gold.keys()+tt.keys()+et.keys())
        gb=base_linker.graph_from_predictions(filter=filter)
        consistent_base=gb.saturate()

        # Local if necessary
        if consistent_base and options.ignore_consistent and options.select_method=="gold":
            score=base_sink.accuracy()
            print >> sys.stderr, "baseline consistent : stick to it"
            print "True",score,score
            total_base += score*rel_nb
        else: 
            local=LocalSearchLinker(doc,classifier,mode="greedy",filter_relations=filter)
            local.local_probabilities.update(tt)
            local.local_probabilities.update(et)
            t0=time.time()
            #local.link(maxIter=10,startmode="astar")
            try:
                # time out = 5mn
                #signal.signal(signal.SIGALRM, alarmHandler)
                #signal.alarm(5*60)
                local.link(maxIter=options.iterations,startmode="bf")
            except AttributeError:
                print >> sys.stderr, "no solution found"
            except TimeOutError:
                 print >> sys.stderr, "init timeout"
            #signal.alarm(0)
            t1=time.time()
            print >> sys.stderr, "time :%fs"%(t1-t0)

            predictions=local.get_predictions()
            gl=local.graph_from_predictions(filter=filter)
            consistent_local=gl.saturate()
            sink = ResultSink() 
            sink.update( predictions, gold )
            score_base,score_heur=base_sink.accuracy(),sink.accuracy()
            total_base += score_base*rel_nb
            total_heur += score_heur*rel_nb
            print consistent_base, score_base,score_heur
            #"/ local opt consistent:", consistent_local
            #print "---------------"
        
        # autres mesures
        # 1) relations simples sur graphes satures
        if options.select_method!="gold":
            # not tested
            doc.relations.update(tt)
            #
            gold_graph=doc.make_graph(doc.relations)
            gold_graph.saturate()
            g1=gold_graph
            g2=gl
            print os.path.basename(onefile),
            print "PREC_EE:", generic_measure(g1,g2,g1.algebra(),
                                   filtreH=(lambda x: False),
                                   filtreS=(lambda x: arc_type(x,"event","event") and (len(x.relation())==1))),
            print "RECALL_EE:", generic_measure(g1,g2,g1.algebra(),
                                   filtreS=(lambda x: False),
                                   filtreH=(lambda x: arc_type(x,"event","event") and (len(x.relation())==1))),
            g1=gold_graph
            g2=gb # ou bien gb=base_linker.graph_from_predictions() pour faire chier la baseline
            print "PREC-baseline-EE:", generic_measure(g1,g2,g1.algebra(),
                                   filtreH=(lambda x: False),
                                   filtreS=(lambda x: arc_type(x,"event","event") and (len(x.relation())==1))),
            print "RECALL-baseline-EE:", generic_measure(g1,g2,g1.algebra(),
                                   filtreS=(lambda x: False),
                                   filtreH=(lambda x: arc_type(x,"event","event") and (len(x.relation())==1)))
    print "total, base,init consistent):",total_rel,total_base,total_heur

