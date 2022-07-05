#!/bin/env python
# -*- coding: iso-8859-1 -*-
""" applying local optimisation to temporal graph building


remaining problem:
       ok not instantiating an edge (when graph becomes inconsistent) gives probability one that edge, while fully instantiated graph will
       have lower score on this one

       ? score with lowest probability in that case or specificity of relation (1/13 for allen)
       move smoother with conceptual neighborhood


TODO:
   - should work not ONLY on allen predictions but other too (tempeval, bruce, etc)
   - performance bottleneck seems to be deepcopying of the graph before saturation!
     -> replace with list of base relations added, and resaturate the graph
     from scratch when an inconsistent relation is tried
"""


#TODO: import from module
from optimisation.Problem import Problem
from graph.Graph import CSPGraph, Edge, allen_algebra
from graph.Relation import Relation
import random
import copy
import math
import sys
import time
from TempGraphSearch import TempGraphBeamSearch, TempGraphSearch, TempGraphState, makeGraphFromList


class TempGraphProblem(Problem):
    """
    instance of temporal graph with probability for each relation on a subset
    of edges.

    implements the 'Problem' interface, to be used by Local Search strategies
    """
    def __init__(self,probability_distribution,filter_relations=None,relation_set="allen"):
        self._prob_dist=probability_distribution
        self._relset=relation_set
        
        if filter_relations is not None:
            keys=self._prob_dist.keys()
            for one in keys:
                if not(one in filter_relations):
                    del self._prob_dist[one]
        #self.graph=CSPGraph(allen_algebra)
        #for (e1,e2) in self._prob_dist:
        #    self.graph.addNode(e1)
        #    self.graph.addNode(e2)

    def get_relset(self):
        return self._relset

    # todo: retry relation instances until one is found consistent
    def randomStart(self,threshold=0.1):
        """generates a consistent graph including some or all
        the edges with given probability at instance creation

        TODO: check for probability =1.0
        """
        self.graph=CSPGraph(allen_algebra)
        for (e1,e2) in self._prob_dist:
            self.graph.addNode(e1)
            self.graph.addNode(e2)

        relschema=self.get_relset()
        consistent=True
        edges=set(self._prob_dist.keys())
        certain=self._prob_dist.get_sure()
        for (e1,e2) in certain:
            rel,pr=self._prob_dist[e1,e2][0]
            rel=Relation(rel)
            if relschema!="allen":
                rel=rel.other2allen(relschema)
            self.graph.add(Edge(e1,e2,rel))

        edges=edges-set(certain)
        #print >> sys.stderr, "starting with %d certain relations"%len(certain)
       
        edges=list(edges)
        random.shuffle(edges)
        for (e1,e2) in edges:
            save = copy.deepcopy(self.graph)
            pr=0
            i=0
            while (pr<threshold and i<10): 
                rel,pr  = random.sample(self._prob_dist[e1,e2],1)[0]
                i=i+1
            rel=Relation(rel)
            if relschema!="allen":
                rel=rel.other2allen(relschema)
            self.graph.add(Edge(e1,e2,rel))
            consistent=self.graph.saturate()
            if not(consistent):
                self.graph=save
                break
            
        return self.graph

    def specificStart(self,threshold=0.1):
        """
        approximate solution generated to initiate the local search

        TODO: check for probability =1.0
        TODO: cleanup that mess
        """
        self.graph=CSPGraph(allen_algebra)
        for (e1,e2) in self._prob_dist:
            self.graph.addNode(e1)
            self.graph.addNode(e2)

        relschema=self.get_relset()
        consistent=True
        edges=set(self._prob_dist.keys())
        certain=self._prob_dist.get_sure()
        for (e1,e2) in certain:
            rel,pr=self._prob_dist[e1,e2][0]
            rel=Relation(rel)
            if relschema!="allen":
                rel=rel.other2allen(relschema)
            self.graph.add(Edge(e1,e2,rel))

        edges=edges-set(certain)
        print >> sys.stderr, "starting with %d certain relations"%len(certain)
        consistent=self.graph.saturate()
        if not(consistent):
            print >> sys.stderr, "warning, inconsistent 'certain' information'"
            return self.graph
        nbedges=len(edges)
        #random.shuffle(edges)
        cpt=1
        for (e1,e2) in edges:
            dist=(x for x in self.sorted_prob(e1,e2))
            pr=1.0
            cont=True
            while (pr>threshold and cont and dist):
                try:
                    pr,rel=dist.next()
                    rel=Relation(rel)
                    save = copy.deepcopy(self.graph)
                    if relschema!="allen":
                        rel=rel.other2allen(relschema)
                    self.graph.add(Edge(e1,e2,rel))
                    t0=time.time()
                    consistent=self.graph.saturate()
                    #print >> sys.stderr, "saturation = %f s"%(time.time()-t0)
                    if not(consistent):
                        self.graph=save
                    else:
                        cont=False
                except:
                    cont=False
            #print >> sys.stderr, "%d/%d edges done, with added rel of proba=%f"%( cpt,nbedges,pr)
            cpt += 1 
        return self.graph

    def specificStartAstar(self,threshold=0.,cumulative=False,beam=0):
        """initiates a graph with Astar search and ad hoc heuristics

        or variant: beam search, with argument for size of beam 
        """
        # UNTESTED/not well tested at least
        used_probs=copy.deepcopy(self._prob_dist)
        used_probs.cut_threshold(threshold,cumulative)
        #print used_probs
        if beam==0:
            a=TempGraphSearch(heuristic=TempGraphState.h_moyenne,shared=used_probs)
        else:
            a=TempGraphBeamSearch(heuristic=TempGraphState.h_moyenne,shared=used_probs,queue_size=beam)
        sol=a.launch({},norepeat=True)
        #print a.iterations
        if sol is None:
            g=None
        else:
            g = makeGraphFromList(sol.data())
        if g is not None:
            g.saturate()
        return g


    def prob_dist(self):
        """the probability distribution on this instance
        """
        return self._prob_dist

    def randomNeighbour(self,current):
        """
        given a graph, generates a similar graph by
        changing only one edge among these with a probability distribution
        """
        edge = random.sample(self.prob_dist().keys(),1)[0]
        e1,e2 = edge
        rel,pr  = random.sample(self.prob_dist()[edge],1)[0]
        new=copy.deepcopy(current)
        relschema=self.get_relset()
        rel=Relation(rel)
        if relschema!="allen":
                rel=rel.other2allen(relschema)
        new.add(Edge(e1,e2,rel))
        consistent=new.saturate()
        if not(consistent):
            return current
        else:
            return new
        
    def bestNeighbour(self,current):
        """
        given a graph, generates a similar graph by
        changing only one edge among these with a probability distribution,
        choosing the best improvement
        TODO: deal with inconsistent choice
        """
        delta=0
        best_change=None
        relschema=self.get_relset()
        for e1,e2 in self.prob_dist():
            rel=current.get_edge(e1,e2)
            if rel is not None:
                rel=rel.relation()
                if relschema!="allen":
                    rel=rel.other2allen(relschema)
            proba=self.probability(e1,e2,rel)
            best_val,best_rel=self.best_prob(e1,e2)
            if best_val-proba>delta:
                best_change=best_rel
                delta=best_val-proba
                best_e1=e1
                best_e2=e2
        if best_change is not None:
            new=copy.deepcopy(current)
            best_change=Relation(best_change)
            if relschema!="allen":
                best_change=best_change.other2allen(relschema)
            new.add(Edge(best_e1,best_e2,best_change))
            consistent=new.saturate()
            if consistent:
                return new
        else:
            return current

            

    def best_prob(self,e1,e2):
        return self.sorted_prob(e1,e2)[0]


    def sorted_prob(self,e1,e2):
        result=sorted([(y,x) for (x,y) in self._prob_dist[(e1,e2)]])
        result.reverse()
        return result

    def probability(self,e1,e2,rel):
        """probability of relation rel on edge (e1,e2)
        """
        # no relation present expressed as universal relation holding with prob = 1/nb possible allen relations
        if rel is None:
            return 1.0/len(self.graph.algebra().universal())
        elif rel==set():
            return 0
        elif len(rel)!=1:# not a single relation, do not evalute for now
            # TODO: add probas of each member of the disjunction
            return 1.0/len(rel)
        else:
            #the_rel=list(rel)[0]
            the_rel=rel
            for relbis,pr in self._prob_dist[(e1,e2)]:
                if the_rel==Relation(relbis):
                    return pr
            return 0.0


    # true because we minimize the -log of the probability   
    def minimize(self):
        return True

    def value(self,instance):
        # value of graph is sum(-log(proba(edge)))
        # to minimize rounding errors on small probabilities
        # and because it can be used by Astar search
        total=0.
        relschema=self.get_relset()
        for (e1,e2) in self.prob_dist():
            rel=instance.get_edge(e1,e2)
            if rel is not None:
                rel=rel.relation()
                if relschema!="allen":
                    rel=rel.other2allen(relschema)
            prob=self.probability(e1,e2,rel)
            if prob==0.0:
                print >> sys.stderr, "warning, probability = 0", e1,e2,rel
                #total=total+sys.maxint
            else:
                total=total+math.log(prob)
        return -total


if __name__=="__main__":
    import sys
    from TimeMLDocument import Document
    from model import MegamClassifier
    from linker import Linker
    from optimisation.LocalSearch import *
    from Evaluation import graph_compare

    # simple example
    prob={('e1','e2'):[('trel_before',0.6),('trel_beforei',0.4)],
          ('e2','e3'):[('trel_before',0.6),('trel_beforei',0.4)],
          ('e1','e3'):[('trel_before',0.6),('trel_beforei',0.4)],
          ('e1','e4'):[('trel_before',1.0)],
          }
    #real graph + model
    modelfile = "../results/models/tb_nodis_sat_nobeth/9.megam"
    # 8 TLINKS in original annotation
    onefile = '../results/folds/tb_folds/9/test/wsj_0340_orig.tml.xml'

    # models in tempeval relation set
    modelfile=  "../results/models/tempeval_sat0_beth0_disj0_models/9.megam"
    onefile= "../results/folds/otc_folds/9/test/wsj_0706_orig.tml.xml"

    doc=Document(onefile)
    gold=doc.get_event_event_relations()
    print "gold contains %d event-event relations"%len(gold)
    classifier=MegamClassifier(paramfile=modelfile)
    l=Linker(classifier)
    prob=l.compute_local_probabilities(doc)
    
    print "nb of possible event pairs:",len(prob)
    #a=TempGraphProblem(prob,filter_relations=gold.keys())
    a=TempGraphProblem(prob,relation_set="tempeval")
    print "problem initialised"
    
    t0=time.time()
    pas = a.specificStart()
    print "while consistent (best) start: %fs, value:%f"%(time.time()-t0, a.value(pas))
    print "real scores", graph_compare(doc.get_graph(),pas,relation_set="tempeval")
    # TODO: evaluate with real measures

    rs=a.randomStart()
    print "random start"
    #print rs
    print "value", a.value(rs)
    print "random neighbour from random start"
    #for i in range(1):
    v= a.randomNeighbour(rs)
    #print v
    print "value",a.value(v)

    print "best neighbour  from random start"
    #for i in range(1):
    v= a.bestNeighbour(rs)
    #print v
    if v is not None:
        print "value:",a.value(v)
    else:
        print "no better neighbour"

    t0 = time.time()
    print "while consistent (best) start"
    search=HillClimbingSearch(a)
    iterNb=search.launch(200,startmode="bf",threshold=0.,verbose=True)
    best=search.getBest()
    print "search with iterations: %fs, value:%f" % (time.time()-t0, a.value(best))
    print "local search with %d iterations"%iterNb
    print "best value of -log(prob)", a.value(best)
    print "coherent solution ?", best.saturate()
    print "real scores", graph_compare(doc.get_graph(),best,relation_set="tempeval")

    if False:
        t0=time.time()
        threshold=0.1
        pas=a.specificStartAstar(threshold=threshold,beam=1000)
        if pas is None:
            value=0
            print "no solution with parameter threshold=%f, beam size=%d"%(threshold,1000)
        else:
            value=a.value(pas)
        print "astar start: %fs, value:%f"%(time.time()-t0, value)
    
