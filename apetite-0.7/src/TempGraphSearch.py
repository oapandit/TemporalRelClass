#!/bin/env python
# -*- coding: iso-8859-1 -*-

"""
module for building of consistent temporal
graphs from probability distribution, using
Astar heuristics based search and variants (beam, b&b)

TODO:
   - store a saturated graph for each state, to avoid redoing it for each successor state
"""

from optimisation.Astar import State, Search, BeamSearch
from graph.Graph import AllenGraph, CSPGraph, Edge, allen_algebra
from graph.Relation import Relation
import random
import copy
import math


def makeGraphFromList(data):
    g=CSPGraph(allen_algebra)
    g.addEdgeList([Edge(e[0],e[1],Relation(rel)) for (e,rel) in data.items()])
    return g

class TempGraphState(State):
    """
    instance of temporal graph with probability for each relation on a subset
    of edges.

    implements the State interface to be used by Search

    strategy: at each step of exploration choose a relation between two events
    related by probability distribution

    data is set of instantiated relations
    shared points to shared data between states (here proba distribution)
    """
    def __init__(self,data,heuristics,shared):
        self._data=data
        self._cost=0
        self.tobedone=None
        self._shared=shared
        self._h=heuristics(self)
        
        
        
    def proba(self,evt_pair):
        return self._shared[evt_pair]

    def shared(self):
        return self._shared

    
    # solution found when everything has been instantiated
    def isSolution(self):
        if self.tobedone is None:
            instantiated=set(self.data().keys())
            self.tobedone=set(self.shared().keys())-instantiated
        return self.tobedone==set()

 
    def nextStates(self):
        all=[]
        one=self.tobedone.pop()
        
        for (r,pr) in self.proba(one):
            newdata=copy.deepcopy(self._data)
            newdata[one]=r
            g=makeGraphFromList(newdata)
            if g.saturate():# is graph consistent ?
                # then add a state to the queue, with cost in log probability 
                all.append((newdata,-math.log(pr)))
        return all

    def __str__(self):
        return str(self.data())+"-"+str(self.cost())

    
    def __repr__(self):
        return str(self.data())+"-"+str(self.cost())


    # heuristiques
    def h_moyenne(self):
        instantiated=set(self.data().keys())
        self.tobedone=set(self.shared().keys())-instantiated
        result=0.
        for one in self.tobedone:
            #moyenne=sum(self.proba(one).values())/len(self.proba(one))
            #result += -math.log(moyenne)
            # always -log(0.5) for prob. dist.            
            result += -math.log(1.0/len(self.proba(one)))
        return result

    def h_best(self):
        instantiated=set(self.data().keys())
        self.tobedone=set(self.shared().keys())-instantiated
        result=0.
        for one in self.tobedone:
            maxp=max([pr for (r,pr) in self.proba(one)])
            result += -math.log(1.0/len(self.proba(one)))
        return result




class TempGraphSearch(Search):

    def newState(self,data):
        return TempGraphState(data,self._hFunc,self.shared())

class TempGraphBeamSearch(BeamSearch):

    def newState(self,data):
        return TempGraphState(data,self._hFunc,self.shared())




if __name__=="__main__":
    import sys
    import time


    prob={('e1','e2'):[('trel_before',0.6),('trel_beforei',0.4)],
          ('e2','e3'):[('trel_before',0.6),('trel_beforei',0.4)],
          ('e1','e3'):[('trel_before',0.4),('trel_beforei',0.6)],}
  
    h0=(lambda x: 0.)
    h1=TempGraphState.h_moyenne
    h2=TempGraphState.h_best

    t0=time.time()
    a=TempGraphSearch(heuristic=h1,shared=prob)
    sol=a.launch({},norepeat=True)
    print "total time:",time.time()-t0
    print sol
    print sol.cost()
    print a.iterations
    print makeGraphFromList(sol.data())
    print "total time:", time.time()-t0
    a = TempGraphSearch(heuristic=h2, shared=prob)
    sol=a.launch({},norepeat=True)
    print sol
    print sol.cost()
    print a.iterations
    print makeGraphFromList(sol.data())
    print "total time:",time.time()-t0