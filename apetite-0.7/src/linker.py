#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-



import os, sys
import operator
from copy import deepcopy
import signal
from graph.Graph import AllenGraph, CSPGraph, allen_algebra, Edge, merge_graphs
from graph_compare import interval2point
from graph.Relation import Relation
from instance import TimeMLEvtEvtInstance
from optimisation.LocalSearch import *
from TempGraphProblem import *
from ilp import ILPTempIntervalGraphProblem, ILPTempPointGraphProblem
from utils import ProbDist


# todo: a lot of functions should now be factorized
# as methods of this class (in linker, localSearch mainly ?)


class Linker:


    def __init__(self, classifier, relset="allen", test_options={}):
        """ Abstract class for linker """
        self.classifier = classifier
        self.relset = relset
        self.options = test_options
        return


    def get_predictions(self, doc):
        link_predictions = self.link( doc )
        result={}
        for pair,rel in link_predictions.items():
            if rel is not None:
                try:
                    result[pair]=Relation(rel)
                except:
                    print >> sys.stderr, "problem with relation", rel
        return result



    def link(self, doc):
        """ to be implemented by subclasses """
        raise NotImplementedError 



    def compute_local_probabilities(self, doc):
        local_probabilities = ProbDist()
        annot_rels =  doc.get_relations()
        entities = doc.get_events() # only events for now
        entity_index = entities.keys()
        for i in range(len(entity_index)-1):
            e1 = entity_index[i]
            ent1 = entities[e1]
            for j in range(i+1,len(entity_index)):
                e2 = entity_index[j]
                ent2 = entities[e2]
                if self.options.get('annotations_only',False):
                    if (e1,e2) not in annot_rels:
                        continue
                test_inst = TimeMLEvtEvtInstance(None,ent1,ent2)
                prob_distrib = self.classifier.class_distribution( test_inst.fv )
                local_probabilities[e1,e2] = prob_distrib
        # add time-time and event-time relations from reference (incl. inferred relations)
        if self.options.get("time_time", False):
             tt_et_rels = doc.get_tt_et_relations(loc_sat=True, isodate=True)
             for e1,e2 in tt_et_rels:
                 if self.relset!="allen":
                     rel=tt_et_rels[e1, e2].allen2other(self.relset)
                 else:
                     rel=tt_et_rels[e1, e2]
                 local_probabilities[e1, e2] = [(rel, 1.0)]

        return local_probabilities


    


class ZeroLinker(Linker):
    """ Oracle linker: a priori knowledge of E-T and T-T relations
    (univ. disj. predicted on all E-E)."""

    def link(self, doc,threshold=0.):
        links = {}
        # local_probs = self.compute_local_probabilities( doc )
        # if self.options.get("threshold",False):
        #     threshold=self.options.get("threshold")
        # for (e1,e2) in local_probs:
        #     if self.relset!="allen":
        #         links[e1,e2] = allen_algebra.universal().allen2other(self.relset)
        #     else:
        #         links[e1,e2] = allen_algebra.universal()
        return links





class BaseLinker(Linker):

    def link(self, doc,threshold=0.):
        """ for each possible edge, return relation that has the
        highest prob. according to local classifier --no consistency
        check

        if threshold is set, don't instantiate edges with best prob<threshold
        """
        links = {}
        local_probs = self.compute_local_probabilities( doc )
        if self.options.get("threshold",False):
            threshold=self.options.get("threshold")
        for (e1,e2) in local_probs:
            rel_distrib = local_probs[e1,e2]
            rel_distrib.sort(key=operator.itemgetter(1),reverse=True)
            best,pr=rel_distrib[0]
            if pr>=threshold:
                links[e1,e2] = best
        return links




class BaseLinkerDisjunctive(Linker):

    def link(self, doc, threshold=0.5):
        """ for each possible edge, return minimal convex disjunction of relations that has
        a total prob. according to local classifier >= threshold"""
        links = {}
        local_probs = self.compute_local_probabilities( doc )
        if self.options.get("threshold",False):
            threshold=self.options.get("threshold")

        for (e1,e2) in local_probs:
            rel_distrib = local_probs[e1,e2]
            rel_distrib.sort(key=operator.itemgetter(1),reverse=True)
            
            links[e1,e2]=Relation(rel_distrib[0][0])
            total_prob=rel_distrib[0][1]
            i=1
            while total_prob<threshold:
                links[e1,e2] = links[e1,e2] | Relation(rel_distrib[i][0])
                total_prob += rel_distrib[i][1]
                i += 1
        return links



class NRO_Linker(Linker):
    """ Natural Reading Order (NRO). Add entity as they appear in
    text, linking each entity to the existing graph with the edge that
    has highest score according to pairwise classifier w/o violating
    graph consistency."""

    def link(self, doc,threshold=0.):
        # get local probabilities
        local_probs = self.compute_local_probabilities( doc )
        if self.options.get("threshold",False):
            threshold=self.options.get("threshold")
        # create empty graph
        graph = AllenGraph()
        # sort entities according to text order
        # TODO/FIXME: changed to event instead of entities but not sure (PM)
        ordered_entities = doc.get_events().values()
        ordered_entities.sort(key=lambda x: x.event.extent)
        # sort all edges according to local probabilities
        sorted_edges = []
        relschema=self.relset
        for (e1,e2) in local_probs:
            rel_distrib = local_probs[e1,e2]
            for (rel,prob) in rel_distrib:
                if prob>=threshold:
                    if relschema!="allen":
                        rel=Relation(rel)
                        rel=rel.other2allen(relschema)
                    sorted_edges.append( (((e1,e2),rel),prob) )
        sorted_edges.sort(key=operator.itemgetter(1),reverse=True)
        # add entity to graph
        added_entity_ct = 0
        for e in ordered_entities:
            # create node for entity
            graph.addNode(e.id)
            # find edge with highest prob            
            edges = [edge for edge in sorted_edges if e.id in edge[0][0]]
            # print "ENTITY:", e.id
            # print "10 best Edges:", edges[:10]
            # add edge that preserves consistency
            for ((e1,e2),rel),prob in edges:
                tmp_graph = deepcopy( graph )
                arc = Edge( e1, e2, rel )
                tmp_graph.add( arc )
                cons = tmp_graph.saturate()
                if cons:
                    added_entity_ct += 1
                    #print ">>> Added arc:", arc
                    graph = tmp_graph
                    break
        print "%s/%s added entities" %(added_entity_ct,len(ordered_entities))
        # convert graph edges into prediction tuples
        links = {}
        for arc in graph.edges().values():
            if relschema!="allen":
                rel=arc.rel().allen2other(relschema)
            else:
                rel=arc.rel()
            links[arc.node1(),arc.node2()] = rel
        return links



class BFI_Linker(Linker):
    """ Best-first inference (BFI). Sort edges by probability score
     and add one edge at a time w/o violating consistency. This aims
     at optimizing graph score."""

    def link(self, doc,debug=False):
        # get local probabilities
        local_probs = self.compute_local_probabilities( doc )
        # create empty graph
        graph = AllenGraph()
        # sort all edges according to local probabilities
        sorted_edges = []
        if debug:
            print >> sys.stderr, local_probs
        relschema=self.relset
        for (e1,e2) in local_probs:
            rel_distrib = local_probs[e1,e2]
            for (rel,prob) in rel_distrib:
                if relschema!="allen":
                    rel=Relation(rel)
                    rel=rel.other2allen(relschema)
                sorted_edges.append( (((e1,e2),rel),prob) ) 
        sorted_edges.sort(key=operator.itemgetter(1),reverse=True)
        if debug:
            print >> sys.stderr, sorted_edges[:10]
        # add edges to graph in that order
        connected_nodes = {}
        for ((e1,e2),rel),prob in sorted_edges:
            if (e1,e2) in connected_nodes:
                continue
            graph.addNode(e1)
            graph.addNode(e2)
            tmp_graph = deepcopy( graph )
            graph.add( Edge(e1, e2, rel) )
            cons = graph.saturate()
            if cons:
                #graph = tmp_graph
                if debug:
                    print >> sys.stderr, "consistent add:", ((e1,e2),rel),prob
                connected_nodes[e1,e2] = 1
            else:
                graph= tmp_graph
        # convert graph edges into prediction tuples
        links = {}
        for arc in graph.edges().values():
            if relschema!="allen":
                rel=arc.rel().allen2other(relschema)
            else:
                rel=arc.rel()
            links[arc.node1(),arc.node2()] = rel 
        return links

    

    
 
class ILPLinker(Linker):

    # FIXME: a refaire plus ou moins integralement
    def link(self, doc):

        links = {}

        # get local probs
        local_probs = self.compute_local_probabilities( doc )

        # get connex graphs
        graph = doc.get_event_event_graph()
        tt_et_graph = None
        # add T-T and E-T relations
        if self.options['time_time']:
            tt_et_graph = doc.get_tt_et_graph()
            #unsat_graph = deepcopy(tt_et_graph)
            #cons = tt_et_graph.saturate()
            #if not cons:
            #    tt_et_graph = unsat_graph
            graph = merge_graphs([graph,tt_et_graph])
        # decompose 
        subgraphs = []
        if graph.nodes():
            subgraphs = graph.decompose()
            print len(subgraphs), "connex sub-graph(s)"
            ct = 1
            for graph in subgraphs:
                print "graph %s: %s nodes" %(ct,len(graph.nodes()))
                ct += 1
        
        # create ILP problem for each subgraph
        for sg in subgraphs:
            local_probs = dict([((e1,e2),local_probs[e1,e2]) for (e1,e2) in local_probs
                                 if e1 in sg.nodes() and e2 in sg.nodes()])
            # ILP intervals
            # tt_et_rels = {}
            # if tt_et_graph:
            #     tt_et_rels = dict([((a.node1(),a.node2()),a.rel()) for a in tt_et_graph.edges().values()])
            # ilp_problem = ILPTempIntervalGraphProblem( sg, local_probs, tt_et_rels=tt_et_rels )

            # ILP points
            pt_tt_et_graph = None
            pt_tt_et_rels = {}
            if tt_et_graph:
                pt_tt_et_graph = interval2point( tt_et_graph )
                pt_tt_et_rels = dict([((a.node1(),a.node2()),a.rel()) for a in pt_tt_et_graph.edges().values()])
            ilp_problem = ILPTempPointGraphProblem( graph, local_probs, tt_et_rels=pt_tt_et_rels )
            # set ILP problem
            ilp_problem.set( trans_limit=self.options['translim'] )
            # solve ILP
            ilp_problem.robust_solve( time_limit=self.options['tmlim'], mem_limit=self.options['memlim'] )
            # collect predictions
            sg_links = ilp_problem.extract_relations()
            links.update( sg_links )

        return links
        
    



class LocalSearchLinker(Linker):
    """try to build a consistent graph from probability distribution,
    maximizing the combined probability of all the relations

    TODO: use test_options for parameter instead of adhoc arguments (maxiter,startmode, threshold

    also for adding info to graph: tt relations/ e-t relations, ect cf testSearch
    """
    def __init__(self, classifier, relset="allen", test_options={},filter=None,mode="greedy"):
        Linker.__init__(self, classifier, relset, test_options)
        self.mode=mode
        self.filter=filter

    def set_mode(self,mode):
        self.mode=mode

    def link(self,doc,maxIter=100,startmode="random",threshold=0.):
        local_probs = self.compute_local_probabilities( doc )
        solutions=TempGraphProblem(local_probs,filter_relations=self.filter,relation_set=self.relset)
        if self.mode=="greedy":
            search=GreedySearch(solutions)
        else:
            search=HillClimbingSearch(solutions)
        iterNb=search.launch(maxIter,startmode=startmode,threshold=threshold)
        best=search.getBest()
        print >> sys.stderr, "local search with %d iterations"%iterNb
        print >> sys.stderr, "best value of -log(prob)", solutions.value(best)

        localSearchPredictions = {}
        
        for (e1,e2) in local_probs:
            pred=best.get_edge(e1,e2)
            if pred is not None and len(pred.relation())!=0:
                pred=list(pred.relation())[0]
            localSearchPredictions[e1,e2] = pred

        return localSearchPredictions


######################################################################
# EXCEPTIONS etc.
######################################################################


class LinkerError(Exception):
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

    








############################################################


