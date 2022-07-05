#!/bin/env python
# -*- coding: iso-8859-1 -*-
#
# tests de mesures de comparaison de graphes
# utilis� dans timeml_compare
# cf exemple dans timeml_compare
#
# 
#
# TODO:
#  ok  - check python2.3/ 2.4
#  ok  - single measure relations
#  ok  - variantes sur arcs consid�r�s
#      - int�gration a timeml_compare
#  ok  - factoriser mesures avec filtres sur arcs et/ou relations
#         ex: single-precision: human arcs with len(relation)=1+ test equality
#             finesse: human arcs non triviaux + mesure finesse
#             event-event seult ou event/time seulement + relation anchor (during)
#      - essayer poids sur mesure d'info suivant theorie de l'information
#      - comparaison des extr�mit�s des intervalles, cf article eacl 2009
#           x- conversion event -> pt
#           x- mesures rel simples
#           x- min graphe (transitive reduction)
#           - mesures convexes
#           - scripts s�par�s (figures, etc)
#      - tout devrait etre dans Evaluation (ou Graph)
#      - indexation des arcs sur premier noeud fout le bordel dans les graphes de point
#        -> mieux encapsuler l'initialisation de ces graphes et remettre un index pour eux

from graph.Graph import Graph, AllenGraph, CSPGraph, Edge, allen_algebra
from graph.Relation import Algebra, rel_name, Relation
import read_timeml as timeml
from utils import make_graph_from_rels

import sys
import os, glob
import copy
try:
	a = set()
except Exception,e:
	from sets import Set as set
import math
from collections import defaultdict

#import pdb


# compare deux relations / ensembles
# (doivent etre non vides = annotation coherente

#inutile
def card_intersection(a,b):
	return len(a & b)

# a inter b / b
def mesure_inter(a,b):
	if len(b)!=0:
		return 1.0*len(a & b)/len(b)
	else:
		print >> sys.stderr, "warning, inconsistent relation, returning 0 as a measure"
		return 0.0
# r1.r2/r2
def relation_finesse(r1,r2,adjusted=False,algebra="allen"):
	"""measure of overlap H/S
	
	adjusted means normalised with respect to chance (not tested) 
	"""
	finesse=mesure_inter(r1,r2)
	#if adjusted:
	#	if algebra=="allen":
	#		all=13
	#	else:
	#		all=7
	#	p=(len(r1 & r2))**2/(len(r2)*len(r1))
	#	return ?
	#else:
	return finesse

# r1.r2/r1
def relation_coherence(r1,r2):
	return mesure_inter(r2,r1)

# r1 -> r2
def relation_compatibilite(r1,r2):
	return r1<=r2

# jaccard
def jaquard(a,b):
	return 1.0*len(a & b)/len(b | a)


# obsolete, use TimeMLdocument instead
def timeml2graph(filename):
    one=filename
    base1=open(one).read()
    data1,t1=timeml.extract_from_string(base1)
    #data1bis=timeml.normalise(data1)
    data1bis=data1
    g1=timeml.make_graph(data1bis)
    return g1 


# Dice coefficient (check formula)
def dice_coeff(a,b):
	pass

# mesure pond�r�e par le niveau d'information
# d'une relation = -p*log(p)
# ou p est la proba que la relation "r�elle" soit dans la relation
# ex en Allen: -1/13(log(1/13) pour une relation simple
from math import log
def information(edge1,edge2):
	#edge1=human
	p=len(edge1.relation())/13
	return (-log(p)*p)

# tests de la nature des arcs
# (bool�en utilis� pour filtrage des mesures)
# type1: type du noeud origine ex: "event","timex"
# type2: type du noeud arriv�e
# Warning: suppose normalisation: le type timex/event est la premiere lettre
# de l'identit� du noeud de l'arc concern� (ex: t53-53 ou e123)
def arc_type(arc,typ1,typ2,oriented=False):
    # premiere lettre suffit a distinguer le type t/e
    try:
        res=(arc.node1()[0].lower()==typ1[0].lower() and arc.node2()[0].lower()==typ2[0].lower())
    except:
        print >> sys.stderr, arc, typ1, typ2
        raise Exception
    if not(oriented):
        res=(res or (arc_type(arc,typ2,typ1,oriented=True)))
    return res


###############################
# definition :pt_algebra
###############################
ptAlgebra_relset=["<","==",">"]
ptAlgebra_ext_relset=["<","==",">","<=",">="]
ptAlgebra_inv={'<':'>',
	       '>':'<',
	       '==':'=='}
ptAlgebra_compo={'<':{'<':['<'],'>':['<','>','=='],'==':['<']},
		 '>':{'>':['>'],'<':['<','>','=='],'==':['>']},
		 '==':{'<':['<'],'>':['>'],'==':['==']}
		 }

ptAlgebra_inv_explicit={'<':'>',
			'>':'<',
			'==':'==',
			'<=':'>=',
			'>=':'<='}

# Restricted to convex relations (no <>)
ptAlgebra_compo_explicit={
	'<':{'<':'<',
	     '<=':'<',
	     '==':'<'},
	'<=':{'<':'<',
	      '<=':'<=',
	      '==':'<='
	      },
	'>':{'>':'>',
	     '>=':'>',
	     '==':'>'},
	'>=':{'>':'>',
	      '>=':'>=',
	      '==':'>='},
	'==':{'<':'<','<=':'<=',
	      '>':'>','>=':'>=',
	      '==':'==',
	      }
	}


ptAlgebra=Algebra(ptAlgebra_relset,ptAlgebra_inv,ptAlgebra_compo)

# conversion d'une relation entre intervalle et bornes gauche
# sch�ma: extremite premier intervalle, ext. 2nd int, relation entre les bornes
# liste de conjonctions
# NOT CHECKED 
_conversion_templt={
	'b':[('right','left','<')],#ok
	'm':[('right','left','==')],#ok
	'o':[('left','left','<'),('right','left','>'),('right','right','<')],#ok
	's':[('left','left','=='),('right','right','<')],#ok
	'd':[('left','left','>'),('right','right','<')],#ok
	'f':[('left','left','>'),('right','right','==')],#ok
	'=':[('left','left','=='),('right','right','==')],#ok
	'e':[('left','left','=='),('right','right','==')],#ok
	'fi':[('left','left','<'),('right','right','==')],#ok
	'di':[('left','left','<'),('right','right','>')],#ok
	'si':[('left','left','=='),('right','right','>')],#ok
	'oi':[('left','right','<'),('right','right','>'),('left','left','>')],#ok
	'mi':[('left','right','==')],#ok
	'bi':[('left','right','>')],#ok
	}


# back conversions from point to allen
# needs to explicit all pt-pt relations between two events
_total_conversion_templt=copy.copy(_conversion_templt)
_total_conversion_templt['b'].extend([('left','left','<'),('right','right','<'),('left','right','<')])
_total_conversion_templt['bi'].extend([('left','left','>'),('right','right','>'),('right','left','>')])
_total_conversion_templt['m'].extend([('left','left','<'),('right','right','<'),('left','right','<')])
_total_conversion_templt['mi'].extend([('left','left','>'),('right','right','>'),('right','left','>')])
_total_conversion_templt['o'].extend([('left','right','<')])
_total_conversion_templt['oi'].extend([('right','left','>')])
_total_conversion_templt['e'].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['='].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['s'].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['si'].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['f'].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['fi'].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['d'].extend([('right','left','>'),('left','right','<')])
_total_conversion_templt['di'].extend([('right','left','>'),('left','right','<')])

_conversion={}
for one_rel in _total_conversion_templt:
	_conversion[rel_name[one_rel]]=_total_conversion_templt[one_rel]


_pt2i_conversion=defaultdict(list)
for allenrel in _total_conversion_templt:
	for b1,b2,ptr in _total_conversion_templt[allenrel]:
		_pt2i_conversion[(b1,b2,ptr)].append(rel_name[allenrel])


# ok 
def read_graphe(filename,algebra,conversion=None,normalise=True):
	"""reads graph written in ascii for a certain algebra
	format: relation node1 node2
	on each line

        if conversion is not None, it can provide a function converting the input into the target algebra
        eg  lambda x : x.other2allen("tempeval")
	"""
        if conversion is not None and algebra==allen_algebra:
            g=AllenGraph()
        else:
            g=CSPGraph(algebra)
	brut=open(filename).readlines()
	metadata=[x.strip().split(" ",2)[1:] for x in brut if x.startswith("##")]
	#print metadata
	metadata=dict(metadata)
	if metadata.has_key("f2"):
		g._extra={"f2":float(metadata["f2"].split()[-1])}
	fin=[x.strip().split() for x in brut if x.strip()!="" and not(x.startswith("#"))]
	for r,n1,n2 in fin:
                if normalise:
                    n1=n1.upper()
                    n2=n2.upper()
		g.addNode(n1,properties=n1)
		g.addNode(n2,properties=n2)
                #
                if conversion is not None:
                    r=conversion(Relation(r))
                # adds an edge and the symmetric
		g.add(Edge(n1,n2,r))
	return g






######################################
# read result from global solver vs local classifier
#
######################################
def read_ILPresults(filename):
	all=open(filename).readlines()
	line=all[0]
	if line.startswith("# No ILP assignments"):
		noILP=True
		print >> sys.stderr, filename, "no ILP assignments"
	else:
		noILP=False
	all=all[1:]
	for (i,line) in enumerate(all):
		if line.startswith("# E-E predictions"):
			break
	reference=set([])
	ilp_res=set([])
	classif=set([])
	for line in all[i+1:]:
		if not(line.startswith('#')) and not(line.startswith('Local')) and not(line.startswith('ILP')):
			try:
				(e1, e2, ilp_pred, classif_pred, gold) = line.split()
				if noILP:
					ilp_pred=classif_pred
				elif ilp_pred is None:
					ilp_pred=allen_algebra.universal()
				classif.add((e1,e2,classif_pred))
				reference.add((e1,e2,gold))
				ilp_res.add((e1,e2,ilp_pred))
			except:
				if not(line.startswith('#')):
					print >> sys.stderr, "warning: no parse for line in", filename, line
	
	total_nb    = len(reference)
	ilp_acc     = len(ilp_res & reference)
	classif_acc = len(classif & reference)
	if total_nb!=0:
		print filename, total_nb, "event pairs; ilp:", ilp_acc/float(total_nb), "; classif :",classif_acc/float(total_nb)
	else:
		print filename, "no event pairs!"

	g_ilp={}
	g_gold={}
	g_classif={}
	classif_inc=-1
	try:
		g_ilp=AllenGraph()
		g_ilp.addRelationList(ilp_res)
		sat=g_ilp.saturate()
		print filename, "ilp consistency:", sat

	except:
		print >> sys.stderr, filename, "error computing ilp graph\n",ilp_res
	try:
		g_classif=AllenGraph()
		g_classif.addRelationList(classif)
		if not(g_classif.saturate()):
			print filename, "inconsistent classif"
			g_classif=AllenGraph()
			g_classif.addRelationList(classif)
			classif_inc=1
		else:
			classif_inc=0
			print filename, "consistent classif"
	except:
		print >> sys.stderr, filename, "error computing classif graph\n",classif_pred
	try:
		g_gold=AllenGraph()
		g_gold.addRelationList(reference)
	except:
		print >> sys.stderr, filename, "error computing gold graph"

	return total_nb,ilp_acc,classif_acc,g_ilp,g_classif,g_gold, classif_inc




def compile_ILPresults(directory,suffix=".solutions"):
        results=glob.glob(os.sep.join((directory,"*"+suffix)))
	total=0
	ilp_acc=0
	classif_acc=0
	classif_inc=0
	
	for filename in results:
		total_nb,one_ilp_acc,one_classif_acc,g_ilp,g_classif,g_gold,one_classif_inc=read_ILPresults(filename)
		total+=total_nb
		ilp_acc+=one_ilp_acc
		classif_acc += one_classif_acc
		classif_inc += one_classif_inc
	print  "pairs:%d, ilp:%d, classif:%d, inconsistent_classif:%d"%( total,ilp_acc,classif_acc,classif_inc)

#####################################
# conversion graphe
#d'intervalles -> graphe de points
#
# Test: ok (simple,convexe)
#####################################
def interval2point(graphe):
	"""  conversion graphe
	d'intervalles -> graphe de points
	"""
	pt_graph=CSPGraph(ptAlgebra,index=False)
	# necessary in case some nodes are not related, but need to be compared
	# later
	for node in graphe.nodes():
		pt_graph.addNode(str(node)+'_left',node)
		pt_graph.addNode(str(node)+'_right',node)
	
	for rel in graphe.edges().values():
		a1,a2=rel.node1(),rel.node2()
		# kind of useless now because of code above
		pt_graph.addNode(str(a1)+'_left',a1)
		pt_graph.addNode(str(a1)+'_right',a1)
		pt_graph.addNode(str(a2)+'_left',a2)
		pt_graph.addNode(str(a2)+'_right',a2)
		all_rels={}
		pt_graph.add(Edge(str(a1)+'_left',str(a1)+'_right','<'))
		pt_graph.add(Edge(str(a2)+'_left',str(a2)+'_right','<'))
		for one_rel in rel.relation():
			# use conv to generate pt relations
			base_set=_conversion[one_rel]
			# extend point graph with all basic relations
			base_set=([("%s_%s"%(str(a1),x[0]),"%s_%s"%(str(a2),x[1]),x[2]) \
					for x in base_set])
			# if multiple solutions, generate new label,
			for (n1,n2,rel) in base_set:
				all_rels.setdefault((n1,n2),[]).append(rel)	 
		# naming must respect conventions: b,e bi,e b, bi are the only possibilities	
		for (n1,n2) in all_rels:
			all_rels[(n1,n2)].sort()
			pt_graph.add(Edge(n1,n2," ".join(all_rels[(n1,n2)])))
		
	return pt_graph






def get_ptRels(ptg,e1,e2):
	"""get all point relations between event e1 and e2 (max: four)

	returns None if inconsistent
	
	"""
	e1left,e1right=e1+'_left',e1+'_right'
	e2left,e2right=e2+'_left',e2+'_right'
	res=[]
	res.append(ptg.cs_with_args(e1left,e2left))
	res.append(ptg.cs_with_args(e1left,e2right))
	res.append(ptg.cs_with_args(e1right,e2left))
	res.append(ptg.cs_with_args(e1right,e2right))
	return [(x.node1(),x.node2(),x.rel()) for x in res if x is not None]
	

def matchRel(ptRelSet):
	""" match a set of point-point relations between two events
	with corresponding set of Allen relations
	
	CHECK: inversion & et | ???
	surveiller sens de la double iteration
	"""
	
	if ptRelSet==[]:
		return allen_algebra.universal()
	else:
		res=allen_algebra.universal()
		for e1pt,e2pt,rels in ptRelSet:
			local=set()
			e1pt=e1pt.split('_')[1]
			e2pt=e2pt.split('_')[1]
			for r in rels:
				local = (local | set(_pt2i_conversion[(e1pt,e2pt,r)]))
			res = res & local
		return res


def point2interval(graphe):
	""" conversion graphe de points en graphe d'intervalle
	suppose la convention e1 cod� par les pts e1_left, e2_right
	"""
	int_graph=CSPGraph(allen_algebra)
	for one in graphe.nodes():
		evt_name=one.split('_')[0]
		int_graph.addNode(evt_name,evt_name)
	
	for e1 in int_graph.nodes():
		for e2 in int_graph.nodes():
			if e1!=e2:
				what=get_ptRels(graphe,e1,e2)
				allenRel=matchRel(what)
				if allenRel!=allen_algebra.universal():# non trivial relation 
					c=Edge(e1,e2,allenRel)
					int_graph.add(c)
	return int_graph



def point2interval2(graphe):
	""" conversion graphe de points en graphe d'intervalle
	suppose la convention e1 cod� par les pts e1_left, e2_right.

	NB: cette version utilise la methode addEdge plutot que add
	lors de la creation du graphe d'intervalles.
	"""
	int_graph=CSPGraph(allen_algebra)
	for one in graphe.nodes():
		evt_name=one.split('_')[0]
		int_graph.addNode(evt_name,evt_name)
	
	for e1 in int_graph.nodes():
		for e2 in int_graph.nodes():
			if e1!=e2:
				what=get_ptRels(graphe,e1,e2)
				allenRel=matchRel(what)
				# print "Adding:", e1,e2,allenRel
				if allenRel!=allen_algebra.universal():# non trivial relation 
					c=Edge(e1,e2,allenRel)
					# int_graph.add(c) # NOTE: this adds inverse too!!!!!!
					int_graph.addEdge(c)  
	return int_graph



###########################################
from copy import deepcopy
###########################################
# 
# Test: ok (simple, convexe)

# BUG: some nodes in merged graph have non-empty intersection !!! -> bug in fuse_nodes

#####################################
def fuse_nodes(ptGraphe):
	"""make one node of temporally equals points:
	
	TODO/BUG ? : check last condition seen(n1) & seen(n2) (was ignored before)

	TODO: points initialised beforehand so code can be simplified. 
	"""
	newptGraphe=CSPGraph(ptAlgebra,index=False)
	# 
	# add nodes and rels in new graph, associate equals pts => equivalence classes
	# to node first met
	# newpts are fused nodes, seen records nodes already processed
	# newpts are indexed by first node encountered and records all equal  nodes
	# seen is indexed by nodes and records newpts index containing the node
	newpts={}
	seen={}
	
	
	equals_rel=[rel for rel in ptGraphe.edges().values() if rel.relation()==Relation("==")]
	#print >> sys.stderr, "relevant rels" , equals_rel
	for rel in equals_rel:
		n1,n2=rel.node1(),rel.node2()
		if not(seen.has_key(n1)) and not(seen.has_key(n2)):
			# newly encountered pts, store under the first one
			newpts[n1]=set([n1,n2])
			seen[n1]=n1
			seen[n2]=n1
		elif not(seen.has_key(n1)) and seen.has_key(n2):
			seen[n1]=seen[n2]
			newpts[seen[n2]].add(n1)
		elif seen.has_key(n1) and not(seen.has_key(n2)):
			seen[n2]=seen[n1]
			newpts[seen[n1]].add(n2)
		else:# there was a bug here ... 
			# fusion of two sets from n1 and n2, reindex them both on n1-set
			#pdb.set_trace()
			newpts[seen[n1]]=newpts[seen[n1]] | newpts[seen[n2]]
			# bug corrected: the second set should then be deleted, unless
			# this is the same set, and all its members point to the fusion
			if seen[n1]!=seen[n2]:
				trash=copy.copy(seen[n2])
				for anode in newpts[trash]:
					seen[anode]=copy.copy(seen[n1])
				del newpts[trash]
	
	for (idx,value) in newpts.items():
		newptGraphe.addNode(idx,properties=value)
	
	# add also non-related nodes
	for node in ptGraphe.nodes():
		if not(seen.has_key(node)):
			newptGraphe.addNode(node,properties=set([node]))
		
	
	
	other_rels=[rel for rel in ptGraphe.edges().values() if rel.relation()!=Relation("==")]
	for rel in other_rels:
		n1,n2=rel.node1(),rel.node2()
		relation=rel.relation()
		if seen.has_key(n1):
			idx1=seen[n1]
		else:
			newpts[n1]=set([n1])
			seen[n1]=n1
			newptGraphe.addNode(n1,set([n1]))
			idx1=n1
		if seen.has_key(n2):
			idx2=seen[n2]
		else:
			newpts[n2]=set([n2])
			seen[n2]=n2
			newptGraphe.addNode(n2,set([n2]))
			idx2=n2
		newptGraphe.addEdge(Edge(idx1,idx2,relation))
		
	return newptGraphe


###########################################
# 
# transitive reduction
#   - ok sur relations simples 
#   - ok sur relations convexes 
#####################################
#
def inf_or_equals(x):
	return x.relation()==Relation("<") or x.relation()==Relation(['<','=='])

# yields relations given by composition of existing ones
# (=RxR) => non minimal relations
# similar to propagate, but only lists new relations
# and uses only '<' or '<='
def compose(ptGraph):
	base=deepcopy(ptGraph)
	#changed=False
	#consistent=True
	a=base.algebra()
	result=[]
	base_rels=[x for x in base.edges().values() if inf_or_equals(x)]
	#print base_rels
	for c in base_rels:
	    node2=c.node2()
	    node1=c.node1()
	    relation1=c.relation()
	    related=[x for x in base.cs_with_node1(node2) if inf_or_equals(x)]
            relation1=c.relation()
            #print "compose avec --:",related
            for c2 in related:
                c2n2=c2.node2()
                if node1!=c2n2:
                    #print "arc2 -----",c2
                    res=a.compose(relation1,c2.relation())
		    result.append((Edge(node1,c2n2,res)))
        return result

	



# transitive reduction : R+ - (R+ x R+)
# or R - (R+ x R+).
# r is relation r+ is positive closure 
# test: ok cf unittest
def transitive_reduction(ptGraph):
	ptg=deepcopy(ptGraph)
	non_min=compose(ptGraph)
	for one in non_min:
		ptg.del_edge(one)
	minGraph=CSPGraph(ptAlgebra,index=False)
	for one in ptg.nodes():
		minGraph.addNode(one,properties=ptGraph.nodes()[one])
	for one in ptg.edges().values():
		if inf_or_equals(one):
			minGraph.addEdge(one)	

	return minGraph

def min_ptGraph_all(ptGraph):
	return transitive_reduction(ptGraph)




# test: ok sur simple
#       ?normalement change rien sur convex 
def remove_obvious(minGraph):
	""" remove obvious edges (left<right) and non informative ones"""
	# DONT DO IT ON CUSTOM PT GRAPHS
	# WORKS ONLY IF LABELS ARE name+_left,name+_right
	obvious=[]
	for rel in minGraph.edges().values():
		# non informative -> out
		if rel.relation()==minGraph.algebra().universal():
			#minGraph.del_edge(rel)
			obvious.append(rel)
		else:
			if rel.relation()==Relation('<'):
				n1,n2=rel.node1(),rel.node2()
				pts1=set([x.replace("_left","").replace("_right","") for x in minGraph.nodes()[n1]])
				pts2=set([x.replace("_left","").replace("_right","")  for x in minGraph.nodes()[n2]])
				if len(pts1 & pts2)>0:
					obvious.append(rel)
					#minGraph.del_edge(rel)
	for rel in obvious:
		# print >> sys.stderr, rel
		minGraph.del_edge(rel)	
	return minGraph


# ok sur simple
# ok sur convex
def min_ptGraph(ptGraph):
	return remove_obvious(min_ptGraph_all(ptGraph))




# test: ok on simple graphe
# test: ok on convex
# bug on empty graphs: if g1 has no relations, and thus no point equalities, should be zero
# (it's not)
# culprit: some nodes in merged graph have non-empty intersection !!! -> bug in fuse_nodes
def split_nb(g1,g2):
	""" match nodes of point-based graphs
	= nb of separation necessary to match g1 to g2
	"""
	# equals sum of (number of parts)-1 in g2 for each set in g1
	# 
	nodeset1=g1.nodes().items()
	nodeset2=g2.nodes().items()
	count=0
	for (key1,one1) in nodeset1:
		parts=[(key2,one2) for (key2,one2) in nodeset2 if (one2 & one1)!=set()]
		# if one part, must be the same node : count = 0
		# otherwise, nb of split= nb of parts minus 1
		##�XT added "!= 0" condition
		if len(parts) != 0:
			count += len(parts)-1
			# discount for parts that are in relation "before or equals"
			for (key,x) in parts:
				if (key1,key) in g2.edges():
					if g2.edges()[(key1,key)].relation()==Relation(['<','==']):
						count = count -0.5
	return count

# ob simple+convex
def merge_nb(g1,g2):
	return split_nb(g2,g1)
			


# test: ok
def nb_evt(ptGraph):
	evt=set()
	for one in ptGraph.nodes().values():
		evt= evt | set([x.replace("_left","").replace("_right","") for x in one])
	nbevt=len(evt)
	return nbevt

# test ok on simple
def close_minGraph(ptGr):
	""" correct relations from given graph (i.e. after saturation), minus inverted(?)
	and obvious ones
	"""
	new=deepcopy(ptGr)
	new=min_ptGraph_all(new)
	new.saturate()
	remove_obvious(new)
	return new

# test ok on simple
# test ok on convex
def graph_value(ptGraph0):
	""" reference value = nb of 'good' relations:
	=(nb evt to start with*2 -nb nodes in ptgraph) + nb of proper relations
	"""
	#ptGraph=close_minGraph(ptGraph0)
	ptGraph=deepcopy(ptGraph0)
	remove_obvious(ptGraph)
	nbnode=len(ptGraph.nodes())
	rel=[x for x in ptGraph.edges().values() if inf_or_equals(x)] 
	nbrel=len(rel)
	evt=nb_evt(ptGraph)
	#print >> sys.stderr, "node, nb rel, nb evt", nbnode, nbrel, evt
	#print >> sys.stderr, rel

	# 
	return (evt*2-nbnode)+nbrel

# ok simple
def match(n1,n2):
	"""true if one node of n1 is in n2"""
	return (n1 & n2)!=set()

# ok simple
def relation_correct(r2,g1,g2):
	"""relation r2 in g2 is correct if it appears in g1 with at least one point
	in super-nodes related
	eg: ({A2},{D1,E1,F1}) in G2 is correct if ({A2},{B2,D1,F1}) in G1
	"""
	n2_A =g2.nodes()[r2.node1()]
	n2_B =g2.nodes()[r2.node2()]
	for r1 in g1.edges().values():
		n1_A=g1.nodes()[r1.node1()]
		n1_B=g1.nodes()[r1.node2()]
		if match(n2_A,n1_A) and match(n2_B,n1_B) and r2.relation()==r1.relation():
			return True
	return False

# a big bowl of wrong ? 
def relation_score(r2,g1,g2):
	"""relation r2 in g2 is correct if it appears in g1 with at least one point
	in super-nodes related with "before or equals"
	eg: ({A2},{D1,E1,F1}) in G2 is correct if ({A2},{B2,D1,F1}) in G1
	"""
	rel2=r2.relation()
	n2_A =g2.nodes()[r2.node1()]
	n2_B =g2.nodes()[r2.node2()]
	if rel2==Relation(['<','==']):
		# check first if nodes are part of the same supernode in g1
		matched=[x for x in g1.nodes().values() if (n2_A.issubset(x) and n2_B.issubset(x))]
		if matched!=[]:
			return 0.5
	for r1 in g1.edges().values():
		n1_A=g1.nodes()[r1.node1()]
		n1_B=g1.nodes()[r1.node2()]
		rel1=r1.relation()
		if match(n2_A,n1_A) and match(n2_B,n1_B):
			if rel1==rel2:
				return 1.
			else:# relation are necessarily < or <=
				relmin,relmax=sorted([rel1,rel2])
				if (relmin,relmax)==(Relation('<'),Relation(['<','=='])):
					return 0.5
	return 0.

# pb g4/G3: b1,e2 + 

# ok simple
def compare(g1_0,g2_0):
	g1=deepcopy(g1_0)
	g2=deepcopy(g2_0)
	remove_obvious(g2)
	#remove_obvious(g1)
	all2=[x for x in g2.edges().values() if inf_or_equals(x)]
	#res= len([x for x in all2 if relation_correct(x,g1,g2)])
	res= [relation_score(x,g1,g2) for x in all2]
	# print >> sys.stderr, "\n",zip(res,all2)

	return sum(res)


# ok simple
def value_precision(g1,g2,fusion=None):
	""" precision on point-based graph
	g1 and g2 are complete graph, maybe not saturated
	= (g2*/g1* + (nb evt*2 -nb noeuds) - fusion(g1,g2))
	"""
	#g1_bis=close_minGraph(g1)
	#g2_bis=close_minGraph(g2)
	if fusion is None:
		fusion=merge_nb(g1,g2)
	val_prec=compare(g1,g2)+(nb_evt(g2)*2- len(g2.nodes())) - fusion
	#print >> sys.stderr, "val prec g1,g2 (17) ", val_prec # 17
	return val_prec

# ok simple
def pt_precision(g1,g2,fusion=None):
	""" precision on point-based graph g2 with respect to g1
	g1 and g2 are complete graph, maybe not saturated
	= value_precision(g1,g2) / (graphe_value(g2*))
	"""
	val_prec=value_precision(g1,g2,fusion=fusion)
	v_g2star=graph_value(g2)
	
	return float(val_prec),v_g2star


# ok simple
def pt_min_recall(g1,g2):
	"""recall on minimal point-based graph
	
	"""
	g2star=deepcopy(g2)
	g2star.saturate()
	g1min=min_ptGraph(g1)
	nb=len(g1min.edges())
	rec_min=compare(g2star,g1min)

	result=graph_value(g1min)-(nb-rec_min+split_nb(g1,g2))
	
	return result
	
# bug / simple
def pt_comp_recall(g1,g2):
	"""recall on complement point-based graph
	
	"""
	g1star=deepcopy(g1)
	g1star.saturate()
	g1min=min_ptGraph(g1)
	g2min=min_ptGraph(g2)
	comp=[x for x in g1star.edges().values() if not(x in g1min.edges().values()) and relation_correct(x,g2min,g1star)]
	#print >>  sys.stderr,"comp recall",comp
	return len(comp)

# 
def pt_recall(g1,g2,alpha=1.0):
	"""pt recall, weighted average of min-recall and comp-recall
	with alpha as coefficient of comp-recall (alpha<1)

	g1 and g2 are point-based graphs
	"""
	alpha=min(abs(alpha),1.0)
	M=graph_value(min_ptGraph(g1))
	min_rec=pt_min_recall(g1,g2)
	comp_rec=pt_comp_recall(g1,g2)
	if M==0:
		ptrec=0
	else:
		ptrec=(min_rec+(comp_rec)*alpha/M)
	print >> sys.stderr, "pt min rec:", min_rec, comp_rec, M, alpha, ptrec
	return ptrec,M

##�Ajout XT
#def graph_values(g):
#	m = graph_value(g)
#	mmin = graph_value(min_ptGraph(g))
#	return mmin, m



############################################
# extraie les graphes "minimaux"
# a partir d'une annotation
# -> renvoie une liste de graphes
############################################
#def minimal_graph(one_graph):
#	pass


#############################################
# extracts those relations that are in each
# minimal graph
#
# input : a saturated graph
# 
# output : renvoie une liste de relations du
# graphe; if one of them is taken out, then
# we lose info even after resaturation
# of the graph
# Symetric relations: only one of them is kept

# TODO: replace deepcopy with regeneration from relations (faster)
#  ... relations=one_graph.edges().values()
#  ... other=make_graph(relations,index=False)
#  ... other.del_edge(edge)
###############################################
def get_kernel_old(one_graph):
	result=[]
	for edge in one_graph.edges().values():
		# should not copy the index ... or problem!
		other=copy.deepcopy(one_graph)
		# temporary fix for previous pb
		other._index=False
		#
		other.del_edge(edge)
		other.saturate()
		#print >> sys.stderr, "done one edge"
		if other.edges()==one_graph.edges():
			#if not(edge.inverse(one_graph.algebra()) in result):
			result.append(edge)
	
	kg=CSPGraph(one_graph.algebra(),index=False)
	for one in result:
		kg.addEdge(one)
	return kg

# revised version : faster (60% of version 1)
def get_kernel(one_graph):
	result=[]
	relations=one_graph.edges()
	try:
		itsalgebra=relations.values()[0].relation().algebra()
	except:
		itsalgebra=allen_algebra
	if itsalgebra is None:
		itsalgebra=allen_algebra
	for edge in relations.values():
		other=make_graph_from_rels(relations,algebra=itsalgebra,index=False)
		other.del_edge(edge)
		other.index_on_node1()
		other.saturate()
		#print >> sys.stderr, "done one edge"
		if other.edges()==one_graph.edges():
			#if not(edge.inverse(one_graph.algebra()) in result):
			result.append(edge)
	
	kg=CSPGraph(one_graph.algebra(),index=False)
	for one in result:
		kg.addEdge(one)
	return kg





#######################################################################
# generic measure function
#   - h: human graph
#   - s: system graph
#   - algebra used 
#   - type: which set of relations within the algebra
#        only allen/bruce/tempeval
#           (TODO: "base" vs. user-defined set)
#   - filter on arcs to compare (default: none)
#     Warning: There is already a filter on edges bearing information only (for H or S)
#   - comparison measure between two relations(defaut: 1 if equality)
# todo - ponderation: fonction de ponderation de la mesure sur chaque arc
#                  (relation arc1,relation arc2) -> float
#                  defaut: (a1,a2)->1 (pas de ponderation)
#             score=somme(ponderation*score)/somme(ponderation)
# ex:
#  - finesse sur arc de H ou S : filtres:defauts,  compare=relation_finesse
#  - coherence sur les arcs informatifs pour H OU S: filtres:defauts,  compare=relation coherence
#    coherence sur les arcs informatifs pour H seulement: filtreS=(lambda x: False), compare=relation coherence
#  - single relation precision: filtreS=(lambda x: False)
#                               filtreH=(lambda x: (len(x.relation())==1)))
#                               compare: defaut
#  - relation event-timex seulements:
#     n�cessite des filtres sur arcs suivants le type de noeud ... cf Graphe/Relation
#    
########################################################################
def generic_measure(h,s,algebra,type="allen",
		    filtreS=(lambda x: True),
		    filtreH=(lambda x: True),
		    ponderation=(lambda x,y: 1.0),
		    compare=(lambda x,y: int(x==y))):
	universal=h.algebra().universal()
	if type!="allen":
		universal=universal.allen2other(type)
	arcs_h = h.edges()
        arcs_s = s.edges()
	id_arcs_h = [x for (x,a) in arcs_h.items() if filtreH(a)]
	id_arcs_s = [x for (x,a) in arcs_s.items() if filtreS(a)]
	# 
	# all arcs  either in H or S graph
        #print  >> sys.stderr, "H:",id_arcs_h
        #print  >> sys.stderr, "S:",id_arcs_s
	all_arcs=set(id_arcs_h+id_arcs_s)
	#all_arcs=[nodes for (nodes,edge) in all_arcs]
	
	score=0
	cpt=0
	for (e1,e2) in all_arcs:
		# only on ordered pairs to avoid counting twice the same edge
		if e1<e2:
			
			arc_h = arcs_h.get((e1,e2))
			arc_s = arcs_s.get((e1,e2))
			if arc_s is None:
				rel2=universal
				#print >> sys.stderr, ">pas d'arc S pour H=%s" % `arc_h`
			else:
				rel2=arc_s.relation()
				if type!="allen":
					rel2=rel2.allen2other(type)
			if arc_h is None:
				#print >> sys.stderr,">pas d'arc H pour H=%s, S=%s" % (`arc_s`,`rel2`)
				rel1=universal
			else:
				rel1=arc_h.relation()
				if type!="allen":
					rel1=rel1.allen2other(type)
			poids=ponderation(rel1,rel2)
			cpt+=poids
			#print >> sys.stderr,"H=%s, S=%s,"%(`rel1`,`rel2`)
			measure=poids*compare(rel1,rel2)
			#print >> sys.stderr," score=%f"%measure
			score=score+measure
	return score,cpt#len(arcs_h)



def transform2pt(g0):
   gpt=interval2point(g0)
   gpt.saturate()
   gpt=fuse_nodes(gpt)
   gmin=min_ptGraph_all(gpt)	
   return gpt,gmin


# wip
# TODO: future: class Mesure
_mesures_names=['FINAllen', 'COHAllen',
               'FINAR', 'COHAR',
               'PREC', 'RECA',
               'PRECG', 'RECAG',
               'JAQ_AR',
               'FINAREE', 'COHAREE',
               'COMPATAR',
               'PRECEE', 'RECAEE',
                'RECNOY','PRECNOY','PREC_PTGRAPH','REC_PTGRAPH','REC_PTGRAPH_SIMPLE']
_mesures={}
for (i,x) in enumerate(_mesures_names):
    _mesures[x]=i

# for each measure, three values
mesures_names=_mesures_names
table_row_labels=zip(_mesures_names,_mesures_names)
table_row_labels=[(x,y,z) for ((x,y),z) in zip(table_row_labels,_mesures_names)]
table_row_labels=reduce((lambda x,y: x+y),table_row_labels)


# one_mesure: abbrev (FINAllen)
# retourne la valeur stock�e dans result (liste crach�e par make_stats)
def get_mesure(result,one_mesure):
    return result[_mesures[one_mesure]]

# mesures cumul�es entre 2 graphes
# cached result : g_noyau du graphe 1, g on interval extremities du graphe 1 
def make_stats( g1, g2, g_noyau=None,gpt1=None,gmin1=None):
    # TODO: nb de noeuds sur chaque graphe + nb arcs non triviaux + densit� des arcs
 
    results=[]
    # finesse sur arcs informatifs H seulement
    # pas d'info H et S -> pas de mesure
    results.append(generic_measure(g1,g2,g1.algebra(),
                                   compare=relation_finesse,filtreS=(lambda x: False)))
    print >> sys.stderr, "FINAllen, finesse allen: %f/%f"%results[-1]
    # coherence sur arcs informatifs pour S seulement
    results.append(generic_measure(g1,g2,g1.algebra(),
                                   compare=relation_coherence,filtreH=(lambda x: False)))
    print >> sys.stderr, "COHAllen, coherence allen: %f/%f"%results[-1]
    #-----------------------------------------------
    # memes chose avec relations d'annotations
    #-----------------------------------------------------
    results.append(generic_measure(g1,g2,g1.algebra(),
                                   compare=relation_finesse,
                                   filtreS=(lambda x: False),type="bruce"))
    print >> sys.stderr, "FINAR, finesse annotation relations: %f/%f"%results[-1]
    results.append(generic_measure(g1,g2,g1.algebra(),
                                   compare=relation_coherence,
                                   filtreH=(lambda x: False),type="bruce"))
    print >> sys.stderr, "COHAR, coherence annotation relations: %f/%f"%results[-1]
    # precision / rappel relations simples 
    # test� sur
    # accord entre graphe non satur�- graphe satur� (prec doit etre 100% et rappel inferieur)
    #precision
    results.append(generic_measure(g1,g2,g1.algebra(),type="allen",
                                   filtreH=(lambda x: False),
                                   filtreS=(lambda x: (len(x.relation())==1))))
    print >> sys.stderr, "PREC, prec. single ann. relations: %f/%f"%results[-1]
    #rappel
    results.append(generic_measure(g1,g2,g1.algebra(),type="allen",
                                   filtreS=(lambda x: False),
                                   filtreH=(lambda x: (len(x.relation())==1))))
    print >> sys.stderr, "RECA, recall single ann. relations: %f/%f"%results[-1]
    # meme chose, avec mesure graduelle (finesse pour rappel, coherence pour precision)
    results.append(generic_measure(g1,g2,g1.algebra(),type="bruce",compare=relation_coherence,
                                   filtreH=(lambda x: False),
                                   filtreS=(lambda x: (len(x.relation().allen2bruce())==1))))
    print >> sys.stderr, "PRECG, prec. single ann. relations: %f/%f"%results[-1]
    #rappel
    results.append(generic_measure(g1,g2,g1.algebra(),type="bruce",compare=relation_finesse,
                                   filtreS=(lambda x: False),
                                   filtreH=(lambda x: (len(x.relation().allen2bruce())==1))))
    print >> sys.stderr, "RECAG, recall single ann. relations: %f/%f"%results[-1]
    
    
    # rappel pour  relations seulement entre evt et timex avec relation=inclus (anchoring)
    # il faudrait en fait chercher combien d'evt sont correctement ancr�s temporellement
    #=generic_measure(g1,g2,g1.algebra(),filtreS=(lambda x: False),filtreH=(lambda x: (x.relation().allen2bruce()==Relation("arel_included")) and arc_type(x,"event","timex")),type="bruce")
    #precision sur relations seulement entre evt et timex avec relation=inclus (anchoring)
    # generic_measure(g2,g1,g1.algebra(),filtreS=(lambda x: False),filtreH=(lambda x: (x.relation().allen2bruce()==Relation("arel_included")) and arc_type(x,"event","timex")),type="bruce")
    #
    
    # jaquard
    results.append(generic_measure(g1,g2,g1.algebra(),compare=jaquard,type="bruce"))
    print >> sys.stderr, "JAQ_AR, jaquard sur annotation relations: %f/%f"%results[-1]
    # event/event relations sur tous les arcs
    results.append(generic_measure(g1,g2,g1.algebra(),compare=relation_finesse,
                    filtreS=(lambda x: arc_type(x,"event","event")),type="bruce"))
    print >> sys.stderr, "FINAREE, finesse event-event annotation relations: %f/%f"%results[-1]
    results.append(generic_measure(g1,g2,g1.algebra(),compare=relation_coherence,
                    filtreH=(lambda x: arc_type(x,"event","event")),type="bruce"))
    print >> sys.stderr, "COHAREE, coherence event-event annotation relations: %f/%f"%results[-1]
    
    # compatibilit� : H implique S 
    results.append(generic_measure(g1,g2,g1.algebra(),compare=relation_compatibilite,type="bruce"))
    print >> sys.stderr, "COMPATAR, : %f/%f"%results[-1]
    
    # event-event single relations
    #precision
    results.append(generic_measure(g1,g2,g1.algebra(),type="bruce",
                                   filtreH=(lambda x: False),
                                   filtreS=(lambda x: arc_type(x,"event","event") and (len(x.relation().allen2bruce())==1))))
    print >> sys.stderr, "PRECEE, event-event prec. single ann. relations: %f/%f"%results[-1]
    #rappel
    results.append(generic_measure(g1,g2,g1.algebra(),type="bruce",
                                   filtreS=(lambda x: False),
                                   filtreH=(lambda x: arc_type(x,"event","event") and (len(x.relation().allen2bruce())==1))))
    print >> sys.stderr, "RECAEE, event-event recall single ann. relations: %f/%f"%results[-1]

   
    if g_noyau is not None:
	pass
    else:
	print >> sys.stderr, "kernel computed again?"
	g_noyau=get_kernel(g1)
    g_noyauS=get_kernel(g2)
    # non satur� ... on veut savoir les relations importantes;
    # ici rappel des relations noyaux de l'annotation humaine
    results.append(generic_measure(g1,g2,g1.algebra(),
				   filtreS=(lambda x: False),
				   filtreH=(lambda x: (x in g_noyau.edges().values()))))
    print >> sys.stderr, "RECNOY, allen rels %f/%f"%results[-1]
    # pr�cision des relations du noyau du systeme
    results.append(generic_measure(g1,g2,g1.algebra(),
				   filtreS=(lambda x: (x in g_noyauS.edges().values())),
				   filtreH=(lambda x: False)))
    print >> sys.stderr, "PRECNOY, allen rels %f/%f"%results[-1]


    # mesures sur graphes de points �quivalent
    if gpt1 is not None:
        pass
    else:
        print >> sys.stderr, "gpt computed again?"
        gpt1,gmin1=transform2pt(g1)
    gpt2,gmin2=transform2pt(g2)
    results.append(pt_precision(gpt1,gpt2))
    print >> sys.stderr, "PREC_PTGRAPH, precision on point graph %f/%f"%results[-1]

    results.append(pt_recall(gpt1,gpt2))
    print >> sys.stderr, "REC_PTGRAPH, recall on point graph %f/%f"%results[-1]

    
    results.append(pt_recall(gpt1,gpt2,alpha=0.))
    print >> sys.stderr, "REC_PTGRAPH_SIMPLE, simple recall on point graph %f/%f"%results[-1]


    ##�Ajout XT
#    results.append(graph_values(gpt2))
#    print >> sys.stderr, "VALUES"
    # une seule ligne bilan pour reimport direct dans un tableur
    # 
    # FINAllen finesse allen
    # COHAllen, coherence allen
    # FINAR, finesse annotation relations
    # COHAR, coherence annotation relations
    # PREC, prec. single ann. relations 
    # RECA, recall single ann. relations
    # PRECG, prec. single ann. relations (mesure graduelle)
    # RECAG, recall single ann. relations (mesure graduelle)
    # JAQ_AR, jaquard sur annotation relations
    # FINAREE, finesse event-event annotation relations
    # COHAREE, coherence event-event annotation relations
    # COMPATAR : H -> S
    # NBARCS_H NBARCS_S FINAllen COHAllen FINAR COHAR PREC RECA PRECG RECAG JAQ_AR FINAREE COHAREE COMPATAR PRECEE RECAEE
    # TODO: len(g1.nodes()),len(g2.nodes())
  
    return  results




###################################################################################
# Tests preliminaires / Visualisation, cf aussi separate unit testing
########################################
def process(filein,display=False):
    """- reads a event graph description, computes point graph and minimum graph
    - returns pt graph and min graph
    - save min graph in graphviz format
    - optionally display the graph"""
    g0=read_graphe(filein+".graph",allen_algebra)
    gpt=interval2point(g0)
    gpt.saturate()
    gpt=fuse_nodes(gpt)
    gmin=min_ptGraph_all(gpt)
    if display:
	    fout=open(filein+".dot","w")
	    print >> fout, gmin.graphviz_output(orientation="landscape",
						rel_filter =lambda x: inf_or_equals(x),
						node_filter=lambda x: x in gmin.nodes(),
						edge_format=lambda x: format_rel(x),
						node_format=lambda x: "|".join([y for y in x]))
	    fout.close()
	    display_graph(filein)
    #gmin=remove_obvious(gmin)
    #pprint(gmin)
    return gpt,gmin


def test_simple():
	a=AllenGraph(allen_algebra)
	c1=Edge("1","2","trel_before")
	c2=Edge("2","3","trel_equals")
	a.addNode("1",{})
	a.addNode("2",{})
	a.addNode("3",{})
	a.add(c1)
	a.add(c2)
	#a.saturate()
	print "point graphe from interval graph"
	b=interval2point(a)
	b.saturate()
	print b
	c=point2interval(b)
	print c
	return a,b,c

def test_disj():
	a=AllenGraph(allen_algebra)
	c1=Edge("1","2","trel_before")
	c2=Edge("1","3","trel_overlap")
	a.addNode("1",{})
	a.addNode("2",{})
	a.addNode("3",{})
	a.add(c1)
	a.add(c2)
	a.saturate()
	print "point graphe from interval graph"
	b=interval2point(a)
	#b.saturate()
	print b
	c=point2interval(b)
	print c
	return a,b,c


#if __name__=="__main__":
#    from pprint import pprint
#    from graph.Graph import allen_algebra, Edge
#    # � ajuster au besoin
#    dot_path          = "/usr/bin/dot"
#    image_viewer_path = "/usr/bin/display"
#    import os
#    def format_rel(rname):
#	    if rname==Relation('<'):
#		    return "<"
#	    elif rname==Relation(['<','==']):
#		    return "<="
#	    else:
#		    return "?"
#	    
#    def display_graph(name):
#	    os.spawnv(os.P_NOWAIT,dot_path,["dot","-Tps",name+".dot","-o",name+".ps"])
#            os.spawnv(os.P_NOWAIT,image_viewer_path,["display",name+".ps"])
#
#    def save_and_display(g,name):
#	    fout=open(name+".dot","w")
#	    print >> fout, g.graphviz_output(orientation="landscape",
#					     rel_filter =lambda x: len(x.relation())==1 and not(list(x.relation())[0].endswith('i')),
#					     node_format=lambda x: `x`)
#	    fout.close()
#	    display_graph(name)
#
#    if 0:#simple test
#	    a=AllenGraph(allen_algebra)
#	    c1=Edge("1","2","trel_before")
#	    c2=Edge("1","3","trel_equals")
#	    a.addNode("1",{})
#	    a.addNode("2",{})
#	    a.addNode("3",{})
#	    a.add(c1)
#	    a.add(c2)
#	    #a.saturate()
#	    print "point graphe from interval graph"
#	    b=interval2point(a)
#	    b.saturate()
#	    print b
#	    print "point graph with equal nodes merged"
#	    c=fuse_nodes(b)
#	    print c
#	    print "point graph with only minimal relations" 
#	    d=min_ptGraph(c)
#	    print d
#	    
#	    fout=open("test.dot","w")
#	    print >> fout, d.graphviz_output(orientation="portrait",node_format=lambda x: "|".join([y for y in x]))
#	    fout.close()
#
#    # allen neighborhood graph
#    #allen_NG=Graph()
#    #allen_NG.read_from_file("allen_NG.graph")
#    #print >> sys.stderr, allen_NG.get_edge('b','m') is not None
#    if 0:
#	# test graphe1 from paper
#	gpt1,g1min=process("test1",display=False)
#	gpt2,g2min=process("test2",display=False)
#	fusion,split=merge_nb(gpt1,gpt2),merge_nb(gpt2,gpt1)
#	print >> sys.stderr, "fusion/split", fusion,split, "=(2,4)" #ok
#	print >> sys.stderr, "value g1min", graph_value(g1min), "= 8"# (ok)
#	print >> sys.stderr, "prec g2/g1\n----\n ", pt_precision(gpt1,gpt2), "=0.607\n----" #not(ok)
#	print >> sys.stderr, "value precision g2", graph_value(g2min), "=17", # not ok
#	
#	gpt3,g3min=process("test3",display=True)
#	gpt4,g4min=process("test4",display=True)
#
#
#    def test_rel(gpt1,gpt2):
#	    g2=(gpt2)
#	    g1=(gpt1)
#	    remove_obvious(g2)
#	    remove_obvious(g1)
#	    print ""
#	    count=0
#	    for one in [x for x in g2.edges().values() if x.relation()==Relation('<')]:
#		    a=relation_correct(one,g1,g2)
#		    print one, a
#		    if a:
#			    count +=1
#	    return count
#    #print test_rel(gpt1,gpt2)
#    
#   
#    def make_fromtimeml(filename,display=False):
#	base1=open(filename).read()
#	data1,t1=timeml.extract_from_string(base1)
#	data1bis=normalise(data1)
#	g1=make_graph(data1bis)
#	gpt=interval2point(g1)
#	gpt.saturate()
#	gpt=fuse_nodes(gpt)
#	gmin=min_ptGraph_all(gpt)
#	if display:
#		fout=open(filename+".dot","w")
#		print >> fout, gmin.graphviz_output(orientation="landscape",
#						    rel_filter =lambda x: inf_or_equals(x),
#						    node_filter=lambda x: x in gmin.nodes(),
#						    edge_format=lambda x: format_rel(x),
#						    node_format=lambda x: "|".join([y for y in x]))
#		fout.close()
#		display_graph(filename)
#	return gpt, gmin
#
#
#    import read_timeml_20 as timeml
#    from timeml_compare import disturb_variantes, variantes, normalise, make_graph, table_row_labels
#    filename=sys.argv[1]
#    f1="CNN19980222.1130.0084_reduced.tml.xml"
#    f2="CNN19980222.1130.0084.tml.xml"
#    gpt1,gmin1=make_fromtimeml(f1)
#    gpt2,gmin2=make_fromtimeml(f2)
#    print "recall", pt_recall(gpt2,gpt1)
#    print "precision", pt_precision(gpt2,gpt1)

if __name__=="__main__":
	from graph.Graph import allen_algebra, Edge
	from optparse import OptionParser
	
	parser = OptionParser()
	parser.add_option("-a", "--ascii-format",dest="ascii_format",action="store_true",default=False,
		help="load graphs in ascii simple description")
	parser.add_option("-l", "--label",dest="label",action="store_true",default=False,
		help="add labels to results")
	(options, args) = parser.parse_args()
	
	if options.ascii_format:
		g1=read_graphe(args[0],allen_algebra)
		g2=read_graphe(args[1],allen_algebra)
	else:
		g1=timeml2graph(args[0])
		g2=timeml2graph(args[1])
	##�XT added nbrel1 and nbrel2 : number of relations in graphs 1 and 2 before closure
	nbrel1=len(g1.edges())/2
	nbrel2=len(g2.edges())/2
	ok1=g1.saturate()
	ok2=g2.saturate()
	if not(ok1 and ok2):
		print >> sys.stderr, "inconsistent graph"
		sys.exit(0)
	results = make_stats( g1, g2)
	if options.label:
		##�XT added nbrel1_b and nbrel2_b : number of relations in graphs 1 and 2 before closure
		print " ".join(["--","nbrel1_b","nbrel2_b","nbrel1","nbrel2"]+["%s %s %s"%(x,x,x) for x in mesures_names])
	##�XT added nbrel1 and nbrel2
	print " ".join(["Bilan", `nbrel1`, `nbrel2`, `len(g1.edges())/2`,`len(g2.edges())/2`]+["%f %f %f"%(x,y,1.0*x/(max(0.00001,y))) for (x,y) in results])#.replace(".",",")
