#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
###############################################
""" 
class for graphs, relational algebra graph,
and Allen algebra graphs

provides:
     
     - class for edge
     - class for graph with annotated arcs
     - subclass for allen-based temporal graph
""" 

#TODO:
#  ok - conversion graphe->tlink
#  ~ok - validation des inferences (bug possible ? sur egalit�par exemple)
#        BUG confirm� pb de propagation en pr�ence de l'�alit�(tester la table)
#      Relation : ok, compo. marche avec equals
#      Graphe   : ?
#  - ameliorer propagation
#     ok - refactoring: fct revise arc + iteration avec file d'attente
#            ok? * revise incorpore test d'enrichissement (au lieu de l'iteration)
#            ok  * iterer sur les arcs i,j puis sur la compo k,i,j
#                  ou i,j,k au lieu de decomposer i,j en i,k,j
#		
#              * file mise a jour (evite de retester des arcs inchang�)
#        (ne change rien)    - ? algo PCIA2 PCIA3 cf. livre sur Raist Sp.Tpl
#  - bottlenecks: recup des edges -> indexation par premier evt serait plus efficace
#                 deepcopy quand on veut sauver le graphe: prohibitif 
#        
##############################################

# inutile ?
#from sets import Set as set

import copy
import sys
import numpy
import collections

from Relation import Relation, Algebra, rel_name, allen_rel, inverse
try:
    import igraph
    import igraph.drawing as draw
except:
    print >> sys.stderr, "igraph needed for visualization"

##################################################
# Edge orient�pour graphe temporel
# noeud: _x, _y : juste l'id du noeud dans le graphe
# arcs : id1, id2, rel
# rel is info on edge (a set of relations)
#
##################################################
class Edge:
    """directed edge graph for temporal graph

    an edge is a temporal relation between two edges (cf Relation module)
    """
    def __init__(self,node1,node2,rel):
        self._x=node1
        self._y=node2
        if isinstance(rel,str):
            self._rel=Relation(rel)
        else:
            self._rel=rel

    def node1(self):
        """source of the relation"""
        return self._x

    def node2(self):
        """target of the relation"""
        return self._y
    

    def __eq__(self,other):
        cond= (self.node1()==other.node1()) and (self.node2()==other.node2()) and (self.relation()==other.relation())
        return cond
        
    
    def relation(self):
        return self._rel

    def set_rel(self,rel):
        self._rel=rel

    def rel(self):
        return self._rel

    def __repr__(self):
        rel=", ".join([x for x in self.rel()])
        return "%s, %s : [%s]\n"%(self.node1(),self.node2(),rel)

    # only methods specific to relational constraint graph
    def inverse(self,algebra):
        """converse of the relation in the given algebra"""
        return Edge(self.node2(),self.node1(),algebra.inverse(self.rel()))

    def is_false(self):
        """inconsistent relation"""
        return len(self.rel())==0




class Graph:
    """generic class for graph with annotated edges
    """
    def __init__(self):
        """nodes are listed with [fac.] associated info"""
        self._nodes={}
        self._arcs={}

    def __repr__(self):
        result=""
        #return "%s\n%s"%(`self._nodes`,`self._arcs`)
        for x in self._nodes.keys():
            result+=(x+"\n")
        for x in self._arcs.values():
            # on n'affiche pas les arcs symetriques
            #if x.node1()<x.node2():
            result+=`x`
        return result
    

    def nodes(self):
        return self._nodes
    
    def edges(self):
        return self._arcs

    def relation_dict(self):
        """ returns dictionary with node identifiers as keys and relation as values """
        return dict( [ ((e.node1(),e.node2()),e.relation()) for e in self.edges().values()] )


    def __eq__(self,other):
        return (self._nodes==other._nodes) and (self._arcs==other._arcs)
        

    def addNode(self,node,properties={}):
        """add a node to the graph, with optional properties"""
        self._nodes[node]=properties

    # edge indexed by argument pair
    # ? better by left-node then right ? and/or inverse index ?
    def addEdge(self,arc):
        self._arcs[(arc.node1(),arc.node2())]=arc

    def del_edge(self,arc):
	"""remove an edge"""
        (a1,a2)=arc.node1(),arc.node2()
        if self._arcs.has_key((a1,a2)):
            del self._arcs[a1,a2]

                        
    def normalise(self,string):
            return string.replace("'","_")


    def read_from_file(self,filename):
        """init from file with
        relation node1 node2
        on each line
        """
        fin=[x.strip().split() for x in open(filename).readlines() if x.strip()!=""]
	for r,n1,n2 in fin:
		self.addNode(n1,properties=n1)
		self.addNode(n2,properties=n2)
		self.addEdge(Edge(n1,n2,r))



    def get_edge(self,n1,n2):
        return self.edges().get((n1,n2),None)

    def graphviz_output(self,
                        node_format=lambda x: x,
                        edge_format=lambda x: `x`,
                        node_filter=lambda x: True,
                        rel_filter=lambda x:True,
                        orientation="landscape"):
        """ output in dot format, with optional filters on nodes or edges
        """
        result=[]
        result.append( 'digraph G {')
        result.append( '\tgraph [orientation=%s];' % (orientation))
        result.append( '\tnode [shape=Mrecord];')
        #result.append( '\tedge [fontsize=10];')
        result.append( "\n")
        
        for one in self.nodes():
            if node_filter(one):
                #normalise names
                nodename=self.normalise(one)
                result.append('\t%s [shape=Mrecord,label="{%s}"]'%(nodename,self.normalise(node_format(self.nodes()[one]))))

        for rel in self.edges().values():
            if rel_filter(rel):
                n1=self.normalise(rel.node1())
                n2=self.normalise(rel.node2())
                result.append('\t%s -> %s [label="%s"]'%(n1,n2,edge_format(rel.relation())))
        result.append( "}")
        return "\n".join(result)

###################################################
class CSPGraph(Graph):
    """binary constraint graph, defined with respect to an algebra of relations on its edges"""
    def __init__(self,algebra,index=True):
        # nodes are listed with [fac.] associated info
        self._nodes={}
        self._arcs={}
        self._algebra=algebra
        self._index=False
        # make it faster !
        if index:
            self.index_on_node1()

    def constraints(self):
        return self._arcs

    def index_on_node1(self):
        self._index=True
        self._indexcst1=collections.defaultdict(list)
        for (n1,n2) in self.constraints():
            self._indexcst1[n1].append((n1,n2))


    def algebra(self):
        return self._algebra
    
    # does a simple check 
    def inconsistent(self):
        for cs in self.constraints().values():
            if cs.is_false():
                return True
        return False

    
    # rapport=tout: rapporte au nb d'arcs possibles (nn-1/2)
    # autrement: seulement arcs avec info 
    def info(self,rapport="tout"):
	"""measure of information density in the graph
        as specificity of relations, wrt to all possible edges
        or only edges with non trivial relations
        """
        la=[1.0/len(x.relation()) for x in self.edges().values()]
        if la==[]:
            return 0
        if rapport=="tout":
            n=len(self.nodes().keys())
            rapport=n*(n-1)*0.5
        else:
            rapport=0.5*len(la)
        return 0.5*sum(la)/rapport
    
##    def cs_with_node1(self,arg):
##        res=[]
##        graph=self.constraints()
##        for c in graph.values():
##            c1,c2=c.node1(),c.node2()
##            if c1==arg:
##                res.append(graph[(c1,c2)])
##        return res

    # filter instead ?
    def cs_with_node1(self,arg):
	"""constraints whose first node is arg """
        graph=self.constraints()
        if self._index:
            return [graph[x] for x in self._indexcst1[arg]]
        else:
            res=[]
            #graph=self.constraints()
            for (c1,c2),c in graph.items():
                #c1,c2=c.node1(),c.node2()
                if c1==arg:
                    res.append(graph[(c1,c2)])
            return res

    # filter instead ?
    def cs_with_node2(self,arg):
	"""constraints whose second node is arg """
        res=[]
        graph=self.constraints()
        for c in graph.values():
            c1,c2=c.node1(),c.node2()
            if c2==arg:
                res.append(graph[c1,c2])
        return res
        
    def cs_with_args(self,node1,node2):
        return self.constraints().get((node1,node2),None)

    # unsafe: adding an arc should check if it's not there already
    # TODO: rename, as this is confusing
    def add(self,arc):
	"""add an edge to the graph, and the reverse edge wrt the algebra """
        (a1,a2)=arc.node1(),arc.node2()
        self._arcs[a1,a2]=arc
        self._arcs[a2,a1]=arc.inverse(self.algebra())

        if self._index:
            self._indexcst1[a1].append((a1,a2))
            self._indexcst1[a2].append((a2,a1))
        #print "added:", arc
        #print "added:", arc.inverse(self._algebra)

    # TODO: rename too
    def addEdgeList(self,edgeList):
        """add a list of edge to the graph 
        """
        for one in edgeList:
            self.add(one)

    def del_edge(self,arc):
	"""remove an edge"""
        (a1,a2)=arc.node1(),arc.node2()
        if self._arcs.has_key((a1,a2)):
            del self._arcs[a1,a2]
        if self._arcs.has_key((a2,a1)):
            del self._arcs[a2,a1]
        if self._index:
            self._indexcst1[a1].remove((a1,a2))
            self._indexcst1[a2].remove((a2,a1))

    # a tester
    def change_edge(self,arc,new_relation):
	"""change the relation assigned to a given edge"""
        (a1,a2)=arc.node1(),arc.node2()
        if self._arcs.has_key((a1,a2)):
            self._arcs[a1,a2].set_rel(new_relation)
        if self._arcs.has_key((a2,a1)):
            self._arcs[a2,a1]=self._arcs[a1,a2].inverse(self.algebra())
  
    # correspond a la fonction REVISE dans les algos classiques de propagation
    # -> devrait etre une methode d'arc, propageant une methode de relation
    def update(self,old_constraint,new):
	"""revise an edge label by intersecting with another relation"""
        r1=old_constraint.relation()
        r2=new.relation()
        r=r1 & r2
        # inutile si test�avant dans propagate
        #if r==r1:
        #    changed=False
        #else:
        changed=True
        # replaces old one with same arg1,arg2
        new.set_rel(r)
        self.add(new)
        # indique une inconsistence
        if len(r)==0:
            return (changed,False)
        else:
            return (changed,True)
        

    # enforce path_consistency with composition table
    # (more or less naive algo -> bottleneck)
    # return (consistent,changed)
    # false if no change, true if sth changed
    # false if inconsistent, true otherwise
    # TODO: 
    #    ok- changer arc+symetrique en meme temps et changer arc ordonn�
    #    - algo plus efficaces
    def propagate(self):
	"""propagate constraints through the graph once (enforce path_consistency)
	return (consistent,changed) (booleans)
	"""
        changed=False
        consistent=True
        a=self.algebra()
        for (node1,node2),c in self.constraints().items():
            #print "---------\n arc1 -:",c
            node2=c.node2()
            related=self.cs_with_node1(node2)
            relation1=c.relation()
            #print "compose avec --:",related
            for c2 in related:
                c2n2=c2.node2()
                if node1!=c2n2:
                    #print "arc2 -----",c2
                    target=self.cs_with_args(node1,c2n2)
                    res=a.compose(relation1,c2.relation())
                    #print "compo->",node1,c2n2, res
                    #print res
                    if target is None and res!=a._universal:
                        changed=True
                        #print "new constr"
                        #print (node1,c2n2,res)
                        self.add(Edge(node1,c2n2,res))
                    elif target is not None and not(target.relation().issubset(res)):
                    # evite de faire le test apres l'intersection
                    #elif target!=None and len(res)<13:
                        #print target
                        (chg,cons)=self.update(target,Edge(node1,c2n2,res))
                        if not(cons):
                            consistent=False
                        if chg:
                            changed=True
                    else:
                        pass
        return (consistent,changed)


    def saturate(self):
	"""propagate constraints until a fixpoint is reached on the graph
        or an inconsistency is found
        """
        cons=True
        changed=True
        while (cons) and (changed):
            (cons,changed)=self.propagate()
        return cons


   
    def remove(self,other):
	"""removes a set of edges, given by another graph"""
        for one in other.edges().values():
            self.del_edge(one)


    def save_state(self):
        """flag every edge to restore it after an inconsistent update """
        


    def export2igraph(tgraph,display="default",directed=False):
        """
        translates temporal graph as something usable by igraph,
        with an edge between two nodes only if there is a simple
        relation between the nodes (eg before), => "default" display mode

        "all" mode means all non trivial edges are marked;
        simple are marked in three colors, disjunction in red
        green for inclusion relations
        orange for before/after relations
        blue for overlaps
        
        beware that 'all/defautl' does not influence connectivity since two nodes in different
        strong components will remain disconnected after saturation of the graph

        if directed if True, the graph is directed and an arrow will show
        the edge direction 

        """
        ig=igraph.Graph(directed=directed)    
        mapping=dict(enumerate(tgraph.nodes()))
        inv_mapping=dict([(x,i) for (i,x) in enumerate(tgraph.nodes())])
        if len(mapping)>0:
            ig.add_vertices(len(mapping)-1)
        else:
            print >> sys.stderr, "warning, no nodes in graph", tgraph
            return ig
        for v in ig.vs:
            v["label"]=mapping[v.index]
            v["node"]=copy.deepcopy(tgraph.nodes()[mapping[v.index]])
        for ((e1,e2),the_edge) in tgraph.edges().items():
            r=the_edge.relation()
            idx1,idx2=(inv_mapping[e1],inv_mapping[e2])
            # ordering nodes so that only one edge is drawn and not its converse
            if len(r)==0:
                r=["inconsistent"]
            if list(r)[0] in ["<","=="] or list(r)[0]=="inconsistent":
                #print r, _color.get(hash(list(r)[0]))
                if display=="default" and len(r)==1:
                    ig.add_edges((idx1,idx2))
                    ig.es[ig.get_eid(idx1,idx2)]["relation"]=the_edge
                    ig.es[ig.get_eid(idx1,idx2)]["color"]=_color.get(list(r)[0],"red")
                elif display=="all":
                    if not(r==tgraph.algebra()._universal):
                        ig.add_edges((idx1,idx2))
                        ig.es[ig.get_eid(idx1,idx2)]["relation"]=the_edge
                        if len(r)==1:
                            edgecolor=_color.get(list(r)[0],"red")
                        else:
                            edgecolor="red"
                        ig.es[ig.get_eid(idx1,idx2)]["color"]=edgecolor
                else:
                    pass
        return ig




class CSPGraphMatrix:
    """version using numpy arrays, for not-too-big graphs

    !!!restriction!!!: that the number of nodes in the graph is fixed in advance
    to keep the same interface for all graphs, this number defaults to 100.

    TODO:
       - composition of relations is bottleneck and is not done
          in numpy -> find a way of pushing it as a ufunc or matrix operation
       - saturation as a matrix operation is doing O(n^3) compositions event with sparse graphs
         use sparse matrix module from numpy ?
    """

    def __init__(self,algebra,entity_nb=100):
        universal=algebra.universal().compile()
        self._matrix = numpy.array([universal]*entity_nb**2,dtype="int16")
        self._matrix=self._matrix.reshape(entity_nb,entity_nb)
        self._entity_nb=entity_nb
        self._nodes={}
        self._arcs={}
        self._algebra=algebra
        self._universal=universal

    def algebra(self):
        """returns the algebra the graph is built on"""
        return self._algebra

    def nodes(self):
        return self._nodes
    
    # does a simple check, do not saturate
    def inconsistent(self):
        return len(self._matrix.nonzero()[0])==0

    def convert(self,onegraph):
        """fills the graph once for all with data from other graph of type CSPGraph
        """
        if len(onegraph.nodes()) > self._entity_nb:
            print >> sys.stderr, "error: more nodes than planned in graph"
            sys.exit(0)
        else:
            i=0
            for one in onegraph.nodes():
                #don't keep node information, only its index in the matrix
                self._nodes[one]=i
                i=i+1

            for one in onegraph.constraints().values():
                e1,e2 = one.node1(),one.node2()
                r = one.relation()
                r.set_algebra(self._algebra)
                r=r.compile()
                i,j= self._nodes[e1],self._nodes[e2]
                self._matrix[i][j]=r

    def cs_with_args(self,node1,node2):
        r=self._matrix[self.nodes()[node1]][self.nodes()[node2]]
        universal=self._algebra._universal
        r=universal.from_compiled(r)
        return r
    

    def export(self):
        """export back to a CSPGraph object"""
        g=CSPGraph(self.algebra())
        
        for n1 in self.nodes():
            g.addNode(n1,n1)
            
        for n1,i1 in self.nodes().items():
            for n2,i2 in self.nodes().items():
                r=self._matrix[i1][i2]
                if r!=self._algebra._universal_compiled:
                    r=self._algebra._relations.from_compiled(r)
                    r=Edge(n1,n2,r)
                    g.addEdge(r)
        return g


    def propagate(self):
        """propagate constraints by transitivity:
        TODO
        - each matrix value i,j is to be intersected with the result of composition:
             i,j = inter_k (eik rond ekj)

        - each composition of relations is union of base compos

        x (Algebra optimized)- each base compos has to be redone by converting the table
        """
        new=numpy.array([0]*self._entity_nb**2,dtype="int16")
        new=new.reshape(self._entity_nb,self._entity_nb)
        m=self._matrix
        changed=False
        for i in range(self._entity_nb):
            for j in range(self._entity_nb):
                compo=self._universal
                for k in range(self._entity_nb):
                    if k!=i and k!=j:
                        compo = compo & self._algebra.compose(m[i][k],m[k][j],compile=True)
                if compo==0:
                    return (False,True)
                else:
                    new[i][j]=self._matrix[i][j] & compo
                    if new[i][j]!=self._matrix[i][j]:
                        changed=True
        mbis=new
        if changed:
            return (True,False)
        else:
            self._matrix=mbis
            return (True,True)

    def saturate(self):
        """propagate constraints until a fixpoint is reached on the graph
        or an inconsistency is found
        """
        cons=True
        changed=True
        while (cons) and (changed):
            (cons,changed)=self.propagate()
        return cons



# TODO: in class AllenAlgebra ?
# entries in the table to read (order is important)
# correspond to R2
list_rel=["b","bi","d","di","o","oi","m","mi","s","si","f","fi"]
# separator for entries in the compo table
FS="--"
# disjunction marker in each entry
VAL_SEP=":"
#
ALL=VAL_SEP.join(list_rel)



import os
#BASE=os.environ.get("APETITEROOT",".")
import apetite
BASE=apetite.__path__[0]

FILE_NAME=BASE+"/Config/ct13.txt"
FILE_NAME=(os.sep).join(FILE_NAME.split("/"))



# code each disjunctive entry
def translate(entry):
    return (map(lambda x:rel_name[x],entry.split(VAL_SEP)))

def read_compo_table(filename):
    """reads a composition table from an external file"""
    table={}
    table["trel_equals"]={"trel_equals":["trel_equals"]}
    for line in open(filename):
        line=line.strip()
        if line!="":
            row=line.split(FS)
            rel_R1=rel_name[row[0]]
            row=row[1:]
            table_row=[]
            for entry in row:
                if entry=="*":
                    entry=ALL+":="
                table_row.append(translate(entry))
            table[rel_R1]=dict(map((lambda x,y:(rel_name[x],y)),list_rel,table_row))
            table[rel_R1]["trel_equals"]=[rel_R1]
            table["trel_equals"][rel_R1]=[rel_R1]
    return table


table=read_compo_table(FILE_NAME)
allen_algebra=Algebra(allen_rel,inverse,table,compiled=True)



# analyse intervalle de texte (entiers ou extra) issus du tagger
def parse_position(pos):
    res=[x for x in pos.split("-") if x!="" and x!="/" and x!="extra"]
    # 
    res=[res[0],res[-1]]
    return res




# TDOO ...
#   - ? aligner les API avec l'API java pour reprendre le script jython/compare
#   - sortie tlinks/timeml (trivial)
#   - sortie ascii pour eval / java
class AllenGraph(CSPGraph):
    """specify a binary constraint graph of temporal interval in Allen's algebra"""
    def __init__(self,algebra=allen_algebra):
        # node: id+attribut (facultatif)
        self._nodes={}
        self._arcs={}
        self._algebra=algebra
        self._index=False

    def from_links(self, links, filter=[]):
        """
        builds a graph with relations given by predictions, restricted
        to edges present in filter
        """
        for (e1,e2) in links:
            if (e1,e2) in filter:
                pg.addNode(e1)
                pg.addNode(e2)
                rel = links[e1,e2]
                if rel is not None:
                    try:
                        pg.addEdge( Edge(e1,e2,Relation(rel)) )
                    except:
                        print >> sys.stderr, "problem with relation", rel
        return 

    # adjunct has properties: ...
    # separate argument for adj and id in the graph
    # for genericity (useless?)
    def addTimex(self,adj,id):
        #print adj
        """add explicit relations with existing timex"""
        # TODO !! not tested (timex() does not work)
        for x in self.timex():
           rel=adj.relation_with(x)
           if rel:
               self.addRelation(adj,id,x,x["pos"],rel)
        self.addNode(id,adj)
        #?self.addNode(id,new)

    def addEvent(self,id,event):
        self.addNode(id,event)

    #TODO: supprimer parametres inutiles
    def addRelation(self,e1,id1,e2,id2,rel):
        """add a relation to the graph, and its inverse on the inverse edge"""
        self.addNode(id1,e1)
        self.addNode(id2,e2)
        #
        self.addEdge(Edge(id1,id2,rel))
        self.addEdge(Edge(id2,id1,self._algebra.inverse(rel)))

    def addRelationList(self,relsList):
        for (e1,e2,r) in relsList:
            self.addRelation({},e1,{},e2,Relation(r))
        
        

    def saturate_w_method(self,method):
        if method=="rien":
            # no inferences -> baseline
            pass
        else:# really saturates
            self.saturate()
        return True

    # plutot dans addtimex (fur et a mesure)
    def expand_dates(self):
        pass

    def test(self):
        e1={"pos":1}
        e2={"pos":2}
        self.addRelation(e1,1,e2,2,"before")
        e3={"pos":3}
        self.addRelation(e2,2,e3,3,"before")

    
    def conversion(self,target="bruce"):
        """convert all Allen relations of the graph to Bruce relations or others (TODO)
        outputs another graph

        WARNING: no algebra declared, for storing only graph is unusuable for computation
        input graph (self) must be saturated or at least symmetric. 
        """
        # declared without an algebra, no inversion are made

        new=Graph()

        for (n1,n2),edge in self._arcs.items():
            rel_allen=edge.relation()
            if target=="bruce":
                rel_ann=rel_allen.allen2bruce()
            elif target=="jurafsky":
                rel_ann=rel_allen.allen2jurafsky()
            elif target=="tempeval":
                rel_ann=rel_allen.allen2tempeval()
            else:
                raise Exception, "unknown conversion target for temporal relation", target
            new.addEdge(Edge(n1,n2,rel_ann))
        return new

    # does not work TODO: dust off
    def timex(self,idnode=None):
        if idnode is not None:
            return self._nodes[idnode]
        else:
            return [y for (x,y) in self._nodes.items() if y.has_key("tid")]
        
    
    def timeml(self):
	"""converts the graph to a list of timeml entities+tlink"""
        data={}
        for x,y in self.nodes().items():
            if isinstance(y,date):
                data[x]=y.attrs
                data[x]["type"]="timex3"
                
            else:
                data[x]=y
                if not(y.has_key("timextype")):
                    data[x]["type"]="event"
                else:
                    data[x]["type"]="timex3"
            start,end=parse_position(data[x]["pos"])
            data[x]["position"]=[start,end]
            data[x].pop("pos")
                
        for x,y in self.edges().items():
            data[x]={"type":"tlink",
                     "value":`y.relation()`}
        return data
    
    def allen2igraph(tgraph,display="default",directed=False):
        """
        translates temporal graph as something usable by igraph,
        with an edge between two nodes only if there is a simple
        relation between the nodes (eg before), => "default" display mode

        "all" mode means all non trivial edges are marked;
        simple are marked in three colors, disjunction in red
        green for inclusion relations
        orange for before/after relations
        blue for overlaps
        
        beware that 'all/defautl' does not influence connectivity since two nodes in different
        strong components will remain disconnected after saturation of the graph

        if directed if True, the graph is directed and an arrow will show
        the edge direction 

        """
        ig=igraph.Graph(directed=directed)    
        mapping=dict(enumerate(tgraph.nodes()))
        inv_mapping=dict([(x,i) for (i,x) in enumerate(tgraph.nodes())])
        if len(mapping)>0:
            ig.add_vertices(len(mapping)-1)
        else:
            print >> sys.stderr, "warning, no nodes in graph", tgraph
            return ig
        for v in ig.vs:
            v["label"]=mapping[v.index]
            v["node"]=copy.deepcopy(tgraph.nodes()[mapping[v.index]])
        for ((e1,e2),the_edge) in tgraph.edges().items():
            r=the_edge.relation()
            idx1,idx2=(inv_mapping[e1],inv_mapping[e2])
            # ordering nodes so that only one edge is drawn and not its converse
            if len(r)==0:
                r=["inconsistent"]
            if list(r)[0] in _direct or list(r)[0]=="inconsistent":
                #print r, _color.get(hash(list(r)[0]))
                if display=="default" and len(r)==1:
                    ig.add_edges((idx1,idx2))
                    ig.es[ig.get_eid(idx1,idx2)]["relation"]=the_edge
                    ig.es[ig.get_eid(idx1,idx2)]["color"]=_color.get(list(r)[0],"red")
                elif display=="all":
                    if not(r==tgraph.algebra()._universal):
                        ig.add_edges((idx1,idx2))
                        ig.es[ig.get_eid(idx1,idx2)]["relation"]=the_edge
                        if len(r)==1:
                            edgecolor=_color.get(list(r)[0],"red")
                        else:
                            edgecolor="red"
                        ig.es[ig.get_eid(idx1,idx2)]["color"]=edgecolor
                else:
                    pass
        return ig


    def decompose(self):
        """decomposes a graph in strongly connected component with respect
        to the relation: 'is related to ... with a non trivial temporal relation'
        """
        g=self.allen2igraph(display="all")
        comps=g.decompose()
        result=[igraph2allen(x) for x in comps]
        return result


def merge_graphs(graph_list):
    res=copy.deepcopy(graph_list[0])
    for one in graph_list[1:]:
        res.nodes().update(one.nodes())
        res.edges().update(one.edges())
    return res
    
    
        
# correspondance between allen relation and node coloring. all blue for now
# but could be changed with ie: 
# _color[Relation('before')[0]]="orange"
_color=dict(zip([x for x in allen_algebra.relations()],["blue"]*13))
_color[Relation('before').pop()]="orange"
_color[Relation('after').pop()]="brown"
_color[Relation('m').pop()]="orange"
_color[Relation('mi').pop()]="brown"
_color[Relation('d').pop()]="green"
_color[Relation('=').pop()]="green"
_color[Relation('s').pop()]="green"
_color[Relation('f').pop()]="green"
_color[Relation('di').pop()]="darkgreen"
_color[Relation('si').pop()]="darkgreen"
_color[Relation('fi').pop()]="darkgreen"
_color[Relation('<').pop()]="orange"

_direct=set([x for x in allen_algebra.relations()if not(x.endswith('i'))])

def igraph2allen(graph):
    """converts a graph in igraph format to AllenGraph
    mainly, this is to be used to convert components from
    a graph built already from a temporal graph

    each vertex must have an id
    and each edge must have a relation as attribute

    
    """
    result=AllenGraph()
    for v in graph.vs:
        result.addNode(v["label"],properties=v["node"])
    for e in graph.es:
        rel=e["relation"]
        #e1=graph.vs[e.source]["label"]
        #e2=graph.vs[e.target]["label"]
        result.add(rel)
    return result






if __name__=="__main__":
    from pprint import pprint
    from copy import deepcopy

    profiling=True

    a=AllenGraph(allen_algebra)
    c1=Edge("1","2","trel_before")
    c2=Edge("3","1","trel_equals")
    a.addNode("1",{})
    a.addNode("2",{})
    a.addNode("3",{})
    a.add(c1)
    a.add(c2)

    print a
    universal=allen_algebra.universal()

    abis=a.conversion(target="bruce")
    print abis

    allen_algebra_comp=copy.deepcopy(allen_algebra)
    allen_algebra_comp.compile()
    b=CSPGraphMatrix(allen_algebra_comp,entity_nb=len(a.nodes()))
    b.convert(a)
    
    c=b.export()
    print c

    #b.propagate()
    #c=b.export()
    #print c

    sparse=False
    if profiling:
        from apetite.TimeMLDocument import Document

        if sparse:
            onefile = '../../data/TimeBank1.1/docs/wsj_1013_orig.tml.xml'
        else:
            onefile = '../../data/OTC/NYT19981026.0446.tml.xml'
        doc=Document(onefile)
        a=doc.get_graph()
        
        
        graph=deepcopy(a)
        # ex de tests de performances
        import cProfile,timeit, pstats
        
        # 1st kind of test: repeated timing
        #t=timeit.Timer("graph=deepcopy(a);graph.saturate()","from __main__ import a, deepcopy")
        # repeat 3 experiment of 10 times calling the statement
        #print t.repeat(3,10)
        print "---"
        # 2nd kind: profiling in detail; also: python -m cProfile myscript.py
        graph=deepcopy(a)
        graph.index_on_node1()
        cProfile.run('ok1=graph.saturate()','/tmp/graph.prof')
        p = pstats.Stats('/tmp/graph.prof')
        p.strip_dirs()
        p.sort_stats('cumulative').print_stats(30)

        b=CSPGraphMatrix(allen_algebra,entity_nb=len(a.nodes()))
        b.convert(a)
        cProfile.run('ok2=b.saturate()','/tmp/graph2.prof')
        p = pstats.Stats('/tmp/graph2.prof')
        p.strip_dirs()
        p.sort_stats('cumulative').print_stats(30)
        print "ok1",ok1
        print "ok2",ok2

        # sanity check
        bsat=CSPGraphMatrix(allen_algebra,entity_nb=len(a.nodes()))
        bsat.convert(graph)
        same=True
        for n1,i1 in b.nodes().items():
            for n2,i2 in b.nodes().items():
                r1=b._matrix[i1][i2]
                r2=bsat._matrix[bsat.nodes()[n1]][bsat.nodes()[n2]]
                if r1!=r2:
                    same=False
                    break
            if not(same):
                break
                print "saturation on compiled version does not match saturation on graph"
        print same
