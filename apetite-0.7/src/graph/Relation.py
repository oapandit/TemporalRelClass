#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
#################################################
"""

 provides:
     - class for relation algebra
     - def of allen algebra
     - def of annotation relations

 TODO:
    - provide for change of annotation relation 
    - clean-up: define set of relations with different names
    - refactoring: definition of AR more explicit
    - algebra class should be cleaned-up
    - add algebra pointer in relation attributes
    - fast composition with relation as boolean vector over relation set
    - cache composition for more speed-up
    - profiling
    - separate unit testing

"""
#################################################

# difference python2.3 / 2.4 (native set module -> gagne 30% de temps sur saturation de graphe)
try:
    a=set()
except NameError:
    #import sys
    #print >> sys.stderr, "python < 2.4"
    from sets import Set as set

#from sets import Set as set
from array import array
from collections import defaultdict

# init an allen-based graph
# This table will be enriched below with ALL names used for relations
# (abbrev + annotation relations)
# (TODO: string -> list)
rel_name={"=":"trel_equals",
          "e":"trel_equals",
          "b" :"trel_before",
          "bi":"trel_beforei",
          "d" :"trel_during",
          "di":"trel_duringi",
          "o" :"trel_overlap",
          "oi":"trel_overlapi",
          "m" :"trel_meet",
          "mi":"trel_meeti",
          "s" :"trel_start",
          "si":"trel_starti",
          "f" :"trel_finish",
          "fi":"trel_finishi",
          }
allen_rel=rel_name.values()
inverse={"trel_equals":"trel_equals"}
for x in rel_name.keys():
    if x[-1]=="i":
        inverse[rel_name[x]]=rel_name[x[:-1]]
        inverse[rel_name[x[:-1]]]=rel_name[x]
##############################################################
# Definition of Annotation relation
# to avoid confusion, system should use prefix arel_
# trel_ is for allen
# does not matter that much if tango is not used
#
#[arel_before, arel_after, arel_included, arel_includes, arel_overlaps, arel_overlapsi]
#
# TODO
#  ok - still a problem with equals being in two ann. rel.
#       (generates artficial disjunctions when converted back and forth to/from allen algebra)
#       two solutions:
#     1) add equals alone and get it out of the others ?
#  ok 2) use a more sophisticated conversion AR-> Allen

# WARNING: bethard uses a 'OVERLAP' that is not clear.
#          here we assume it means trel_overlap,trel_overlapi

_rel_conv_def="""before:trel_before
after:trel_beforei
iafter:trel_meeti
ibefore:trel_meet
finish:trel_finish
ended_by:trel_finishi
begun_by:trel_starti
ends:trel_finish
begins:trel_start
start:trel_start
includes:trel_duringi
overlaps:trel_overlap,trel_overlapi
overlap:trel_overlap
overlap_before:trel_overlap
overlap_after:trel_overlapi
overlapsi:trel_overlapi
is_included:trel_during
simultaneous:trel_equals
during:trel_during
during_inv:trel_duringi
identity:trel_equals""".split("\n")
#during: trel_during ?
_rel_conv={}
for x in _rel_conv_def:
    r1,r2=x.split(":")
    _rel_conv[r1]=r2
#print _rel_conv

rel_name.update(_rel_conv)

# TODO: use full names !
_annotation2allen={
    "arel_before":["b","m"],
    "arel_after":["bi","mi"],
    "arel_included":["s","d","f"],
    "arel_includes":["si","di","fi"],
    "arel_overlaps":["o"],
    #"arel_equals":["="],
    "arel_overlapsi":["oi"]
    # pascal a ajoute
    ,"arel_simultaneous":["="]
    #,"arel_is_included":["s","d","f"]
    ,"arel_identity":["="]
    ,"arel_ended_by":["fi"]
    ,"arel_ends":["f"]
    ,"arel_begins":["s"]
    ,"arel_begun_by":["si"]
    ,"arel_iafter":["mi"]
    ,"arel_ibefore":["m"]
    ,"arel_during":["d"]
    }

annotation_relations=_annotation2allen.keys()
# "=" by default associated with included
# but tested for more precise conversion (see below
# allen2annotation
_allen2annotation={
    #'=': ['arel_included', 'arel_includes'],
    #'=': ['arel_equals'],
    'b': ['arel_before'],
    'bi': ['arel_after'],
    'd': ['arel_included'],
    'di': ['arel_includes'],
    'f': ['arel_included'],
    'fi': ['arel_includes'],
    'm': ['arel_before'],
    'mi': ['arel_after'],
    'o': ['arel_overlaps'],
    'oi': ['arel_overlapsi'],
    's': ['arel_included'],
    'si': ['arel_includes']
    # pascal a ajoute
    ,'=':['arel_simultaneous']
    }


_allen2tempeval={
    'b': ['tpval_before'],
    'bi': ['tpval_after'],
    'd': ['tpval_overlap'],
    'di': ['tpval_overlap'],
    'f': ['tpval_overlap'],
    'f': ['tpval_overlap'],
    'fi': ['tpval_overlap'],
    'm': ['tpval_before'],
    'mi': ['tpval_after'],
    'o': ['tpval_overlap'],
    'oi': ['tpval_overlap'],
    's': ['tpval_overlap'],
    'si': ['tpval_overlap'],
    '=':['tpval_simultaneous']
    }

_tempeval2allen=defaultdict(list)
for r,conv in _allen2tempeval.items():
    _tempeval2allen[conv[0]].append(r)

tempeval_relations=_tempeval2allen.keys()

_allen2jurafsky={
    'b': ['jur_before'],
    'bi': ['jur_after'],
    'd': ['jur_unk'],
    'di': ['jur_unk'],
    'f': ['jur_unk'],
    'f': ['jur_unk'],
    'fi': ['jur_unk'],
    'm': ['jur_before'],
    'mi': ['jur_after'],
    'o': ['jur_unk'],
    'oi': ['jur_unk'],
    's': ['jur_unk'],
    'si': ['jur_unk'],
    '=':['jur_unk']
    }

_jurafsky2allen=defaultdict(list)
for r,conv in _allen2jurafsky.items():
    _jurafsky2allen[conv[0]].append(r)

jurafsky_relations=_jurafsky2allen.keys()

    

_allen2timeml={
    'b':['arel_before'],
    'bi':['arel_after'],
    'di':['arel_includes'],
    #'d':['arel_is_included'],
    '=':['arel_simultaneous'],
    'mi':['arel_iafter'],
    'm':['arel_ibefore'],  #'bi':['arel_ibefore'],
    's':['arel_begins'],
    'f':['arel_ends'],
    'si':['arel_begun_by'],
    'fi':['arel_ended_by'],
    'o': ['arel_overlap'], # ['overlap_before'],
    'oi': ['arel_overlapi'] # ['overlap_after']
    }



_conceptual_neigh={
    'b':['m'],
    'bi':['mi'],
    'di':['=','si','fi'],
    'd':['=','s','f'],
    '=':['si','fi','di','d','s','f','o','oi'],
    'mi':['bi','oi'],
    'm':['b','o'],
    's':['d','=','o'],
    'f':['d','=','oi'],
    'si':['di','=','oi'],
    'fi':['di','=','o'],
    'o': ['m','s','fi','='],
    'oi':['mi','si','f','=']
    }

extension={}
for one in _conceptual_neigh:
    extension[rel_name[one]]=[rel_name[x] for x in _conceptual_neigh[one]]

_conceptual_neigh.update(extension)

abbrev=_allen2annotation.keys()
for x in abbrev:
    _allen2annotation[rel_name[x]]= _allen2annotation[x]
    _allen2tempeval[rel_name[x]]= _allen2tempeval[x]
    _allen2jurafsky[rel_name[x]]= _allen2jurafsky[x]

abbrev2=_allen2timeml.keys()
for x in abbrev2:
    _allen2timeml[rel_name[x]]= _allen2timeml[x]


#

# class relation h�rite de la classe ensemble
class Relation(set):
    """temporal relation, in a given algebra
    so is a subset of the set of all possible relations, seen as a disjunction

    >>> a=Relation(["trel_equals","trel_during"])
    >>> a.allen2bruce()
    [arel_simultaneous,arel_included]
    >>> a=Relation(["trel_equals","trel_duringi"])
    >>> a.allen2bruce()
    [arel_simultaneous,arel_includes]
    >>> a=Relation(["trel_equals","trel_duringi","trel_before","trel_during"])
    >>> a.allen2bruce()
    [arel_before,arel_simultaneous,arel_includes,arel_included]
    >>> a=Relation(["trel_duringi","trel_before","trel_during"])
    >>> a.allen2bruce()
    [arel_before,arel_includes,arel_included]
    >>> b=a.allen2bruce()
    >>> b
    [arel_before,arel_includes,arel_included]
    >>> b.bruce2allen()
    [trel_during,trel_start,trel_finishi,trel_finish,trel_meet,trel_duringi,trel_starti,trel_before]
    >>> a=Relation(["trel_equals"])
    >>> a.is_simple()
    True
    >>> a.conceptual_neighbours()
    ['trel_starti', 'trel_finishi', 'trel_duringi', 'trel_during', 'trel_start', 'trel_finish', 'trel_overlap', 'trel_overlapi']
    """

    def __init__(self,description,myalgebra=None):
        """ description can be
           - alias ("after"=trel_beforei)
           - disjunction "before after" ou "before, after"
           - formatted list "[before,after]"

           optionally, an algebra to which the relation belongs can be added

        """
        self.descript = description
        if isinstance(description,str):
            description=rel_name.get(description,description)
            #description=rel_name[description]
            description=description.replace(","," ")
            if description.startswith("["):
                description=description[1:-1]
            if " " in description:
                rel=description.split()
            else:
                rel=[description]
        else:
            rel=description
        rel=[rel_name.get(x,x) for x in rel]
        set.__init__(self,rel)
        self._algebra=myalgebra
        self._compiled=None

    def get_rel(self):
        return self.descript.lower()

    def __repr__(self):
        result=",".join([x.lower() for x in self])
        return "[%s]"%result

    def includes(self,other):
        """
        is self more general than other ? (other is included in self)
        """
        return other in self


    def __hash__(self):
        return hash(self.__repr__())

    # cas des relations allen ou annotation
    # 
    def allen2bruce(self):
        """allen relation converted to Bruce relation

        warning: this operation loses information
        """
        res=[]
        # cas de include/is_included
        #has_equal=("trel_equals" in rel)
        #rel.discard("trel_equals")
        for x in self:
            res.extend(_allen2annotation[x])
        res=Relation(res)
        #if ("arel_includes" in res) or ("arel_included" in res):
        #    pass
        #elif has_equal:# not very likely equals alone ... but still
        #    res.add(_allen2annotation["="][0])
        return res

    def allen2tempeval(self):
        """allen relation converted totempeval relation

        warning: this operation loses information

        >>> a=Relation(["trel_equals","trel_during"])
        >>> a
        [trel_during,trel_equals]
        >>> a.allen2tempeval()
        [tpval_overlap,tpval_simultaneous]

        """
        res=[]

        for x in self:
            res.extend(_allen2tempeval[x])
        res=Relation(res)

        return res

    def allen2jurafsky(self):
        """allen relation converted to jurafsky relation scheme

        warning: this operation loses information
        >>> a=Relation(["trel_equals","trel_during"])
        >>> a
        [trel_during,trel_equals]
        >>> a.allen2jurafsky()
        [jur_unk]
        """
        res=[]

        for x in self:
            res.extend(_allen2jurafsky[x])
        res=Relation(res)

        return res

    def allen2other(self,target):
        if target=="bruce":
            return self.allen2bruce()
        elif target=="jurafsky":
            return self.allen2jurafsky()
        elif target=="tempeval":
            return self.allen2tempeval()
        else:
            raise Exception, "unknown conversion target for temporal relation"+target

    def other2allen(self,source):
        if source=="bruce":
            return self.bruce2allen()
        elif source=="jurafsky":
            return self.jurafsky2allen()
        elif source=="tempeval":
            return self.tempeval2allen()
        else:
            raise Exception, "unknown conversion source for temporal relation"+source



    def allen2timeml(self):
        """tested ? converts allen relation to timeml normalised names ?
        """
        res=[]
        for x in self:
            res.extend(_allen2timeml[x])
        res=Relation(res)
        return res

    def tempeval2allen(self):
        """tempeval relation converted to allen relation
        >>> a=Relation(["trel_during"])
        >>> a.allen2tempeval().tempeval2allen()
        [trel_overlap,trel_start,trel_overlapi,trel_finishi,trel_finish,trel_during,trel_duringi,trel_starti]
        """
        res=[]
        for x in self:
            res.extend([rel_name[x] for x in _tempeval2allen[x]])
        return Relation(res)

    def jurafsky2allen(self):
        """jurasky 3-relation scheme converted to allen relation
        >>> a=Relation(["trel_equals","trel_during"])
        >>> a.allen2jurafsky().jurafsky2allen()
        [trel_overlap,trel_start,trel_overlapi,trel_finishi,trel_finish,trel_during,trel_duringi,trel_equals,trel_starti]
        """
        res=[]
        for x in self:
            res.extend([rel_name[x] for x in _jurafsky2allen[x]])
        return Relation(res)

    def bruce2allen(self):
        """bruce relation converted to allen relation
        """
        res=[]
        for x in self:
            res.extend([rel_name[x] for x in _annotation2allen[x]])
        return Relation(res)

    def timeml2allen(self):
        """tested ? converts timeml normalised names to allen relation ?
        """
 
        res=[]
        for x in self:
            res.extend(Relation(x[5:].encode())) # strip arel_
        return Relation(res)


    def set_algebra(self,alg):
        """
        makes the relation aware of the algebra alg, if
        (if the relation belongs to the algebra)
        """
        self._algebra=alg

    def algebra(self):
        return self._algebra

    
    def compile(self):
        """
        relation made as a boolean vector, coded in an int
        algebra needed, otherwise does nothing

        NOT TESTED

        buys union and intersection for free as | and &
        so composition has to be faster ...
        except coding/decoding must be redone: index rel on bits
        """
        if self._algebra and self._compiled is None:
            self._compiled=0
            for (i,rel) in enumerate(self._algebra.universal()):
                # each relation corresponds to a bit position
                # according to its (arbitrary)  position in the universal relation
                if rel in self:
                    self._compiled += (1 << i)
        return self._compiled


    def from_compiled(self,value):
        """translates a compiled version into normal relation"""
        if self._algebra is not None:
            if value==self._algebra._universal_compiled:
                return self._algebra._universal
            rels=Relation([],myalgebra=self._algebra)
            for (i,rel) in enumerate(self._algebra._universal):
                filter = (1 << i)
                if filter & value:
                    rels.add(rel)
        return rels


    def is_simple(self):
        """TODO: test"""
        return len(self)==1

    def conceptual_neighbours(self):
        """TODO: test"""
        if not(self.is_simple()):
            return None
        else:
            return _conceptual_neigh[list(self)[0]]






class Algebra:
    """ defined by a set of exclusive, exhaustive relations
     and an internal composition relation (double indexed dict)
     and an interval inverse relation (dict)
    """
    
    def __init__(self,rel_set,inverse,compo_table,compiled=False):
        """
        rel_set is complete set of relation
        inverse is a table relating relations with their inverses
        compo_table is a composition table
        """
        self._relations = Relation(rel_set,myalgebra=self)
        self._universal=self._relations
        self._inverse = inverse
        self._compose = compo_table
        self._relations_nb=len(self._relations)
        if compiled:
            self._universal_compiled=self._relations.compile()
            self.compile()
        else:
            self._universal_compiled=None

    def compile(self):
        """computes composition table on optimized version of relations
        TODO: compile inverses
        """
        if self._universal_compiled is None:
            self._universal_compiled=self._relations.compile()
        table_optimized={}
        table=self._compose
        for one in table:
            r1=Relation(one,myalgebra=self)
            r1comp=r1.compile()
            table_optimized[r1comp]={}
            for two in table[one]:
                r2=Relation(two,myalgebra=self)
                r2comp=r2.compile()
                result=Relation(table[one][two],myalgebra=self)
                resultcomp=result.compile()
                table_optimized[r1comp][r2comp]=resultcomp
        self._compose_optimized=table_optimized


    def get_relations(self):
        """return the set of all relations
        """
        return self._relations

    def relations(self):
        """return the set of all relations
        """
        return self._relations


    def member(self,rel):
        """is rel in this algebra ?
        """
        return rel in self.get_relations()

    # relation non contrainte: disjonction de tout
    def universal(self,compiled=False):
        """relation that is the disjunction of all relations"""
        if compiled:
            return self._universal_compiled
        else:
            return self.relations()

        
    # rel is a string description
    def inverse_base(self,rel):
        """
        converse of a base relation
        """
        return self._inverse[rel]

    # 
    def inverse(self,rel,compile=False):
        """
        converse of any relation

        TODO: compile
        """
        res=Relation([])
        for x in rel:
            res.add(self.inverse_base(x))
      
        return res
        
    # rel+rel->rel
    def compose_base(self,rel1,rel2,compile=False):
        """
        compositions of two base relations; 
        returns a set of possible relations (disjunctive relation)

        TODO: compile
        """
        #print rel1,rel2
        if compile:
            return self._compose_optimized[rel1][rel2]
        else:
            return self._compose[rel1][rel2]

    # rel set + rel set ->     
    # 
    def compose(self,rel1,rel2,compile=False):
        """
        compositions of any two relations; 
        returns a set of possible relations (disjunctive relation)

        TODO: compose directly compiled relation: union is or, intersection is and
        """
        #print "compose:", rel1,rel2

        if compile:
            # first check if the two relations are informative else compo is no-info
            univ=self._universal_compiled
            if rel1==univ or rel2==univ:
                return univ
            # then checks if this has been already computed (results are cached)
            if self._compose_optimized.get(rel1,{}).get(rel2):
                return self._compose_optimized[rel1][rel2]
            # otherwise, suck it up and cache the results
            result=0
            all_rels=[1 << i for i in range(self._relations_nb)]
            for i in all_rels:
                r1=rel1 & i
                for j in all_rels:
                    r2=rel2 & j
                    if r1!=0 and r2!=0:
                        result = result | self.compose_base(r1,r2,compile=True)
                        if result==univ:
                            if not(self._compose_optimized.has_key(rel1)):
                                self._compose_optimized[rel1]={}
                            self._compose_optimized[rel1][rel2]=univ
                            return univ
            if not(self._compose_optimized.has_key(rel1)):
                self._compose_optimized[rel1]={}
            self._compose_optimized[rel1][rel2]=result
            return result
        else:
            univ=self._universal
            # should'nt happen as we compose only informative edges
            #if rel1==univ or rel2==univ:
            #    return univ
            #if self._compose.get(rel1,{}).get(rel2):
            #    return self._compose[rel1][rel2]
            res=[]
            # faster if we use sets and stop when universal relation obtained ?
            for x in rel1:
                res2=[]
                for y in rel2:
                    res2.extend(self.compose_base(x,y))
                res.extend(res2)
            resf=Relation(res)
            
            return resf



    def __repr__(self):
        return "algebra, with relations " + `self._relations`

    def __str__(self):
        return `self._relations`



def _test():
    import doctest, Relation
    return doctest.testmod(Relation)

# tests
if __name__=="__main__":

    import sys
    a=Relation
    # tests:
    _test()


