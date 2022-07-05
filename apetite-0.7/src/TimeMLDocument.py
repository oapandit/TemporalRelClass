#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
document class for representing TimeML annotated texts

a faire:

   - gestion du temps du document (ignor� pour l'instant)
   - gestion meta-donn�es
   - default encoding is from TimeML11Reader = latin1 = not safe
"""
import os.path

import sys
import os
from copy import deepcopy
from timeml_reader import TimeML11Reader
from preprocessed_reader import PreprocessedReader
from markable import Markable, Token, Event, EventInstance, Timex3, Enamex
import re
from graph.Graph import AllenGraph, allen_algebra, Edge



'''

preprocessed data turned off for now...

'''

latex_template=r"""
\documentclass[a4,12pt]{paper}

\usepackage{a4}

\usepackage{pstricks,pst-grad}
\usepackage{tikz}
\usepackage{times}
\usepgflibrary{shapes}
\usepackage[utf8x]{inputenc}
\usepackage{epsfig}
\usepackage{multicol}

\setlength{\columnsep}{0cm}
\setlength{\columnseprule}{1mm}
\setlength{\parindent}{0.0cm}
\setlength\topmargin{-2cm}
\setlength\headheight{0in}
\setlength\headsep{-0.5cm}
\setlength\textheight{28.5cm}
\setlength\textwidth{21cm}
\setlength\oddsidemargin{-2.5cm}
\setlength\evensidemargin{-2.5cm}

\renewcommand{\baselinestretch}{2}


\fontsize{8pt}{8pt}\selectfont

\newcommand{\lex}[3][orange!30]{\tikz[remember picture,baseline] \node[anchor=base,fill=#1,inner sep=1pt] (#2) {#3};}
\newcommand{\lien}[3][orange!60]{\draw[->,#1,dashed,line width=1.5pt] (#2) -- (#3);}
\newcommand{\HR}{\\ \rule{\linewidth}{0.2mm} \\}

\pagestyle{empty}

\begin{document}

%s

\begin{center}
\begin{tikzpicture}
\node[rectangle,rounded corners,shade] (idxx)
{
\begin{tabular}{ll}
before/meet & orange \\
after/mi & brown    \\ 
s, d, f, = & green   \\ 
si, di fi & darkgreen   \\ 
overlaps & blue\\  
\end{tabular}
};
\end{tikzpicture}
\end{center}

\begin{tikzpicture}[remember picture,overlay]
%s
\end{tikzpicture}

\end{document}
""".decode("utf8")


_escape_chars=re.compile("(?P<echar>(&|\$|_|%|#))")



class Document:

    def __init__(self,filename, do_prep=False):
        ''' load timeml annotations '''
        reader = TimeML11Reader( filename )
        self.string = reader.get_rawstr()
        self.path = reader.path
        self.id = os.path.basename(self.path)
        self.events = reader.get_events()
        self.timex = reader.get_timex()
        self.event_instances = reader.get_event_instances()
        self.temporal_entities = dict(self.event_instances, **self.timex)
        self.tlinks = reader.get_tlinks()
        self.enamex = reader.get_enamex()
        self.numex = reader.get_numex()
        self.sentences = reader.get_sentences()
        self.signals = reader.get_signals()
        self._graph = self.set_graph( self.tlinks )
        self.markable=None # what does this doo?????

        ''' load preprocessing info '''
        if do_prep:
            prep_source = PreprocessedReader( filename[:-4]+'.prep.xml' )
            self.prep_annotations = prep_source.annotations
            self.do_prep=True
            self.set_tokens(self.prep_annotations['token']) 
        else:
            self.do_prep=False
            self.tokens = []
            self.offset2token = {}
        return
        

    def set_tokens(self, annotation_dict):
        ''' call constructor Token on annotation_dict elements and
        append Token instances into list sorted on character
        extents. also build map from offset position to token'''
        self.tokens = []
        for (_id, _dict) in annotation_dict.items():
            instance = Token(_dict)
            self.tokens.append(instance)
        self.tokens.sort(cmp=lambda x,y:cmp(x.extent,y.extent))
        self.set_offset2token_map()
        return
    

    def set_offset2token_map(self):
        ''' build dictionary: offset position => token. this
        dictionary is used to find tokens for different markables'''
        self.offset2token = {}
        for tok in self.tokens:
            for cpos in xrange(tok.extent[0],tok.extent[1]+1):
                self.offset2token[cpos] = tok
        return


    def set_dict(self, annotation_dict, _class):
        ''' call constructor _class on annotation_dict elements insert
        into dict with ID keys'''
        mark_dict = {}
        for (_id, _dict) in annotation_dict.items():
            if issubclass(_class, Markable): 
                instance = _class(_dict, self.offset2token, self.tokens)
            elif _class == EventInstance:
                instance = _class(_dict, self.events)
            mark_dict[instance.id] = instance
        return mark_dict


    def set_graph(self, tlinks):
        ''' create unsaturated Allen graph from tlinks '''
        # print("Setting graph")
        g = AllenGraph(allen_algebra)
        for eid in self.temporal_entities:
            g.addNode(eid, self.temporal_entities[eid])
        for tlink in tlinks.values():
            id1, id2, rel = tlink.arg1, tlink.arg2, tlink.relation
            # print (id1)
            # print(rel)
            # OP : changes made to see relation is not list
            if isinstance(rel, list):
                rel = rel[0]
            if id1 is not None and id2 is not None and id1!=id2 and rel.get_rel()!="vague":
                x1 = self.temporal_entities.get(id1)
                g.addNode(id1, x1)
                x2 = self.temporal_entities.get(id2)
                g.addNode(id2, x2)
                # print rel
                g.add(Edge(id1, id2, rel))
        return g


    def consistent_graph_saturation(self):
        g = deepcopy(self._graph)
        return g.saturate()


    ''' Accessors '''
    def get_events(self):
        return self.event_instances

    def get_timex(self):
        return self.timex

    def get_entities(self):
        return self.temporal_entities

    def get_graph(self, relset="allen", bethard_rels={}, isodate=False, saturation=False):
        agraph = deepcopy(self._graph)
        # add bethard relations
        if bethard_rels: # note: rel_dict is indexed by timeml filename
            try:
                 agraph = self.add_bethard_relations( agraph, bethard_rels )
            except KeyError,e:
                 print >> sys.stderr, "WARNING: problem adding Bethard relations (%s)" %e
        # add extra time-time computed through isodate normalization
        if isodate:
            agraph = self.add_extra_time_time_relations( agraph )
        # saturate
        if saturation:
            agraph = saturate( agraph )
        # return graph using selected relation set / algebra
        return relset_conversion( agraph,  relset)


    def get_derived_relations_if_consistent(self):
        agraph = self.get_graph(isodate=True)
        orig_relation_dict = agraph.relation_dict()
        cons = agraph.saturate()
        if cons:
            # print "graph consistent"
            new_relation_dict = agraph.relation_dict()
            derived_rel_dict = {}
            for key in new_relation_dict.keys():
                if orig_relation_dict.get(key,None) is None:
                    derived_rel_dict[key] = new_relation_dict[key]

        else:
            # print "graph inconsistent"
            derived_rel_dict = None

        return derived_rel_dict


    def get_relations(self, relset="allen", bethard_rels={}, isodate=False, saturation=False):
        agraph = self.get_graph(relset=relset, bethard_rels=bethard_rels, isodate=isodate, saturation=saturation)
        return agraph.relation_dict()


    def add_bethard_relations(self, agraph, rel_dict): # note: rel_dict is indexed by timeml filename
        rels = rel_dict.get( os.path.basename(self.path) )
        if rels:
            for (n1, n2), rel in rels.items():
               if n1 not in agraph.nodes():
                   raise KeyError("Bethard rel. uses undefined node %s in doc %s" %(n1,self.id))
               if n2 not in agraph.nodes():
                   raise KeyError("Bethard rel. uses undefined node %s in doc %s" %(n2,self.id))
               agraph.add(Edge(n1, n2, rel))
        else:
            print >> sys.stderr, "No Bethard relations for document %s" %self.path
        return agraph


    def add_extra_time_time_relations(self, agraph):
        ''' relations computed based on isodate normalization '''
        relations = {}
        timex_ids = self.timex.keys()
        for i in xrange(len(timex_ids)):
            for j in xrange(len(timex_ids)):
                if i == j or j > i:
                    continue
                t1,t2 = timex_ids[i],timex_ids[j]
                tex1 = self.timex[t1]
                tex2 = self.timex[t2]
                rel = tex1.relation_with(tex2) # rel instance or None
                # print tex1.referent,tex2.referent,rel
                if rel:
                    relations[(t1,t2)] = rel
        if relations:
            for (n1, n2), rel in relations.items():
               agraph.addEdge(Edge(n1, n2, rel))
        return agraph

        
    def get_sub_graph(self, relset="allen", condition=lambda x,y:True, \
        nodisjunction=False, bethard_rels={}, isodate=False, loc_sat=False, glob_sat=False):
        """ accessor for extracting sub-graph using a filter on the type of edges:
        e.g. edge's nodes need to be events or on type of relations: e.g., only simple relations

        NOTE: local saturation (loc_sat) is performed AFTER filtering, while global saturation (glob_sat)
        is performed before.
        """
        # copy entire base graph (i.e., an Allen graph)
        agraph = self.get_graph(relset="allen", bethard_rels=bethard_rels, isodate=isodate, saturation=glob_sat)
        # apply filters
        agraph = filter_edge_types( agraph, condition)
        if nodisjunction:
            agraph = filter_disjunctions( agraph )
        # local saturation
        if loc_sat:
            agraph = saturate( agraph )
        # re-apply filters
        agraph = filter_edge_types( agraph, condition)
        if nodisjunction:
            agraph = filter_disjunctions(agraph)
        # conversion to specified relset
        return relset_conversion( agraph, relset )


    def get_event_event_graph(self, relset="allen", nodisjunction=False, bethard_rels={}, isodate=False, loc_sat=False, glob_sat=False):
        two_events = lambda x,y: x in self.event_instances and y in self.event_instances
        return self.get_sub_graph(relset=relset, condition=two_events, nodisjunction=nodisjunction,
                                  bethard_rels=bethard_rels, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)
        
    def get_event_event_relations(self, relset="allen", nodisjunction=False, bethard_rels={}, isodate=False, loc_sat=False, glob_sat=False):
        ee_graph = self.get_event_event_graph(relset=relset, nodisjunction=nodisjunction,
                                              bethard_rels=bethard_rels, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)
        return ee_graph.relation_dict()

    def get_event_time_graph(self, relset="allen", nodisjunction=False, isodate=False, loc_sat=False, glob_sat=False):
        one_evt_one_tmx = lambda x,y: (x in self.event_instances and y in self.timex) or (y in self.event_instances and x in self.timex)
        return self.get_sub_graph(relset=relset, condition=one_evt_one_tmx, nodisjunction=nodisjunction, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)

    def get_event_time_relations(self, relset="allen", nodisjunction=False, isodate=False, loc_sat=False, glob_sat=False):
        et_graph = self.get_event_time_graph(relset=relset, nodisjunction=nodisjunction, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)
        return et_graph.relation_dict()

    def get_time_time_graph(self, relset="allen", nodisjunction=False, isodate=False, loc_sat=False, glob_sat=False):
        two_timex = lambda x,y: x in self.timex and y in self.timex
        return self.get_sub_graph(relset=relset, condition=two_timex, nodisjunction=nodisjunction, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)

    def get_time_time_relations(self, relset="allen", nodisjunction=False, isodate=False, loc_sat=False, glob_sat=False):
        tt_graph = self.get_time_time_graph(relset=relset, nodisjunction=nodisjunction, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)
        return tt_graph.relation_dict()

    def get_tt_et_graph(self, relset="allen", nodisjunction=False, isodate=False, loc_sat=False, glob_sat=False):
        not_two_events = lambda x,y: not (x in self.event_instances and y in self.event_instances)
        return self.get_sub_graph(relset=relset, condition=not_two_events, nodisjunction=nodisjunction, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)
 
    def get_tt_et_relations(self, relset="allen", nodisjunction=False, isodate=False, loc_sat=False, glob_sat=False):
        tt_et_graph = self.get_tt_et_graph(relset=relset, nodisjunction=nodisjunction, isodate=isodate, loc_sat=loc_sat, glob_sat=glob_sat)
        return tt_et_graph.relation_dict()


    def __str__(self):
        return "EVENTS:%s\n\nINSTANCES:%s\n\nRELATIONS:%s\n\nTIMEX:%s\n\nENAMEX:%s" \
               %(self.events,self.event_instances,self.get_relations(),self.timex,self.enamex)


    # TODO: probably deprecated...
    def compare(self,other,cats=["timex3"]):
        """ compare self (reference) with document other (to be evaluated)"""
        return CompReport(self,other,cats=cats)


    def index_markable(self):
        """build an index of markable -> token range

        TODO: publication time has no span .... mark it at (-1,0) 
        """
        self.markable={}
        markable= self.markable
        for one in self.timex:
            begin,end=self.timex[one].extent
            markable[one]=('TIMEX3',self.offset2token.get(begin,-1),self.offset2token.get(end,0))

        for one in self.events:
            begin,end=self.events[one].extent
            markable[one]=('EVENT',self.offset2token[begin],self.offset2token[end])

            
        self.markable_index={}
        index=self.markable_index={}
        for one in markable:
            ttype,token1,token2=markable[one]
            if token1==token2:
                index[token1]=("startend",one,ttype)
            else:
                index[token1]=("start",one,ttype)
                index[token2]=("end",one,ttype)
            

    def entity_extent(self,ent_instance):
        """extent of a timex, or, for an event instance, its corresponding event 
        """
        one=ent_instance
        if one.id.lower().startswith("t"): 
                mark_type="TIMEX3" 
        else: 
            mark_type="EVENT"
        if mark_type=="EVENT":
            extent = one.event.extent
        else:
            extent = one.extent
        return extent


    def to_latex(self,color={},nopubtime=False,large=False,**kwargs):
        """export to latex for links visualisation
        
        optional arguments:
        nopubtime= do not show links with publication time
        eventevent= show only those links

        """
        text=self.string
        open_markup=r"\lex[%s!30]{%s}{"
        end_markup=r"}"
        if large:
            end_markup += r"\\ "
            
        fmt_link=r"\lien[%s!60]{%s}{%s}"

        # each insertion blows the location by a length of delta
        offset=0
        end_delta=len(end_markup)
        temp_entities=[(self.entity_extent(x[1]),x) for x in self.temporal_entities.items()]
        temp_entities.sort()
        temp_entities=[x[1] for x in temp_entities]
        pubtime=None
        for idx,one in temp_entities:
            if one.id.lower().startswith("t"): 
                mark_type="TIMEX3" 
            else: 
                mark_type="EVENT"
            extent=self.entity_extent(one)
            # just keep track of it
            if mark_type=="TIMEX3":
                if one.fcn_in_doc=="CREATION_TIME":
                    pubtime=idx
                    open_markup=r"\hfill\lex[%s!30]{%s}{Pub time"
                else:
                    open_markup=r"\lex[%s!30]{%s}{"
            else:
                    open_markup=r"\lex[%s!30]{%s}{"
            # record extent and compute new offset from original text
            start,end=extent
            start = start + offset
            end   = end + offset
            offset= offset + len(open_markup%(color.get(mark_type,"blue"),idx))+end_delta
            #print >> sys.stderr, mark_type, extent, start,end,offset,text[start:end+1]
            text=text[:start]+ open_markup%(color.get(mark_type,"blue"),idx) + text[start:end+1] +end_markup +text[end+1:]
            
        links=[]
        all_rels=self.get_relations(**kwargs)
        for (id1,id2) in all_rels:
            relation=all_rels[(id1,id2)]
            if nopubtime and (id1==pubtime or id2==pubtime):
                pass
            else:
                try:
                    relation=relation.pop()
                    links.append(fmt_link%(color.get(relation,"red"),id1,id2))
                except:
                    print >> sys.stderr, "problem on relation (ignored)", relation, (id1,id2)
        # reader encodes in latin1 for now, but safer to use unicode
        output_text=_escape_chars.sub(r"\\\g<echar>",text).decode("latin1")
        return latex_template%(output_text,u"\n".join(links))



    def to_latex_old(self,color={},**kwargs):
        """export to latex for links visualisation
        """
        if self.markable is None:
            self.index_markable()
        markable=self.markable
        index=self.markable_index
       
        out=[]
        links=[]
        publication_time=index[-1]
        id=publication_time[1]
        
        out.append(r"\hfill\lex[blue!30]{%s}{Publication time})"%id)
        for token in self.tokens: 
            if token in index:
                (type,id,mark_type)=index[token]
                if type=="start":
                    out.append(r"\lex[%s!30]{%s}{%s"%(color.get(mark_type,"blue"),id,token.word))
                elif type=="end":
                    out.append(r"%s}"%(token.word,))
                else:#type="startend"
                    out.append(r"\lex[%s!30]{%s}{%s}"%(color.get(mark_type,"blue"),id,token.word))
            else:
                out.append(token.word)
        fmt_link=r"\lien[%s!60]{%s}{%s}"
        all_rels=self.get_relations(**kwargs)
        for (id1,id2) in all_rels:
            relation=all_rels[(id1,id2)]
            # align instances and events if they are not identical
            try:
                if id1.lower().startswith('e'):
                    id1=self.event_instances.get(id1)
                    if id1 is not None:
                        id1=id1.eventID
                    else:
                        print >> sys.stderr, "warning, unknwon event instance"
                        raise
                if id2.lower().startswith('e'):
                    id2=self.event_instances.get(id2)
                    if id2 is not None:
                        id2=id2.eventID
                    else:
                        print >> sys.stderr, "warning, unknwon event instance"
                        raise
                relation=relation.pop()
                links.append(fmt_link%(color.get(relation,"red"),id1,id2))
            except:
                print >> sys.stderr, "problem on relation (ignored)", relation, (id1,id2)


        output_text=" ".join(out)
        output_text=_escape_chars.sub(r"\\\g<echar>",output_text)
        return latex_template%(output_text,"\n".join(links))


## utils

def saturate(agraph):
    ''' simple wrapper around graph's saturate() method. return input graph if saturation is inconsistent'''
    sat_graph = deepcopy( agraph )
    cons = sat_graph.saturate()
    if cons:
        agraph = sat_graph
        #else:
        #     print >> sys.stderr, "WARNING: Saturation yields inconsistent graph: using unsaturated graph instead"
    return agraph


def relset_conversion(allen_graph, target):
     if target == "allen":
        return allen_graph
     elif target in ["bruce", "tempeval", "jurafsky"]:
        return allen_graph.conversion(target)
     else:
        raise TypeError("Invalid value for relset: %s" %target)


def filter_edge_types(agraph, condition):
    for edge in agraph.edges().values():
        if not condition(edge.node1(),edge.node2()):
            agraph.del_edge( edge )
    return agraph

    
def filter_disjunctions(agraph):
    for edge in agraph.edges().values():
        if not edge.relation().is_simple():
            agraph.del_edge( edge )
    return agraph


    
# NOT TESTED -- DO NOT USE
class CompReport:
    """ compare self (reference) with document other (to be evaluated)
    
    TODO: port to current API
    """
    
    def __init__(self,doc1,doc2,cats=["timex3","event"]):
        self.approximate={}
        self.missed={}
        self.spurious={}
        self.correct={}
        self.mapping={}# mapping approximate matches, indexed by target object (value:matched reference object)
        for one in cats:
            approximate=[]
            missed=[]
            correct=[]
            mapping={}
            #ref1=doc1.temporal_entities[one]
            #ref2=doc2.temporal_entities[one]
	    ref1=[doc1.temporal_entities[x] for x in doc1.temporal_entities if x.startswith(one[0])]
	    ref2=[doc2.temporal_entities[x] for x in doc2.temporal_entities if x.startswith(one[0])]
            #print ref1
	    #print ref2
	    for objet in ref1:
                if objet in ref2:
                    correct.append(objet.id())
                else:
                    overlap=[x.id() for x in ref2 if objet.overlaps(x)]
                    approximate.extend(overlap)
                    # there can be multiple match (ex split dates)
                    # so it's hard to score this ...
                    if overlap==[]:
                        missed.append(objet.id())
                    else:
                        for error in overlap:
                            mapping[error]=objet.id()
            self.spurious[one]=list(set([x.id() for x in ref2]) - set(correct) - set(approximate))
            self.correct[one]=correct
            self.approximate[one]=approximate
            self.missed[one]=deepcopy(missed)
            self.mapping[one]=mapping
 
        self.ref=doc1.temporal_entities
        self.target=doc2.temporal_entities

    def precision(self):
        return 1

    def rappel(self):
        return 0

    def display(self,label="textid",cats=["timex3","event"],header=False):
        """affiche la comparaison pour import tableur
        format:

        textid,score,position correcte,position trouvee,chaine correcte, chaine trouvee,(feature correcte,feature trouvee,score)*

        si la ref est vide (ex position correcte=None), c'est une erreur (precision)
        si la cible est vide, c'est un oubli (rappel)
        """
        header_text="textid,type entite,score,chaine correcte, chaine trouvee,meme chaine,position correcte,position trouvee,meme position,valeur correcte,valeur trouvee,meme valeur"
        if header:
            print header_text
        for cat in cats:
            for one in self.correct.get(cat,[]):
                ref=self.ref[one].main_features()
                tar=self.target[one].main_features()
                format=zip(ref,tar)
                print ",".join([label,cat,"correct",]+["%s,%s,%s"%(x,y,x==y) for (x,y) in format])
            
            
            for one in self.approximate.get(cat,[]):
                ref=self.ref[self.mapping[cat][one]].main_features()
                tar=self.target[one].main_features()
                format=zip(ref,tar)
                print ",".join([label,cat,"approximate",]+["%s,%s,%s"%(x,y,x==y) for (x,y) in format])

            
            for one in self.missed.get(cat,[]):
                ref=self.ref[one].main_features()
                tar=[None]*len(ref)
                format=zip(ref,tar)
                print ",".join([label,cat,"missed",]+["%s,%s,%s"%(x,y,x==y) for (x,y) in format])

            for one in self.spurious.get(cat,[]):
                tar=self.target[one].main_features()
                ref=[None]*len(tar)
                format=zip(ref,tar)
                print ",".join([label,cat,"spurious",]+["%s,%s,%s"%(x,y,x==y) for (x,y) in format])
            






############################





class Directory:

    def __init__(self, path):
        self.path = path
        self.documents = []
        files = [d for d in os.listdir(path) if d.endswith('tml.xml') and not d.startswith('.')]
        for (i,f) in enumerate(files):
            os.write(1, "%s" %"\b"*len(str(i+1))+str(i+1))
            doc = Document( os.path.join(self.path,f) )
            self.documents.append( doc )
        os.write(1,' documents\n')
        return 


    def stats(self):
        print "-"*124
        print "| %30s | %15s | %15s | %15s | %15s | %15s |" \
              %("Document", "# E-E Rels", "# E-T Rels", "# T-T Rels", "All Rels", "Cons. sat.")
        print "-"*124
        total_rels, total_rels2 = 0,0 
        total_ee, total_ee2  = 0, 0
        total_et, total_et2 = 0, 0
        total_tt, total_tt2 = 0, 0
        cons_graph_ct = 0
        doc_ct = 0
        for doc in self.documents:
            doc_ct += 1
            # without saturation
            rel_ct = len(doc.get_relations()) 
            ee_rel_ct = len(doc.get_event_event_relations())
            et_rel_ct = len(doc.get_event_time_relations())
            tt_rel_ct = len(doc.get_time_time_relations())
            # with saturation
            rel_ct2 = len(doc.get_relations(saturation=True))
            ee_rel_ct2 = len(doc.get_event_event_relations(glob_sat=True))
            et_rel_ct2 = len(doc.get_event_time_relations(glob_sat=True))
            tt_rel_ct2 = len(doc.get_time_time_relations(glob_sat=True))
            # consistent saturation?
            cons = doc.consistent_graph_saturation()
            if cons: cons_graph_ct += 1
            print "| %30s | %15s | %15s | %15s | %15s | %15s |" %(os.path.basename(doc.path),
                                                                  "%s/%s" %(ee_rel_ct,ee_rel_ct2),
                                                                  "%s/%s" %(et_rel_ct,et_rel_ct2),
                                                                  "%s/%s" %(tt_rel_ct,tt_rel_ct2),
                                                                  "%s/%s" %(rel_ct,rel_ct2),
                                                                  cons)
            total_rels += rel_ct
            total_rels2 += rel_ct2
            total_ee += ee_rel_ct
            total_ee2 += ee_rel_ct2
            total_et += et_rel_ct
            total_et2 += et_rel_ct2
            total_tt += tt_rel_ct
            total_tt2 += tt_rel_ct2
        print "-"*124
        print "| %30s | %15s | %15s | %15s | %15s | %15s |" %("TOTAL",
                                                               "%s/%s" %(total_ee,total_ee2),
                                                               "%s/%s" %(total_et,total_et2),
                                                               "%s/%s" %(total_tt,total_tt2),
                                                               "%s/%s" %(total_rels,total_rels2),
                                                               "%s/%s" %(cons_graph_ct,doc_ct))
        print "-"*124
        return
    





def test_dir(path="tb_folds/0/train/"):
    _dir = Directory( path )
    _dir.stats()

def test_file(path="tb_folds/0/train/wsj_0026_orig.tml.xml"):
    doc = Document( path )
    print >> sys.stderr, "Testing document %s" %path
    # before saturation
    ee_graph = doc.get_event_event_graph()
    print >> sys.stderr, "E-E graph consistent before saturation: %s" %(ee_graph.saturate())
    et_graph = doc.get_event_time_graph()
    print >> sys.stderr, "E-T graph consistent before saturation: %s" %(et_graph.saturate())
    tt_graph = doc.get_time_time_graph()
    print >> sys.stderr, "T-T graph consistent before saturation: %s" %(tt_graph.saturate())
    tt_et_graph = doc.get_tt_et_graph()
    print >> sys.stderr, "T-T E-T graph consistent before saturation: %s" %(tt_et_graph.saturate())
    all_tt_et_graph = doc.get_tt_et_graph(isodate=True)
    print >> sys.stderr, "All T-T E-T graph consistent (incl. isodate inferred) before saturation: %s" %(all_tt_et_graph.saturate())

    ee_rels = doc.get_event_event_relations()
    print >> sys.stderr, "E-E relations before saturation (%s): %s" %(len(ee_rels),ee_rels)
    et_rels = doc.get_event_time_relations()
    print >> sys.stderr, "E-T relations before saturation (%s): %s" %(len(et_rels),et_rels)
    tt_rels = doc.get_time_time_relations()
    print >> sys.stderr, "T-T relations before saturation (%s): %s" %(len(tt_rels),tt_rels)
    tt_et_rels = doc.get_tt_et_relations()
    print >> sys.stderr, "T-T and E-T relations before saturation (%s): %s" %(len(tt_et_rels),tt_et_rels)
    all_tt_et_rels = doc.get_tt_et_relations(isodate=True)
    print >> sys.stderr, "All T-T and E-T relations (incl. isodate inferred) before saturation (%s): %s" %(len(all_tt_et_rels),all_tt_et_rels)
    print >> sys.stderr

    # after local saturation
    ee_graph = doc.get_event_event_graph(loc_sat=True)
    print >> sys.stderr, "E-E graph consistent after local saturation: %s" %(ee_graph.saturate())
    et_graph = doc.get_event_time_graph(loc_sat=True)
    print >> sys.stderr, "E-T graph consistent after local saturation: %s" %(et_graph.saturate())
    tt_graph = doc.get_time_time_graph(loc_sat=True)
    print >> sys.stderr, "T-T graph consistent after local saturation: %s" %(tt_graph.saturate())
    tt_et_graph = doc.get_tt_et_graph(loc_sat=True)
    print >> sys.stderr, "T-T E-T graph consistent after local saturation: %s" %(tt_et_graph.saturate())
    all_tt_et_graph = doc.get_tt_et_graph(isodate=True,loc_sat=True)
    print >> sys.stderr, "All T-T E-T graph consistent (incl. isodate inferred) after local saturation: %s" %(all_tt_et_graph.saturate())

    ee_rels = doc.get_event_event_relations(loc_sat=True)
    print >> sys.stderr, "E-E relations after local saturation (%s): %s" %(len(ee_rels),ee_rels)
    et_rels = doc.get_event_time_relations(loc_sat=True)
    print >> sys.stderr, "E-T relations after local saturation (%s): %s" %(len(et_rels),et_rels)
    tt_rels = doc.get_time_time_relations(loc_sat=True)
    print >> sys.stderr, "T-T relations after local saturation (%s): %s" %(len(tt_rels),tt_rels)
    tt_et_rels = doc.get_tt_et_relations(loc_sat=True)
    print >> sys.stderr, "T-T and E-T relations after local saturation (%s): %s" %(len(tt_et_rels),tt_et_rels)
    all_tt_et_rels = doc.get_tt_et_relations(loc_sat=True, isodate=True)
    print >> sys.stderr, "All T-T and E-T relations (incl. isodate inferred) after local saturation (%s): %s" %(len(all_tt_et_rels),all_tt_et_rels)
    print >> sys.stderr


    # after global saturation
    ee_graph = doc.get_event_event_graph(glob_sat=True)
    print >> sys.stderr, "E-E graph consistent after global saturation: %s" %(ee_graph.saturate())
    et_graph = doc.get_event_time_graph(glob_sat=True)
    print >> sys.stderr, "E-T graph consistent after global saturation: %s" %(et_graph.saturate())
    tt_graph = doc.get_time_time_graph(glob_sat=True)
    print >> sys.stderr, "T-T graph consistent after global saturation: %s" %(tt_graph.saturate())
    tt_et_graph = doc.get_tt_et_graph(glob_sat=True)
    print >> sys.stderr, "T-T E-T graph consistent after global saturation: %s" %(tt_et_graph.saturate())
    all_tt_et_graph = doc.get_tt_et_graph(isodate=True,glob_sat=True)
    print >> sys.stderr, "All T-T E-T graph consistent (incl. isodate inferred) after global saturation: %s" %(all_tt_et_graph.saturate())

    ee_rels = doc.get_event_event_relations(glob_sat=True)
    print >> sys.stderr, "E-E relations after global saturation (%s): %s" %(len(ee_rels),ee_rels)
    et_rels = doc.get_event_time_relations(glob_sat=True)
    print >> sys.stderr, "E-T relations after global saturation (%s): %s" %(len(et_rels),et_rels)
    tt_rels = doc.get_time_time_relations(glob_sat=True)
    print >> sys.stderr, "T-T relations after global saturation (%s): %s" %(len(tt_rels),tt_rels)
    tt_et_rels = doc.get_tt_et_relations(glob_sat=True)
    print >> sys.stderr, "T-T and E-T relations after global saturation (%s): %s" %(len(tt_et_rels),tt_et_rels)
    all_tt_et_rels = doc.get_tt_et_relations(glob_sat=True, isodate=True)
    print >> sys.stderr, "All T-T and E-T relations (incl. isodate inferred) after global saturation (%s): %s" %(len(all_tt_et_rels),all_tt_et_rels)
    print >> sys.stderr
    
    all_rels = doc.get_relations(isodate=True, saturation=False)
    all_rels_sat = doc.get_relations(isodate=True, saturation=True)
    # print all_rels_sat
    print >> sys.stderr, "All relations (incl. isodate inferred): %s/%s before/after saturation" %(len(all_rels),len(all_rels_sat))


    et_rels = doc.get_event_time_relations(loc_sat=True)
    print "E-T relations after local saturation (%s): %s" %(len(et_rels),et_rels)
    return doc


def test(path="otc_folds/0/train/wsj_0006_orig.tml.xml"):
    doc=test_file(path=path)
    print doc._graph
    try:
        dir_path = os.path.dirname(path)
        test_dir(  dir_path)
    except:
        print >> sys.stderr, "path problem with", dir_path
    return doc



####################################################################################
####################################################################################
    
if __name__ == '__main__':

    #import sys
    #doc = Document(sys.argv[1])
    #print doc.event_instances.keys()
    #print doc._graph.nodes().keys()
    import sys
    #doc=test(path=sys.argv[1])
    print sys.argv[1]
    doc = Document(sys.argv[1], do_prep=False)
    #print doc.get_event_event_relations(glob_sat=True,nodisjunction=False)
    print doc.get_event_event_relations(loc_sat=True,nodisjunction=False)
    print doc.string

    
    

    
