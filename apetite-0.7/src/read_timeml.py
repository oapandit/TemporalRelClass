#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# extraction des données timeml, version 2
# peut lire la sortie de callisto/tango
# et aussi la sortie de log2timeml (obsolete)
# 
# usage:
# read_timeml_2.0 [file.xml]
#
# (ok) option de normalisation : si vient d'annotation timex2
# pas d'identité des timex -> ajouter 

# renvoie une liste d'entites+chaine correspondante+position dans le texte [+valeur]
# et une liste de tlink entre eux.


# TODO:
#  en lecture
#   - normaliser description des relations ou bien prevoir table
#     d'equivalences timeml/autres
#   - lire relation générale (disjonctive/allen/annotation)
# ok- retrouver values, type evt, tense, signal, etc (dans desc des noeuds)
# ok- associer position dans le texte brut ou prevoir importation
#     comme attribut de la position (<item pos=..>mot</item>)
#   - pas tres robuste sur tags imbriqués ...
# URGENT  - ne respecte pas la norme TimeML entre makeinstance et autres...
#     -> fusionner infos de makeinstance/event
#      MAIS: norme en cours d'évolution
#   - lire la date de publication a part (trait function in document=CREATION_TIME PUBLICATION_TIME)
#          ou bien DATE_TIME en timex/ace


try:
    import xml.etree.ElementTree as ElementTree
except:
    try:
         import cElementTree as ElementTree
    except:
        import ElementTree
       
#import cElementTree as ElementTree
from elementtools import  html2unicode
from tokenise import tokenise
from graph.Relation import Relation
import sys
import string
#import mx.DateTime as Date
from pprint import pformat
import copy


NORMALISE=False

def text_span(entity):
    res=entity.text
    if res==None:
        res=""
    for x in entity:
        res=res+text_span(x)
        if x.tail is not None:
            res=res+x.tail
    return res


# for robustness, input can be in html
# coding with entity refs. this puts them as unicode (well, latin)
# also used if text is parsed after a fromstring-tostring
# ideally, should be used to read timeml too

# otherwise, should be in tokenise ? to make it accept tagged/html input
# TODO: take cleaner code from html2txt.py in ~/bin/

import HTMLParser as html
from htmlentitydefs import entitydefs
class MyHtml(html.HTMLParser):
    
    def reset(self):                       
        self.result = ""
        html.HTMLParser.reset(self)
    
    def close(self):
        html.HTMLParser.close(self)
        return self.result
    
    def handle_data(self,data):
        self.result+= data
    
    def handle_charref(self,data):
        self.result += "\\"+data
    
    def handle_entityref(self,data):
        self.result += entitydefs[data]

    # to safeproof tokenising
    def handle_starttag(self,tag,attr):
        self.result+=" "

    # idem
    def handle_endtag(self,tag):
        self.result+=" "

# wraps the preceding stuff
def read_html(input):
    htmlread=MyHtml()
    htmlread.feed(input)
    res=htmlread.close()
    #res=(unicode(res,"latin"))
    #res=res.encode("latin")
    res=res.decode("latin").encode("latin")
    return res

############################################
#
# should be able to accept anything that
# is correctly defined in module Relation
# or left as is
############################################
def parse_relation(tlink):
    # relation between x and y
    x=tlink.get("eventInstanceID")
    if x is None:
        x=tlink.get("timeID")
    y=tlink.get("relatedToEventInstance")
    if y is None:
        y=tlink.get("relatedToTime")
    rel=tlink.get("relType").lower()
    rel=Relation(rel)
    return (x,y,rel)

# instance id -> event id + ?location
def event_id(eiid,instlist):
    for x in instlist:
        if x.get("eiid")==eiid:
            return x.get("eventID")
    return None


def get_event(eid,event_list):
    for x in event_list:
        if x.get("eid")==eid:
            return x
    return None


import re


# fait la liste des positions des entites marquées (event/timex)
# complete eventuellement le texte 
# doit prendre la partie TEXT exclure le reste
# TODO:
#   ok - dumper la tokenisation dans un fichier externe pour validation
#   - verifier que les entites hors textes sont exclues -> position non pertinente
#     l'entité est définie par les autres champs.
#   - l'annotation "in-line" de  TimeML est un peu problématique, l'annotation avec position explicite devrait
#     etre faite systematiquement
def extract_markup_position(reference,idtag,idprefix,type,fresh=None,logfile=None):
    # on separe aux balises
    base1=reference.replace(">",">µ").replace("<","µ<").split("µ")
    base1=[x.strip() for x in base1 if x.strip()!=""]
    evt_position={}
    count=0
    evt=False
    skip=False
    # if we're treating from-timex2 entries, id tags may be missing
    if NORMALISE:
        re_markup=re.compile(r'<%s [^>]*(%s="(?P<id>%s[0-9,-]+)").*?>'%(type,idtag,idprefix))
    else:
        re_markup=re.compile(r'<%s [^>]*%s="(?P<id>%s[0-9,-]+)".*?>'%(type,idtag,idprefix))
    for (i,x) in enumerate(base1):
        #sys.stderr.write("%s %s\n" % (x, count))
        if x.startswith("<"+type):
            #print "%"+x
            evt=True
            skip=False
            evtsearch=re.search(re_markup,x)
            if evtsearch is not None:
                evt_no=evtsearch.group("id")
                #print "no id"
                # if timex2, ad id tags referents
                if evt_no==None and NORMALISE:
                    evt_no="t"+str(fresh.next())
                    base1[i]=x.replace("<"+type,'<%s %s="%s"'%(type,idtag,evt_no))
            else:
                evt_no=-1
            #print evt_no
        elif "<" in x and (("timex" in x.lower()) or "event" in x.lower()):
            evt=False
            skip=False
        # preamble annotation (skipped) or trailer or other (signal)
        elif "<" in x:
            skip=True
        else:
            # pose un pb d'accent...
            n_text=read_html(x)
            n_text=x
            wordlist=tokenise(html2unicode(n_text))
            wordlist=[(yy,str(xx+1+count)) for (xx,yy) in enumerate(wordlist)]
            if logfile:
                logfile.write(("\n".join([" ".join(z) for z in wordlist])+"\n").encode("latin-1"))
            if evt:
                evt_position[evt_no]=`count+1`
                #if logfile:
                #    logfile.write("%i:%s/%i\n"%(i,x,count))
                evt=False
                count=count+len(wordlist)
                evt_position[evt_no]+="-"+`count`
            else:
                count=count+len(wordlist)
    #print evt_position
    return evt_position,"".join(base1)




# find the text span of a given entity e (timex or event)
# should tree be the text ? .... or preprocess the text to extract
# position (better), and add it as an attribute of every lexical item
# d_evt: dictionary of evts
# d_times: dictionary of timexes
def get_location(e,base,d_evt,d_times):
    #look in evt list and if not present look in timex list
    # or None
    return d_evt.get(e,d_times.get(e,None))


# get entities (timex+events) and relations (tlinks)
# renvoie table for event, timex, tlink
#   - entities indexed by position; values=
#        * event : id ("e1"), text, value
#        * timex : id ("t2"), text, value 
#   - relations indexed by argument pair; value=relation
#     ("e1","e2"): "trel_meets"
# arguments:  base=xml timeml  string
###########################################
# TODO: 
#    - pb instances multiples d'un evt...
###########################################
def extract_from_string(base,fresh=None,logfile=sys.stderr):
    #read
    data={"timex":{},"event":{},"tlink":{},"inst":{}}
    base=base.replace("\n"," ")
    # difference in timeml formats: text may be between<TEXT>
    # with <header> and <trailer> to be ignored for now
    tree_base=ElementTree.fromstring(base)
    relevant=tree_base.findall(".//TEXT")
    if relevant!=[]:
        #sys.stderr.write("this timeml has a header/text %s\n"%`relevant`)
        base=ElementTree.tostring(relevant[0])
        
    d_evt,base=extract_markup_position(base,"eid","e","EVENT",fresh)
    d_times,base=extract_markup_position(base,"tid","t","TIMEX3",fresh,logfile)

    if DEBUG:
        d_times.keys().sort()
        d_evt.keys().sort()
        logfile.write(pformat(d_times)+"\n")
        logfile.write(pformat(d_evt))
    # this is redone after the extraction of markup position
    # in case markup  has to be completed (from timex2 annotation
    # for instance
    tree=ElementTree.fromstring(base)
    events=tree.findall(".//EVENT")
    timex=tree.findall(".//TIMEX3")
    inst=tree_base.findall(".//MAKEINSTANCE")
    tlinks=tree_base.findall(".//TLINK")

    #sys.stderr.write(repr(d_times))
    i=0
    for t in timex:
        #print t.get("tid")
        location=get_location(t.get("tid"),base,d_evt,d_times)
        value=t.get("value")
        if value is None:
            value=t.get("VAL")
        #if value is not None and value!="":
        #    value=Date.ISO.str(Date.ISO.ParseAny(value))
        text=t.text
        if text is not None:
            text=text.encode("latin-1").replace("\n"," ")
        else:
            text=""
        #sys.stderr.write('%s "%s" %s "%s"\n' % (t.get("tid"),text,location,value))
        # xx change: value -> attributs
        data["timex"][location]=(t.get("tid"),text,t.attrib)
        
    for e in events:
        location=get_location(e.get("eid"),base,d_evt,d_times)
        if e.text is not None:
            chaine=e.text.encode("latin-1")
        else:
            # TODO: rendre plus robuste en renvoyant tout le contenu textuel
            sys.stderr.write("warning: embedded entity in an event or no description : %s\n" % e.get("eid"))
            chaine=text_span(e)
        #print "%s %s %s" % (e.get("eid"),chaine,location)
        data["event"][location]=e.get("eid"),chaine,e.attrib

    # lit aussi infos liees aux instances, pas clair ou va quoi
    # car norme changeante et notion d'instance a 1/2 claire (?)
    # instances indexés par id d'event reliés
    # attention: susceptible d'etre changé par normalisation de l'index de l'evt
    for one_i in inst:
        #ex <MAKEINSTANCE aspect="PROGRESSIVE" eiid="ei2256" tense="NONE" eventID="e33" />
        eid=one_i.get("eventID")
        data["inst"][eid]=one_i.attrib
    #
    # attention supposerait relation unique ... donc non saturation
    # -> *changed* in parse_relation
    for tl in tlinks:
        (x1,x2,rel)=parse_relation(tl)
        #if event instance finds its id, otherwise, direct event reference of timex id
        if x1.startswith("ei"):
            x1=event_id(x1,inst)
        if x2.startswith("ei"):
            x2=event_id(x2,inst)
        #print "%s, %s : trel_%s" % (x1,x2,rel)
        #clone?
        data["tlink"][(x1,x2)]=rel
        #clone=copy.deepcopy(rel)
        #data["tlink"][(x1,x2)]=clone
    return data,base


def print_extracted_data(dico):
    tlink=dico["tlink"]
    events=dico["event"]
    timex=dico["timex"]

    for dict in timex,events,tlink:
        #l.sort()    
        for key in dict.keys():
            value=dict[key]
            if isinstance(value,Relation):# relation tlink
                print "%s, %s : %s" % (key[0],key[1],`value`)
            elif len(value)==3:# timex & event
                print '%s "%s" %s "%s"' % (value[0],value[1],key,value[2])
            else:# error
                sys.stderr.write("error : entity %s\n"%`key`)
       
# generates new integers to provides
# unused values
# i=starting point for values-1
# NB: existe deja (itertools python 2.3/2.4?)
def fresh_reference(i=0):
    while True:
        i=i+1
        yield i


DEBUG=False
if __name__=="__main__":
    if len(sys.argv)>1:
        #sys.stderr.write("reading file %s\n"%sys.arv[1])
        #tree=ElementTree.parse(sys.argv[1])
        base=open(sys.argv[1]).read()
        # option 
        if len(sys.argv)==3:
            NORMALISE=True
    else:
        # testing
        tree=ElementTree.parse("pm_aqaba.xml")
        
    #base=base.replace("\n"," ")
    #sys.stderr.write("file read and parsed\n")
    
    #ElementTree.parse("attac_181204.pre.xml")
    fresh_ref=fresh_reference(0)
    
    d,new_text=extract_from_string(base,fresh_ref)
    print_extracted_data(d)

    if NORMALISE:
        output=open("n_"+sys.argv[1],"w")
        output.write(new_text)
        output.close()

    #from timeml_compare import *
    #d1=normalise(d)
    #g=make_graph(d1)
    #g.saturate()
