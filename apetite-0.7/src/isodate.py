#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# module dealing with datetimes/deltas
# subclassing datetime+dateutils
# to deal with TimeML/ISO ref
#
# vague= non absolute date (ex: january 3rd)
#
#  ?  - vague reference
#  ok - inclusion/precedence/etc  tests (result=allen relation),
#     -  even for vague refs. -> what ?
#     - deltas for vague refs.
#  ok - compute next/previous weekday

import locale
locale.setlocale(locale.LC_ALL, '')
import apetite

import datetime
import os
import sys
#import dateutil
import roman
from dateutil.relativedelta import *

#from clauses import timex2iso
from graph.Relation import Relation


# attribute configuration for temporal adverbials

configpath=apetite.__path__[0]+"/Config/date_value.txt"
# os independent path, for a change
_semadv_config_file=(os.sep).join(configpath.split("/"))

def get_config(filename=_semadv_config_file):
    translation={}
    lines=open(filename).readlines()
    lines=[x.split() for x in lines if x.strip()!="" and not(x.startswith("#"))]
    for (lex,value) in lines:
        # value specifies which part of a Time/Date it is a value for
        # ex avril->m:4
        if ":" in value:
            value=value.split(":")
        else:# it is a number
            value=(None,value)
        translation[lex]=value
    return translation

_default_config=get_config()

# timex= dictionnaire d'attributs temporels+type
# sortie: une expression ISO-TimeML
# TODO: liens avec datetime pour les calculs
# -> sous-classe de datetime ?
#       date&time->date,
#       duration ->delta

# a=annee, s=saison, m=mois ou minute j=jour h=heure n=nombre p=specifieur
# durees: u=unite n=nombre
_time_attributes="a s m j h n".split()
_duration_attributes="u n".split()
# pb: t, s, ... sont des chaines plus complexes
#
# in theory these attributes should alreayd be sorted
# but there is some overlap (to be changed)
# ex: s is century OR specifier of sth else
# in that case the lexical item should be unambiguous
# -> robust treatment, and clean up the mess
# TODO: this will be __init__ in the future class for interpreted timex

# todo: p="GMT" or timezone local.
_clean_att={"j":"d","a":"y"}

# may output: value, mod, anchor, temporal function (Speech-Time/Temporal-Focus)
# cleans up a little bit the structure for robustness
# translation: table de conversion valeurs des traits en timeML standard
# TODO:
#   - separate cleaning-up the attributes and computing values
#   - clean-up that mess, more declarative (in date_values.txt)
#   ok?- NON INVOLUTIF (nettoyer une timex propre la detruit a cause ambiguite 
def semantique_timex(timex,translation):
    cleaned={}
    quant="XX"
    #sys.stderr.write("%s\n"%`timex`)
    # duration
    if timex.has_key("u"):
        unitcode=timex["u"].encode("latin")
        # unit may have been already cleaned up earlier
        # must be ready for it...                
        unit=translation.get(unitcode,["",unitcode])[1]
        # cas de DT -> duree=D
        if unit.upper()=="DT":
            unit="D"
        many=timex.get("n",quant)
        if many!=quant:
            many=translation.get(many,(None,many))[1]
            #del timex["n"]
        cleaned["u"]=unit
        cleaned["nb"]=many
    # all other attribs
    for (att,val) in timex.items():
        if att in _time_attributes:
            # is this is not in the translation, it must be a number
            val=val.encode("latin").lower()
            (type,result)=translation.get(val,(None,val))
            #print >> sys.stderr, att, val, type, result
            if type=="m":
                type="mo"
            if att=="n" and (timex.get("h","?")!="?"):
                type="mm"
            if type=="a":
                type="y"
            if result=="?":# there is quantity somewhere
                quant="X"
            #case of roman numerals (centuries)
            # TODO: could be a string digit too (one million and seven)
            # or a decade...
            # .... needs to be translated too
            elif type==None and att!="h" and not(result.isdigit()):
                try:
                    type="y"
                    result=str(100*(roman.fromRoman(result[:-1].upper())-1))
                    #cleaned["u"]="siecle"
                except:
                    # all other cases will have to be dealt with
                    type="specifier"
                    #result="not recognised"
                    print >> sys.stderr, "number rejected: ",result
                #print >> sys.stderr, "roman: ",result
            elif type==None:
                # a number or a specifier for hours
                # in that case keep the name of the original attribute
                # or its normalised translation 
                type=_clean_att.get(att,att)
            cleaned[type]=result
        else:
            cleaned[att]=val
    #cleaned["m"]=cleaned.get("mm","?")
    strip_timex(cleaned)
    return cleaned

def strip_timex(cleaned):
    for (x,y) in cleaned.items():
        if y=="?":
            cleaned.pop(x)

# timex is a "cleaned-up" date or time
# __str__/ iso
# TODO:
#    - specifieurs (s=matin,...)
def timex2iso(timex,timemlconfig=_default_config):
    timextype=timex["timextype"]
    if timextype.startswith("dur") or timextype=="tatom":
        return duration2iso(timex,timemlconfig)
    default=["XXXX"]+["XX"]*5
    res=[""]*6
    date=["y","mo","d"]
    time=["h","mm","ss"]
    # should be done earlier in semantique_timex
    modif=timex.get("specifier","?")
    #if modif!="?":
    #    print >> sys.stderr, "Specifieur found ?", modif
    if modif.lower() in ["ev","mo","af","ni","dt","dt"]:
        timex["h"]=modif
    for (i,x) in enumerate(date):
        res[i]=timex.get(x,default[i])
        if x=="y" and timex.get("u")=="siecle":
            res[i]=res[i][:-2]
        if x=="d" and res[i].startswith("w"):
            res[i-1]="WXX"
            res[i]=res[i][1:]
    for (i,x) in enumerate(time):
        res[i+3]=timex.get(x,default[i+3])
    # remove trailing uninstanciated part
    timestring="T"+":".join(res[3:])
    while timestring.endswith("XX"):
        timestring=timestring[:-3]
    datestring="-".join(res[:3])
    while datestring.endswith("XX"):
        datestring=datestring[:-3]
    if datestring=="X":
        datestring=""
    # if hour is there, date must be a precise day
    if timestring!="" and timex.get("d","?")=="?":
        datestring=""
    return (datestring+" "+timestring).strip()


def duration2iso(timex,timemlconfig):
    nb=timex.get("nb","XX")
    code=timex.get("u")
    # TODO dur�es a calculer (dureerel: "jusqu'au 1er f�vrier")
    if code is None:
        return "P"
    else:
        code=code.encode("latin")
        code=timemlconfig.get(code,(None,code))[1]
        if code.upper()=="DT":
            code="D"
    if not(nb.isdigit()):
        nb=timemlconfig.get(nb.lower(),(None,"XX"))[1]
    #print >> sys.stderr, nb
    if code in ["H","MM"]:
        prefix="PT"
    else:
        prefix="P"
    return prefix+nb+code.upper()


#
def string_number2int(number):
    res=0
    return res

###################################
# calcule valeur absolue d'une date sous-sp�cifi�e si possible
# arguments:
#    - une date absolue a partir de laquelle on peut
#      trouver la valeur (TimeMLdate)
#    - orientation (pass�/futur)
# ex:
#  1) Speech Time= 1998-7-12 (dimanche)
#     "Lundi", futur => 1998-7-13
#  2) Focus=1998
#     "Le 3 mars", none => 1998-3-3
#  3) Focus=1998-3
#     "Le 3", none => 1998-3-3
#  3) Focus=1998-12-3
#     "En avril", none(?past) => 1998-4
#  4) Speech Time= 1998-7-12
#   "Lundi matin" futur -> 1998-7-13TMO
#  
# modifies timex
def relative_timex_value(timex,date,orientation="same"):
    # date incomplete ("le 3 mars")
    print >> sys.stderr, "completion of %s with %s" %(timex,date)
    if orientation=="same" and not(timex["d"].startswith("w")):
        for trait in ("y","mo","d"):
            if (timex.get(trait,"?")=="?") and date.has_key(trait):
                timex[trait]=date[trait]
                # ? arret des qu'un nouveau trait est instanci�
            #return timex
    else:# date = jour de la semaine (lundi...)
        # timex = isoweekday = +2 / codage calendrier (monday=0/monday=2)
        theday=int(timex["d"][-1])-2
        if orientation=="past":
            func=date.previous
        else:
            func=date.next
        timex.update(from_date(func(theday)))



#  retro-conversion datetime -> timex
def from_date(date):
    return {"y" :str(date.year),
            "mo":str(date.month),
            "d" :str(date.day)}
    

def parse_date(expr):
    all=expr.split('-')
    res={}
    res['y']=all[0]
    if len(all)>1:
        res['mo']=all[1]
    if len(all)>2:
        res['d']=all[2]
    return res

def parse_time(expr):
    if expr.startswith('T'):
        expr=expr[1:]
    all=expr.split(':')
    res={}
    res['h']=all[0]
    if len(all)>1:
        res['mm']=all[1]
    if len(all)>2:
        res['s']=all[2]
    return res

def parse_duration(expr):
    """not implemented"""
    print >> sys.stderr, "warning: parse_duration not implemented"
    return None
    

# useful when reading timeML
#  >> iso2date('2003-5-29 T03:15') -> ok
#  >> iso2date('T03:15') -> None
#  >> iso2date('P3D') -> None
def iso2date(iso_expression):
    #duration
    if iso_expression.startswith('P'):
        #unit,dur=parse_duration(iso_expression)
        #return TimeMLduration({'u':unit,'nb':dur})
        return None
    elif iso_expression.startswith('T'):# time only ? that's an error
        return None
    elif iso_expression[0].isdigit():# absolute date proper
        parts=iso_expression.split()
        timex=parse_date(parts[0])
        if len(parts)>1:#time also
            timex.update(parse_time(parts[1]))
        timex["timextype"]="dateabs"
    else:
        return None
    return TimeMLdate(timex)

# time points / allen relation correspondance
# intervalle=(d,f) (points in time)
# d1<f1 
# 
_intervals="""d1=d2 f1=f2 =
f1<d2 b
f1=d2 m
f2<d1 bi
f2=d1 mi
d1=d2 f2<f1 si
d1=d2 f1<f2 s
f1=f2 d2<d1 f
f1=f2 d1<d2 fi
d1<d2 f2<f1 di
d2<d1 f1<f2 d
d2<f1 d1<d2 f1<f2 o
d1<f2 d2<d1 f2<f1 oi
""".split("\n")


def find_relation(x,y):
    # bloated but shorter & easier to type...
    #print x.__dict__
    #print y.__dict__
    m={"d1":x.start(),
       "d2":y.start(),
       "f1":x.end(),
       "f2":y.end(),
       "=":lambda u,v: u==v,
       "<":lambda u,v: u<v,
       ">":lambda u,v: u>v
       }
    for condition in _intervals:
        found=True
        l_conds=condition.split()
        #print l_conds
        relation=l_conds[-1]
        l_conds=l_conds[:-1]
        for c in l_conds:
            #ouch
            found=found & m[c[2]](m[c[0:2]],m[c[3:5]])
            #print found
        if found:
            return relation
    # shouldn't happen with specified dates -> error swh else
    return None

#################################################################
# an absolute datetime (and a duration)
#
# init with a cleaned-up timex (attribute-value pairs)
# difference with datetime: can have granularity (year/month/day)
# a start/end/duration
# TODO:
#  ok  - saisons -> debut/fin
#################################################################
class TimeMLdate:
    _keys=["a","y","mo","d","h","mm","s"]
    _units=["years","years","months","days","hours","minutes","seconds"]
    # first month in season (days ?)
    _seasons={"SP":3,"WT":12,"SU":6,"FA":9}

    # date/time seulement (pas durees)
    def __init__(self,attributes,duration=None,config=_default_config):
        self.iso=timex2iso(attributes,config).upper()
        self.attrs=attributes
        self.attrs["value"]=self.iso
        if duration==None:
            for (i,x) in enumerate(self._keys):
                if attributes.has_key(x) and attributes[x]!="?":
                    duration=self._units[i]
                else:
                    pass
            # print >> sys.stderr, "- unit:",duration,attributes
            # duration of season = 4 months
            if self.attrs.get("mo",1) in self._seasons.keys():
                dur_value=4
            else:
                dur_value=1
            # tricky
            self._duration=relativedelta(**{duration:dur_value})
            self._grain=duration
        # cleaning-up (should replace semantique_timex)
        for (x,y) in self.attrs.items():
            if y=="?":
                self.attrs.pop(x)
        # a ameliorer pour l'instant si saison on zappe avant
        #if !self.attrs.get("mo",1).isdigit():
        # si heures "morning", etc a traiter
        hour=self.attrs.get("h","")
        if not(hour.isdigit()):
            hour="0"
        # saisons
        month=self.attrs.get("mo",1)
        if month in self._seasons.keys():
            month=self._seasons[month]
        self._begin=datetime.datetime(int(self.attrs.get("y")),
                                      int(month),
                                      int(self.attrs.get("d",1)),
                                      int(hour),
                                      int(self.attrs.get("mm",0)),
                                      int(self.attrs.get("s",0)))
        self._end=self._begin+self._duration


    def __getitem__(self,tag):
        return self.attrs[tag]

    def has_key(self,tag):
        return self.attrs.has_key(tag)

    def start(self):
        return self._begin

    def end(self):
        return self._end

    def grain(self):
        return self._grain
    
    def is_vague(self):
        return "XX" in self.iso


    # weekday for absolute dates, (error otherwise)
    def weekday(self):
        #print >> sys.stderr, self
        year  = int(self.attrs.get("y"))
        month = int(self.attrs.get("mo"))
        day   = int(self.attrs.get("d"))
        return datetime.date(year,month,day).weekday()


    # date of the following weekday (e.g. next monday)
    # TODO:
    #   - renvoie type TimeMLdate
    #   - +/-7 devrait etre le meme jour sauf specifieur contraire
    #     ("mercredi" vs "mercredi prochain" si la date du texte est mercredi, style AFP)
    def previous(self,day):
        delta=(self.weekday()-day)%7
        #if delta==0:
        #    delta=7
        tdelta=datetime.timedelta(days=delta)
        return self._begin-tdelta

    # date of the previous weekday (e.g. last monday)
    def next(self,day):
        delta=(day-self.weekday())%7
        #if delta==0:
        #    delta=7
        tdelta=datetime.timedelta(days=delta)
        return self._begin+tdelta



    def __repr__(self):
        return self.iso

    def __str__(self):
        return self.iso

    # is there a relation with other date
    # (must be a date with a value)
    def relation_with(self,other):
        #print self,other
        if isinstance(other,TimeMLdate):
            relation=find_relation(self,other)
            return Relation(relation)
        else:
            return None

#################################################################
# a duration
#
# init with a cleaned-up timex (attribute-value pairs)
# difference with datetime.timedelta:
#    - can be underspecified ( "days" -> value=PXXD in TimeML)
#    - keep track of originating timex
#    - iso representation conforming to TimeML
#################################################################
class TimeMLduration:
    _keys=["u","n","m","s"]
    _units=[]

    # 
    def __init__(self,attributes):
        self.iso=duration2iso(attributes).upper()
        self.attrs=attributes
        self.attrs["value"]=self.iso

    def __repr__(self):
        return self.iso

    def __str__(self):
        return self.iso




#tests
if __name__=="__main__":
    import calendar
    # une date au "hasard" (dimanche) ! 
    a={"y":"1998","mo":"7","d":"12","timextype":"dateabs"}
    b={"y":"1998","timextype":"dateabs"}
    ta=TimeMLdate(a)
    tb=TimeMLdate(b)
    print ta.relation_with(tb)#Relation(['trel_during'])
    print ta.weekday()#6
    print ta.next(calendar.MONDAY)#1998-07-13 00:00:00
    print ta.previous(calendar.SUNDAY)#1998-07-05 00:00:00
    c={"mo":"9","d":"9"}
    d={"d":"w3"}
    print relative_timex_value(c,tb)#{'y': '1998', 'mo': '9', 'd': '9'}
    print relative_timex_value(d,ta,orientation="past")#1998-07-15 00:00:00
    timex={'timextype': 'daterelTF1', 'd': '29', 'h': '03',
           'mo': '5', 'm': '15', 'pos': '55-57', 'value': '2003-5-29 T03:15', 'y': '2003'}
    timex={'timextype': 'daterelTF1', 'position': ['59', '60'],
           's': u'journ\xe9e', 'type': 'timex3', 'value': ''}
    timex={'timextype': 'tatom', 'nb': '1', 'n': '1',
      'u': u'journ\xe9e', 'position': ['441', '442'], 'type': 'timex3'}
    new=semantique_timex(timex,_default_config)
    print new


    a=iso2date('2003-5-29 T03:15')
    b=iso2date('2003-SU')
    print a,"/",b,"->",a.relation_with(b)
