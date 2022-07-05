"""
Class for Markable ...
"""

import re
from graph.Relation import Relation
from isodate import iso2date

class Markable(object):
    """ abstract class for event, timex, signal and general markable in texts
    elt is from reader output , must have at least text and char_offsets attribute
    
    ex 
	- signal: {'char_offsets': (1264, 1273), 'text': 'tentative', u'sid': u's664'}
        - timex: {u'functionInDocument': u'NONE', 'text': 'September', u'anchorTimeID': u't138', u'value': u'P1M', u'tid': u't139', 'char_offsets': (726, 735), u'type': u'DURATION', u'temporalFunction': u'true'}
        - event {'char_offsets': (2664, 2674), 'text': 'earthquake', u'eid': u'e668', u'class': u'OCCURRENCE'}
    """
    def __init__(self, annotation_dict):
        # expose annotation attributes as object attributes
        self.__dict__.update( annotation_dict )
        self.extent = self.char_offsets
        self.tokens = []
        #self.set_tokens( offset_map, tokens )
        return

    def set_tokens(self, offset_map, tokens ):
        # print self.text, self.extent
        self.tokens = []
        self.left_context = []
        self.right_context = []
        try:
            start_token = offset_map[self.extent[0]]
            end_token = offset_map[self.extent[1]]
            idx1 = tokens.index(start_token)
            idx2 = tokens.index(end_token)
            self.tokens = tokens[idx1:idx2+1] # inclusive slice
            self.left_context = tokens[:idx1]
            self.right_context = tokens[idx2+1:]
        except KeyError, ValueError:
            # TODO: deal with timex in header!!!!
            print >> sys.stderr, "WARNING: %s could not be mapped into tokens" %self
        return 

    def is_included(self,other):
        a1,b1=self.extent
        a2,b2=other.extent
        return (a2<=a1) and (b1<=b2)

    # TODO: degre d'overlap 
    def overlaps(self,other):
        a1,b1=self.extent
        a2,b2=other.extent
        return (a1<=a2<=b1) or (a2<=a1<=b2)

    def __repr__(self):
        return "%s: ID:%s EXT:%s TXT:%s TOKENS:%s" %(type(self).__name__,
                                                     self.id,
                                                     self.extent,
                                                     self.text,
                                                     self.tokens)



##################
class Event(Markable):

    def __init__(self, annotation_dict):
        Markable.__init__(self, annotation_dict)
        self.cat = annotation_dict['class']
        return

    def __repr__(self):
        return "%s CAT:%s" %(Markable.__repr__(self),
                             self.cat)
    

###################    
re_heure=re.compile(r"T0(?=[1-9])")


class Timex3(Markable):
    ''' Timex object defined from annotation
    
    <TIMEX3 tid="t100" type="DURATION" temporalFunction="false" functionInDocument="NONE" value="P2Y" >a couple of years</TIMEX3>    
    '''

    def __init__(self, annotation_dict):
        Markable.__init__(self, annotation_dict)
        self.value = self.normalise_value(annotation_dict['value'])
        self.temp_fcn = annotation_dict.get('temporalFunction', 'false')
        self.fcn_in_doc = annotation_dict.get('functionInDocument', 'None')
        self.referent=self.set_referent()
        # ... anchorID etc.
        return

    def normalise_value(self, val):
        val=val.upper().replace(" ","")
        val=val.replace("-0","-")
        val=val.replace("T00","T0")
        val=val.replace("-T","T")
        val=re_heure.sub("T",val)
        val=val.replace("PXX","PX")
        val=val.replace("PTXX","PTX")
        return val.encode("latin")


    def set_referent(self):
        """
        computes date referent for absolute dates
        -> can then be used in temporal calculations
        """
        try:
            return iso2date(self.value)
        except:
            return None
        

    def relation_with(self,other):
        """
        returns allen temporal wrt to other absolute date
        """
        if self.referent is not None and other.referent is not None:
            return self.referent.relation_with(other.referent)
        else:
            return None


    def __repr__(self):
       return "%s VAL:%s TEMP_FCN:%s FCN_IN_DOC:%s" %(Markable.__repr__(self),
                                                      self.value,
                                                      self.temp_fcn,
                                                      self.fcn_in_doc)
                            


##################
class Enamex(Markable):
    '''
    <ENAMEX TYPE="PERSON">Jim Laurie</ENAMEX>
    '''
    def __init__(self, annotation_dict):
        Markable.__init__(self, annotation_dict)
        self.type = annotation_dict.get('TYPE')        
        return

    def __repr__(self):
        return "%s TYPE:%s" %(Markable.__repr__(self),
                              self.type)
                                    
    



##################
class Numex(Markable):
    '''
    <NUMEX TYPE="PERCENT">seventy percent</NUMEX> 
    '''
    def __init__(self, annotation_dict):
        # ...
        Markable.__init__(self, annotation_dict)
        return

   
##################
class Signal( Markable ):
    '''
    <SIGNAL sid="s13" >before</SIGNAL> 
    '''
    def __init__(self, annotation_dict):
        # ...
        Markable.__init__(self, annotation_dict)
        return


##################
class EventInstance( object ):
    '''
    e.g.
    <MAKEINSTANCE aspect="PROGRESSIVE" eiid="ei375" tense="PRESENT" eventID="e1" />
<MAKEINSTANCE aspect="NONE" eiid="ei376" tense="NONE" eventID="e84" />

    '''
    
    def __init__(self, annotation_dict, events):
        self.__dict__.update( annotation_dict )
        self.event = events[self.eventID] # pointer to event
        self.modality = annotation_dict.get('modality','NONE')
        self.polarity = annotation_dict.get('polarity', 'POS')
        return

    def __repr__(self):
        return "%s: ID:%s EVT-ID:%s TENSE:%s ASP:%s POL:%s" %(type(self).__name__,
                                                              self.id,
                                                              self.eventID,
                                                              self.tense,
                                                              self.aspect,
                                                              self.polarity)




class Sentence( Markable ):
    '''
    <s>Hotels <EVENT eid="e279" class="STATE" >are</EVENT> only <NUMEX TYPE="PERCENT">thirty percent</NUMEX> full.</s>
    '''
    
    def __init__(self, annotation_dict):
        # ...
        Markable.__init__(self, annotation_dict)
        return



class Text( Markable ):
    
    def __init__(self, annotation_dict):
        # ...
        Markable.__init__(self, annotation_dict)
        return

############################################################
    

class Token(object):

    def __init__(self, annotation_dict):
        self.id = annotation_dict['id']
        self.sentence_id = annotation_dict['s_id']
        self.word = annotation_dict['text']
        self.pos = annotation_dict['pos']
        self.extent = annotation_dict['extent']
        return

    def get_word(self):
        return self.word

    def get_pos(self):
        return self.pos

    def __repr__(self):
        return "%s: WD:%s POS:%s EXT:%s" %(type(self).__name__,
                                           self.word,
                                           self.pos,
                                           self.extent)



class TemporalRelation( object ):
    '''
    <TLINK signalID="s141" relatedToTime="t139" eventInstanceID="ei763" relType="SIMULTANEOUS" />
    '''
    
    def __init__(self, annotation_dict):
        self.__dict__.update( annotation_dict )
        x=annotation_dict.get("eventInstanceID")
        if x is None:
            x=annotation_dict.get("timeID")
        y=annotation_dict.get("relatedToEventInstance")
        if y is None:
            y=annotation_dict.get("relatedToTime")
        rel=annotation_dict.get("relType")
        rel = Relation(rel.lower().encode()) 
        
        self.arg1, self.arg2 = x,y
        self.relation = rel
        return

    def __repr__(self):
        return "%s ==%s==> %s" %(self.arg1,self.relation,self.arg2)
