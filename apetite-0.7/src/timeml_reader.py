#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
lecture timeml 


TODO: 
   - default encoding is Latin1 should be changed to unicode
"""

import os
from xml.dom import minidom
from collections import defaultdict
from markable import *


class TimeMLReader:
    pass


class TimeML11Reader( TimeMLReader ):
    """
    Parse un fichier au format TimeML 1.1
    """
    def __init__(self, path):
        self.path = os.path.abspath( path )
        self._rawdata = ""
        self._char_offset = 0 # offset counter
        self._char_offset_stack = [] # for embedded annotations
        # annotations
        self._annotations = {
            'TEXT': {},
            'EVENT':{},
            'TIMEX3':{},
            'SIGNAL':{},
            'MAKEINSTANCE':{},
            'TLINK':{},
            #'SLINK':{},
            #'ALINK':{},
            's':{},
            'ENAMEX':{},
            'NUMEX':{},
            }
        # for synchronization 
        self._offset_map = defaultdict(dict) 
        # load and parse XML
        self._load_source( path )
        self._parse_TimeML()
        # synchronize annotations
        self._synchronize()
        return

    # private methods
    def _load_source(self, sourcefile):
        self._xmldoc = minidom.parse( sourcefile )
        return
        
    def _parse_TimeML(self):
        timemlroot = self._xmldoc.firstChild
        self._parse_element( timemlroot )
        if _DEBUG: print >> sys.stderr,self._rawdata, (0, len(self._rawdata))
        if len(self._char_offset_stack) != 0:
            print >> sys.stderr, "Error: stack size is %s" %len(self._char_offset_stack) 
            sys.exit(1)
        return

    def _parse_element(self, node, s="  "):
        ''' recursive, depth-first traversal of XML tree that keeps
        track of character offsets '''

        if _DEBUG: print >> sys.stderr,"%s IN NODE: %s, %s" \
           %(s,node.nodeName,self._char_offset)

        # push stack
        curr_char_offset = self._char_offset
        self._char_offset_stack.append( curr_char_offset )

        # parse element
        if node.nodeName == '#text':
            # base case
            self._handle_data(node)
            # pop stack
            self._char_offset_stack.pop()
            if _DEBUG: print >> sys.stderr,"%s OUT NODE %s" %(s,node.nodeName)
            return
        else:
            # recursive case
            for c in node.childNodes:
                self._parse_element( c, 2*s )
        
        # handle element
        if node.tagName in self._annotations.keys():
            # get char offsets for element
            char_offsets = self._char_offset_stack[-1], self._char_offset
            self._handle_element(node, char_offsets)

        if _DEBUG: print >> sys.stderr,"%s OUT NODE %s, %s" \
           %(s,node.nodeName,(self._char_offset_stack[-1],self._char_offset))

        # pop stack
        self._char_offset_stack.pop()
        return


    def _handle_data(self, node):
        self._rawdata += node.data.replace('\n',' ').encode('Latin-1')
        self._char_offset = self._char_offset+len( node.data )
        return

        
    def _handle_element(self, node, (s_offset, e_offset)):
        # print node.tagName, node.attributes.items()
        # read in node info
        elt_type = node.tagName
        # convert to dict and normalize string values: e.g., mod=WOULD mod=would
        elt_dict = dict( [(k,v.upper()) for (k,v) in node.attributes.items()] )
        elt_dict['char_offsets'] = (s_offset, e_offset-1)
        elt_dict['text'] = self._rawdata[s_offset:e_offset]
        # call element specific handler
        handler_method = getattr(self, "_handle_%s" % elt_type)
        obj = handler_method( elt_dict )
        # set index attribute corresp. to index in sequence
        index = len(self._annotations[node.tagName])
        obj.index = index
        _id = obj.__dict__.get('id')
        if _id is None:
            # index serves as default id when missing
            _id = str(index)
        _id = str(_id).decode('Latin-1').encode('Latin-1')
        obj.id = _id 
        # register object position for sync
        for pos in range(s_offset, e_offset+1):
            self._offset_map[elt_type][pos] = obj
        # store in annotations dict
        self._annotations[elt_type][_id] = obj
        #print ">>>", elt_type, obj.__dict__
        return

    def _handle_TEXT(self, _dict):
        return Text(_dict)
        
            
    def _handle_s(self, _dict):
        sent = Sentence(_dict)
        return sent

    def _handle_EVENT(self, _dict):
        evt = Event(_dict)
        evt.id = evt.eid
        return evt

    def _handle_TIMEX3(self, _dict):
        timex = Timex3(_dict)
        timex.id = timex.tid
        return timex

    def _handle_ENAMEX(self, _dict):
        ne = Enamex(_dict)
        return ne

    def _handle_NUMEX(self, _dict):
        num = Numex(_dict)
        return num

    def _handle_SIGNAL(self, _dict):
        signal = Signal(_dict)
        return signal

    def _handle_MAKEINSTANCE(self, _dict):
        events = self._annotations["EVENT"]
        evt_inst = EventInstance(_dict, events)
        evt_inst.id = evt_inst.eiid
        return evt_inst
 
    def _handle_TLINK(self, _dict):
        tlink = TemporalRelation(_dict)
        return tlink

    def _handle_SLINK(self, _dict):
        raise NotImplementedError
        
    
    def _handle_ALINK(self, _dict):
        raise NotImplementedError


    def _synchronize(self):
        for _type in self._annotations:
            for _,annot in self._annotations[_type].items():
                pos = annot.char_offsets[0]
                # sentence pointer
                annot.sentence = self._offset_map["s"].get(pos)
        return

    def _printout(self):
        for _type in self._annotations:
            print >> sys.stdout, _type
            for _id,annot in self._annotations[_type].items():
                print _id, annot, annot.__dict__
            print >> sys.stdout,'\n'

    def _stats(self):
        for _type in self._annotations:
            print _type, len(self._annotations[_type])
        return

            
    # public methods
    def get_rawstr(self):
        return self._rawdata

    def get_events(self):
        return self._annotations["EVENT"]

    def get_timex(self):
        return self._annotations["TIMEX3"]

    def get_tlinks(self):
        return self._annotations["TLINK"]

    def get_event_instances(self):
        return self._annotations["MAKEINSTANCE"]

    def get_enamex(self):
        return self._annotations["ENAMEX"]

    def get_numex(self):
        return self._annotations["NUMEX"]

    def get_sentences(self):
        return self._annotations["s"]

    def get_signals(self):
        return self._annotations["SIGNAL"]

    
    

        

####################################################################################

    
_DEBUG=False

if __name__ == '__main__':

    import sys
    
    reader = TimeML11Reader( sys.argv[1] )
    print sys.argv[1]
    reader._printout()
    reader._stats()
    print reader._rawdata
    


