#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

'''

lecture des phrases tokenisees facon PTB et pos-taggees par MXPOST
(i.e., fichiers *.prep.xml)

'''

import xml.etree.cElementTree as ET


class PreprocessedReader:

    """
    Parse un fichier *.prep.xml 
    """
    def __init__( self, filepath ):
        self.path = filepath
        self.annotations = {
            'token': {}
            }
        self.load()
        self.parse()
        return
    
    def load( self ):
        self.xmltree = ET.ElementTree()
        self.xmltree.parse( self.path )
        return
    
    def parse( self ):
        token_elts = self.xmltree.findall('.//token')
        for tok_elt in token_elts:
            self.handle_token( tok_elt )
        return

    def handle_token( self, element ):
        start = element.get('start_choffset')
        end = element.get('end_choffset')
        extent = int(start),int(end)
        _id = element.get('id')
        s_id = element.get('s_id')
        text = self.unescape(element.get('word'))
        pos = self.unescape(element.get('pos'))
        token_dict = { 'id':_id, 's_id':s_id, 'extent':extent, 'text':text, 'pos':pos }
        self.annotations['token'][_id] = token_dict 
        return 

    def unescape( self, _str ): 
        # you can also use 
        # from xml.sax.saxutils import escape 
        # Caution: you have to escape '&' first! 
        _str = _str.replace(u'&amp;',u'&') 
        _str = _str.replace(u'&lt;',u'<') 
        _str = _str.replace(u'&gt;',u'>')
        _str = _str.encode('utf-8')
        return _str
    




if __name__ == '__main__':

    import sys
    import os

    doc = sys.argv[1]
    reader = PreprocessedReader( doc )
    print reader.annotations['token']
    
    

    

    
    
