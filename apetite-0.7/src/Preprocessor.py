#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

'''

TODO: DEPRECATED SINCE LAST CHANGE OF READER AND DOCUMENT API!!!

preprocessing of TimeML sentences:
 - PTB-style tokenization for MXPOST
 - POS-tagging with MXPOST
 - character offset synchronization with original text


TODO: preprocess header!!!

WARNING: missing sentence splitting in some documents!!!! use MXSPLIT

'''

import sys
import os
import tempfile
import re
from timeml_reader import TimeML11Reader
import xml.etree.cElementTree as ET



MXTOOLS_DIR = os.environ.get("MXTOOLS_DIR",None)
if not MXTOOLS_DIR:
    sys.exit("Please set MXTOOLS_DIR to directory containing MXPOST tagger.")





class Preprocessor:

    def __init__(self):
        self.tokenizer_exec = os.path.join(MXTOOLS_DIR,'ptb_tokenizer.sh')
        self.mxpost_exec = os.path.join(MXTOOLS_DIR,'mxpost')
        self.mxpost_model = os.path.join(MXTOOLS_DIR,'tagger.project')
        self.mxpost_tok_re = re.compile(r'(.+)\_([A-Z,;\.\!\?\-:#$\`\'\"]+)')
        return

    def preprocess_dir(self, _dir):
        timeml_files = [os.path.join(_dir,f) for f in os.listdir(_dir)
                        if f.endswith('.tml.xml')
                        if not f.startswith('.')]
        for (i,f) in enumerate(timeml_files):
            print >> sys.stderr, 'Preprocessing file %s: %s' %(i,f)
            self.preprocess_file( f )
        return


    def preprocess_file(self, xml_file):
        # extract sentences
        sent_file, sent_dict = self.extract_sentences( xml_file )
        # tokenize sentences
        tok_file = self.tokenize_file( sent_file )
        # pos-tag sentences
        pos_file = self.pos_tag_file( tok_file )
        # extract tokens along with their POS
        tokens = self.extract_tokens( pos_file )
        # synchronize offsets with original sentences
        synced_tokens = self.synchronize_tokens( tokens, sent_dict )
        # write output XML file
        outfile_name = xml_file[:-4]+'.prep.xml'
        self.write_xml(outfile_name, synced_tokens, xml_file)
        # cleanup tempfiles
        os.unlink(sent_file)
        os.unlink(tok_file)
        os.unlink(pos_file)
        return


    def extract_sentences(self, xml_file):
        # read in XML
        reader = TimeML11Reader( xml_file )
        sent_dict = reader.annotations['s']
        # sort sentences by IDs (i.e., extent, see cf. timeml_reader.py)
        sent_items = sorted(sent_dict.items(),cmp=lambda x,y:cmp(int(x[0]),int(y[0])))
        # write sentence strings to temp file
        txt_file = tempfile.mktemp()
        tfh = open(txt_file,'w')
        for (sid,sent) in sent_items:
            print >> tfh, sent['text']
        tfh.close()
        return txt_file, sent_dict
    

    def tokenize_file(self, txt_file):
        ''' tokenize file in PTB format '''
        out_file = tempfile.mktemp()
        cmd = "%s %s > %s" %(self.tokenizer_exec, txt_file, out_file)
        os.system( cmd )
        return out_file


    def pos_tag_file(self, tok_file):
        ''' POS tag file '''
        out_file = tempfile.mktemp()
        cmd = "%s %s < %s > %s" %(self.mxpost_exec, self.mxpost_model, tok_file, out_file)
        os.system( cmd )
        return out_file


    def extract_tokens(self, pos_file):
        ''' extract (word, pos) pairs from POS-tagged file '''
        tokens = []
        for l in file(pos_file):
            l = l.strip()
            mxpost_items = l.split()
            sent_tokens = [] # sublist for each sentence
            for mit in mxpost_items:
                match = self.mxpost_tok_re.match( mit )
                if match:
                    word,pos = match.groups()
                else:
                    print "Error: token '%s' cannot be split!" %mit
                    sys.exit()
                    return
                sent_tokens.append((word,pos))
            tokens.append(sent_tokens)
        return tokens


    def synchronize_tokens(self, tokens, sent_dict):
        ''' compute character offsets for tokens in original sentence '''
        synced_tokens = []
        for (sct,s_tokens) in enumerate(tokens):
            s_synced_tokens = []
            original_sent = sent_dict[str(sct)] # cf. sentences are indexed based on count
            original_sent_start = original_sent['char_offsets'][0]
            original_sent_str = original_sent['text']
            # find tokens by scanning original sentence string
            for (wd,pos) in s_tokens:
                try:
                    idx = original_sent_str.index( wd )
                except ValueError:
                    # replace MXPOST bracket symbols
                    wd = self.replace_bracket_symbols(wd)
                    # normalize double quotes
                    wd = self.normalize_quotes(wd)
                    idx = original_sent_str.index( wd )
                start = original_sent_start+idx
                extent = (start,start+len(wd)-1)
                token = (wd,pos,extent)
                s_synced_tokens.append( token )
                # reset sentence str and sentence start
                original_sent_str = original_sent_str[idx+len(wd):]
                original_sent_start = original_sent_start+idx+len(wd)
            synced_tokens.append( s_synced_tokens ) 
        return synced_tokens


    def replace_bracket_symbols(self,_str):
        _str = _str.replace('-LRB-','(')
        _str = _str.replace('-RRB-',')')
        _str = _str.replace('-LSB-','[')
        _str = _str.replace('-RSB-',']')
        _str = _str.replace('-LCB-','{')
        _str = _str.replace('-RCB-','}')
        return _str

    def normalize_quotes(self,_str):
        _str = _str.replace('``','"')
        _str = _str.replace("''",'"')
        return _str
    

    def write_xml(self, out_file, tokens, source_file):
        doc = ET.Element("document")
        doc.set('source',os.path.basename(source_file))
        for i in xrange(len(tokens)):
            sElement = ET.SubElement(doc,"s")
            sElement.set('id',str(i))
            s_tokens = tokens[i]
            for (j,(wd,pos,extent)) in enumerate(s_tokens):
                tokElement = ET.SubElement(sElement,"token")
                tokElement.set('id','%s-%s' %(str(i),str(j)))
                tokElement.set('s_id',str(i))
                tokElement.set('word',wd)
                tokElement.set('pos',pos)
                tokElement.set('start_choffset',str(extent[0]))
                tokElement.set('end_choffset',str(extent[1]))
        tree = ET.ElementTree(doc)
        indent(tree.getroot()) # hack to get pretty-print
        # tree.write(sys.stderr, "utf-8")
        tree.write(out_file, "utf-8")
        return




def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
            if not e.tail or not e.tail.strip():
                e.tail = i + "  "
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i







if __name__ == '__main__':

    
    _dir = sys.argv[1]
    p = Preprocessor()
    p.preprocess_dir( _dir )
