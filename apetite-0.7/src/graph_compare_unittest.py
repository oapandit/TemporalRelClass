#!/bin/env python
# -*- coding: iso-8859-1 -*-


# usage:
#    python graph_compare_unittest.py -v

# files with error conversion pt->evt->pt
error_conv_files="""../data/TimeBank1.1/docs/ABC19980304.1830.1636.tml.xml -> 
erreur fichier : <TLINK relatedToEventInstance="ei286" eventInstanceID="ei286" relType="INCLUDES" />
TODO: guard dans make_graph + warning
idem
../data/TimeBank1.1/docs/wsj_0918_orig.tml.xml [(u'ei2049', u'ei2049'), (u'ei2050', u'ei2050')])
!!!! ei2049, ei2049 : [trel_beforei]

../data/TimeBank1.1/docs/wsj_0679_orig.tml.xml set([(u'ei309', u'ei309')])

../data/TimeBank1.1/docs/wsj_0325_orig.tml.xmlset([(u'ei199', u'ei199')])


-------------------
../data/TimeBank1.1/docs/APW19980219.0476.tml.xml
-> nodename=None à cause <TLINK signalID="s2202" eventInstanceID="ei2507" relType="BEFORE" />
TODO: guard dans make_graph + warning
idem dans:
../data/TimeBank1.1/docs/APW19980227.0489.tml.xml tlink [(u'ei2055', None)]
../data/TimeBank1.1/docs/VOA19980303.1600.0917.tml.xml [(u'ei2231', None)]
../data/TimeBank1.1/docs/WSJ900813-0157.tml.xml[(u't397', None)]
../data/TimeBank1.1/docs/WSJ910225-0066.tml.xml[(u'ei1252', None)]
../data/TimeBank1.1/docs/wsj_0068_orig.tml.xml [(u'ei144', None)]
../data/TimeBank1.1/docs/wsj_0160_orig.tml.xml [(None, u't38')]
../data/TimeBank1.1/docs/wsj_0168_orig.tml.xml[(u'ei138', None)]
../data/TimeBank1.1/docs/wsj_0695_orig.tml.xml [(u't79', None), (u't37', None)]


inconsistent: 
../data/aquaint_timeml_1.0/data/APW19980811.0474.tml.xml
../data/aquaint_timeml_1.0/data/APW19980818.0515.tml.xml
../data/aquaint_timeml_1.0/data/APW19980820.1428.tml.xml
../data/aquaint_timeml_1.0/data/APW19990506.0155.tml.xml
../data/aquaint_timeml_1.0/data/APW19991008.0151.tml.xml
../data/aquaint_timeml_1.0/data/APW199980817.1193.tml.xml
../data/aquaint_timeml_1.0/data/APW20000115.0031.tml.xml
../data/aquaint_timeml_1.0/data/APW20000107.0318.tml.xml

APW19980213.1380.tml.xml
AP900816-0139.tml.xml

timebank inconsistent (avec bethard)
APW19980227.0468.tml.xml 
CNN19980227.2130.0067.tml.xml     
NYT19980206.0460.tml.xml      
NYT19980402.0453.tml.xml
PRI19980303.2000.2550.tml.xml
wsj_0032_orig.tml.xml   
wsj_0068_orig.tml.xml     
wsj_0132_orig.tml.xml    
wsj_0144_orig.tml.xml    
wsj_0160_orig.tml.xml    
wsj_0169_orig.tml.xml
wsj_0325_orig.tml.xml  
wsj_0430_orig.tml.xml   
wsj_0505_orig.tml.xml     
wsj_0520_orig.tml.xml          
wsj_0542_orig.tml.xml    
wsj_0568_orig.tml.xml     
wsj_0586_orig.tml.xml   
wsj_0660_orig.tml.xml      
wsj_0675_orig.tml.xml
wsj_0760_orig.tml.xml
wsj_0762_orig.tml.xml
wsj_0778_orig.tml.xml
wsj_0786_orig.tml.xml
wsj_0918_orig.tml.xml 
wsj_0924_orig.tml.xml      
wsj_0927_orig.tml.xml    
wsj_1013_orig.tml.xml
wsj_1033_orig.tml.xml

"""

import unittest
from pprint import pprint
import sys, time
from copy import deepcopy
import glob

from apetite.graph_compare import *
from TimeMLDocument import Document


def testOne(filename,sature=False):
    doc = Document(filename)
    g=doc.graph
    gpt=interval2point(g)
    gbis=point2interval(gpt)  
    if sature:
	g.saturate()
	gbis.saturate()
    print [(x,y) for (x,y) in doc.relations.keys() if x is None or y is None]   
    return g,gbis,gpt


#wsj_0161
def testZero(doc):
    """test of point recall with relations removed"""
    g1=doc.graph
    g1bis=clone_graphe(g1,doc)
    res=variantes(g1,g1bis,n=len(g1bis.edges()))
    return res,g1bis


def testMingraph(g1,g2):
    """gpt1,gmin1,gpt2,gmin2=testMingraph(g1,g2)
    """
    gpt1,gmin1=transform2pt(g1)
    gpt2,gmin2=transform2pt(g2)
    v1=graph_value(gpt1)
    print v1,
    return gpt1,gmin1,gpt2,gmin2

class Conversion(unittest.TestCase):
    GRAPH_NB=0
    # graph tests dans test1.graph, test2.graph, etc
    all_test_graphs=[]
    for i in range(1,GRAPH_NB+1):
        g=read_graphe("tests/test%d.graph"%i,allen_algebra)
        gpt=interval2point(g)
        all_test_graphs.append((g,gpt,"test%d.graph"%i))
    
    
    #for file1 in glob.glob("../data/aquaint_timeml_1.0/data/AP*.tml.xml"):
    #for file1 in ["../data/aquaint_timeml_1.0/data/APW20000115.0209.tml.xml"]:
    #for file1 in glob.glob("../data/TimeBank1.1/docs/*.tml.xml"):
    for file1 in glob.glob("../data/aquaint_timeml_1.0/data/*.tml.xml"):
	doc = Document(file1)
	g=doc.graph
	gpt=interval2point(g)
	all_test_graphs.append((g,gpt,file1))	
    
    
    def similar(self,g1,g2):
	"""graphs have same node labels and same relations between them"""
	self.assertEqual(set(g1.nodes().keys()),set(g2.nodes().keys()))
	#self.assertEqual(set(g1.edges().keys()),set(g2.edges().keys()))
	#except: print >> sys.stderr, set(g1.edges().keys())-set(g2.edges().keys())
        self.assertEqual(g1.edges(),g2.edges())
 
    def testConversion(self):
        """ test equality of conversion int->pt->int"""
        for (g,gpt,name) in self.all_test_graphs:
	    gbis=point2interval(gpt)
	    try:
	      self.similar(g,gbis)
	    except:
	      print >> sys.stderr, "error on file", name
	    
    def testConversionSaturation(self):
        """ test equality of conversion with saturation"""
        for (g,gpt,name) in self.all_test_graphs:
	    print >> sys.stderr, "\n doing file: ", name, time.ctime(),

            g1=deepcopy(g)
            cons_evt=g1.saturate()
	    print >> sys.stderr, "\n evt saturation done ", time.ctime(),
            gpt1=deepcopy(gpt)
            cons_pt=gpt1.saturate()
	    try:
	      self.assertEqual(cons_pt,True)
	    except:
	      print >> sys.stderr, "\n inconsistent point graph ",name,
	    try:
	      self.assertEqual(cons_evt,True)
	    except:
	      print >> sys.stderr, "\n inconsistent event graph ",name,
	      continue
            try:
		self.similar(g1,point2interval(gpt1))
	    except:
		print >> sys.stderr, "error on file", name
            #pprint(g1)
            #pprint(gpt1)
            




if __name__ == "__main__":
    unittest.main()
