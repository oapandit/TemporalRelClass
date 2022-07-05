#!/bin/env python
# -*- coding: iso-8859-1 -*-


# usage:
#    python graph_compare_unittest.py -v


import unittest
from pprint import pprint
import sys
from copy import deepcopy

import TimeMLDocument

class Read(unittest.TestCase):
    
    def testRead(self):
        """Test en lecture, verif nb d'evt, timex, tlink"""
        self.assertEqual(len(self.doc.events),28)
        self.assertEqual(len(self.doc.timex),5)
        self.assertEqual(len(self.doc.timeml_annotations['TLINK']),24)


    def testGraph(self):
        """Test graphe de relations, saturation"""
        self.assertEqual(len(self.doc.get_relations()),24)
        self.assertEqual(self.doc.saturate_graph(),True)
        self.assertEqual(len(self.doc.get_relations()),166)

    def testConversion(self):
	"""conversion events->points->events"""
	pass


if __name__ == "__main__":
    unittest.main()


