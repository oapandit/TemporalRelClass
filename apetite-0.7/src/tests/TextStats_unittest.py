#!/bin/env python
# -*- coding: iso-8859-1 -*-


import unittest
from pprint import pprint
import sys

from apetite.TextStats import *

class TextStatTest(unittest.TestCase):

    file2="tests/ea980120.1830.0071.tml.xml.autocomp2.csv"
    file1="tests/ea980120.1830.0071.tml.xml.autocomp1.csv"
    
    def test(self):
        """dummy test"""
        self.assertEquals(self.report.name(),"EACL09")
    
    def setUp(self):
        filelist=["tests/ea980120.1830.0071.tml.xml",
                  "tests/ea980120.1830.0071.tml.xml"]
        expe={"remove":"autocomp1.csv",
              "disturb":"autocomp2.csv"
              }
        r=Report("EACL09")
        r.addFiles(filelist)
        for one in expe:
            r.addExperiment(one,expe[one])
        self.report=r
        
    def tearDown(self):
        print self.report

if __name__ == "__main__":
    unittest.main()   
