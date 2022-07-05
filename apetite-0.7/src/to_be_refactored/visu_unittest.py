#!/bin/env python
# -*- coding: iso-8859-1 -*-


import unittest

from visu_autocomp import *
from TextStats import *

class TextStatTest(unittest.TestCase):

    file2="ea980120.1830.0071.tml.xml.autocomp2.csv"
    file1="ea980120.1830.0071.tml.xml.autocomp1.csv"

    t1=TextStat(file1)
    t2=TextStat(file2)

    
    def test(self):
        self.assertEqual(t1.name()==file1)



if __name__ == "__main__":
    unittest.main()   
