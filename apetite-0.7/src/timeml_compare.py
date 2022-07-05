#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


"""
TODO: DEPRECATED SINCE LAST CHANGE OF READER AND DOCUMENT API!!!
"""


from timeml_reader import TimeML11Reader as Document




if __name__=="__main__":

    from optparse import OptionParser
    from os.path import basename

    parser = OptionParser()
    parser.add_option("-t","--timex",dest="timex3",default=True,action="store_true",
                      help="compare timex objects (defaut)")
    parser.add_option("-n","--no-timex",dest="timex3",default=True,action="store_false",
                      help="compare timex objects (defaut)")
    parser.add_option("-e","--event",dest="event",default=False,action="store_true",
                      help="compare event objects")
    parser.add_option("-s","--signal",dest="signal",default=False,action="store_true",
                      help="compare signal objects")
    
    parser.add_option("-l","--header",dest="header",default=False,action="store_true",
                      help="add a header to csv file (defaut=False)")
    

    (options, args) = parser.parse_args()

    standard = args[0]
    target   = args[1]
    
    document1=Document(standard)
    document1.parse_TimeML()
    document2=Document(target)
    document2.parse_TimeML()

    cats=dict([("timex3",options.timex3),("event",options.event),("signal",options.signal)])

    report=document1.compare(document2,cats=[x for x in cats if cats[x]])

    report.display(label=basename(standard).split(".")[0],header=options.header)

    
