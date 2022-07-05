#!/usr/bin/env python


import os
import apetite
import optparse

usage = "usage: %prog [options] folds dir"
parser = optparse.OptionParser(usage=usage)
parser.add_option("-m", "--models", default='models', help="path to models directory (default: ./models)")
parser.add_option("-l", "--logs", default='logs', help="path to logs directory (default: ./logs)")
parser.add_option("-r", "--relset", default='allen,bruce,tempeval,jurafsky', help="relation set (default: allen,bruce,tempeval,jurafsky)")
parser.add_option("-e", "--resample", default=1000, type=int, help="resample the no-class during training by considering only N events back (default: 1000)")
parser.add_option("-g", "--globalmodel", default=False,action="store_true",help="use global models")
#parser.add_option("-a", "--annotation", default=False,action="store_true",help="use only gold relations")

(options, args) = parser.parse_args()

# paths
FOLD_DIR = args[0]
MODEL_DIR = options.models
LOG_DIR = options.logs
INDUCER = os.path.join(apetite.__path__[0],"inducer.py")


# parameters
relsets = options.relset.split(',')
sat = [0,1]
beth = [0,1]
disj = [0,1]


n = options.resample

for r in relsets:
    for s in sat:
        for b in beth:
            for d in disj:
                if d and not s:
                    continue
                # file name
                if options.globalmodel: 
                    filename = "%s_sat%s_beth%s_disj%s_res%s_models" %(r,s,b,d,n)
                else:
                    filename = "%s_sat%s_beth%s_disj%s"%(r,s,b,d)
                if options.globalmodel: 
                    filename +="_global"
                else:
                    filename +="_local"
                    
                # command 
                cmd = "python %s -r %s" %(INDUCER,r)
                if s: cmd += " -s"
                if b: cmd += " -b"
                if d: cmd += " -d"
                if not(options.globalmodel):
                    cmd += " -l" 
                
                if options.globalmodel: 
                    cmd += " -e %s" %n 
                    filename += "_res%s" %n
                cmd += " -p %s/%s %s 2> %s/%s" %(MODEL_DIR,filename,FOLD_DIR,LOG_DIR,filename)
                print cmd
                #os.system( cmd )
                # reset
                filename = ""
                cmd = ""
    
