#!/usr/bin/env python

"""
exemple: lancer Evaluation  sur output 

python ../src/launch_eval.py -d base -m "simple_prec,simple_recall,tempeval_prec,tempeval_recall" -i allen -r tempeval ../data/OTC outputs

python  ../src/Evaluation.py  -s tempeval -i allen -m "simple_prec,simple_recall,tempeval_prec,tempeval_recall" -v ../data/OTC outputs/nro_allen_sat0_beth0_disj0/ > scores/nro_allen_sat0_beth0_disj0 2> logs/evalnro_allen_sat0_beth0_disj0
"""

import os
import apetite



from optparse import OptionParser

usage="%prog [options] reference_directory system_output_directory"

parser = OptionParser(usage=usage)
parser.add_option("-s", "--relation_set",type="choice",choices=["allen","bruce","jurafsky","tempeval"],
                  default="allen",help="relation set in which the evaluation is done (default=allen=full timeml)")
parser.add_option("-i", "--inputset",default=None,
                  help="relation set in which the predictions are expressed (default when None: evaluation relation set)")
parser.add_option("-m", "--measure",default="simple_prec,simple_recall,tempeval_prec,tempeval_recall",
                  help="measure type (in ...)")
parser.add_option("-d", "--method", default='base', help="method used to link events")
parser.add_option("-g", "--globalmodel", default=False,action="store_true",help="use global models")
parser.add_option("-E", "--resampled", default=False, action="store_true",help="use the resampled models")
parser.add_option('-p','--suffix',default=None,help="force suffix for inputs/outputs (appended to method_relset_satX_bethX_disjX_...)")
parser.add_option('-b','--baseclassifier',default=False,action="store_true",help="evaluate classifier only on gold relations, wo saturation")



(options, args) = parser.parse_args()
        

# paths
REF_DIR  = args[0]
SCORES    = "./scores"
OUTPUTS   = "./outputs"
LOG_DIR   = "./logs"
EVALUER   = os.path.join(apetite.__path__[0],"Evaluation.py")


# parameters

sat = [0,1]
beth = [0,1]
disj = [0,1]


rel=options.relation_set
inputset=options.inputset

for s in sat:
    for b in beth:
        for d in disj:
            if d and not s:
                continue
             # command 
            cmd = "python %s -s %s -v  -m %s" %(EVALUER,rel,options.measure)
            if inputset is not None:
                cmd += " -i %s"%inputset
                filename_out = "%s_%s_in_%s_sat%s_beth%s_disj%s" %(options.method,inputset,rel,s,b,d)
                filename = "%s_%s_sat%s_beth%s_disj%s" %(options.method,inputset,s,b,d)
            else:
                filename_out = "%s_%s_sat%s_beth%s_disj%s" %(options.method,rel,s,b,d)
                filename = "%s_%s_sat%s_beth%s_disj%s" %(options.method,rel,s,b,d)
            
            if options.globalmodel:
                filename += "_global"
                filename_out += "_global"
            else:
                filename += "_local"
                filename_out +="_local"
            if options.resampled:
                filename += "_resampled"
                filename_out +="_resampled"
                
            if options.suffix:
                filename += "_"+options.suffix
                filename_out += "_"+options.suffix

            if options.baseclassifier:
                cmd += " -b "
                filename_out += "_raw"
            cmd += " %s %s/%s > %s/%s 2> %s/eval%s" %(REF_DIR,OUTPUTS,filename,SCORES,filename_out,LOG_DIR,filename_out)
            print cmd
            #if options.execute:
            #    os.system( cmd )
            # reset
            filename = ""
            cmd = ""

        
