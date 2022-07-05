#!/usr/bin/env python

"""
exemple: lancer base  sur modeles globaux tempeval, avec textes par ordre croissant

python ../src/launch_tests.py -d base -r tempeval -g -x -e "-s -k -t" folds/otc_folds/


"""


import os
import apetite
import optparse

usage = "usage: %prog [options] folds_dir"
parser = optparse.OptionParser(usage=usage)
parser.add_option("-m", "--models", default='models', help="path to models directory (default: ./models)")
parser.add_option("-o", "--outputs", default='outputs', help="path to models directory (default: ./outputs)")
parser.add_option("-l", "--logs", default='logs', help="path to logs directory (default: ./logs)")
parser.add_option("-s", "--scores", default='scores', help="path to logs directory (default: ./scores)")
parser.add_option("-r", "--relset", default='allen,bruce,tempeval,jurafsky', help="relation set (default: allen,bruce,tempeval,jurafsky)")
parser.add_option("-e", "--extra", default="-s -k -t", help="extra options (def= -k -t); must use also '-x number' to pass on a threshold to the linker")
parser.add_option("-d", "--method", default='base', help="method used to link events (must be accepted by n_fold !) default=baseline Mani with no threshold")
parser.add_option("-g", "--globalmodel", default=False,action="store_true",help="use global models")
parser.add_option("-x", "--execute", default=False,action="store_true",help="by default commands are printed; use this to actually execute the commands")

parser.add_option("-E", "--resampled", default=False, action="store_true",help="use the resampled models")
parser.add_option("-a", "--annotation", default=False, action="store_true",help="classify only gold event-event pairs")

parser.add_option('-i','--suffix',default=None,help="force suffix for models (appended to relset_satX_bethX_disjX_)")
parser.add_option('-p','--output-suffix',default=None,help="force suffix for outputs (appended to method_relset_satX_bethX_disjX_...)")


(options, args) = parser.parse_args()



# paths
FOLD_DIR  = args[0]
MODEL_DIR = options.models
SCORES    = options.scores
OUTPUTS   = options.outputs
LOG_DIR   = options.logs
TESTER   = os.path.join(apetite.__path__[0],"n_fold_experiment.py")


# parameters
relsets = options.relset.split(',')
sat = [0,1]
beth = [0,1]
disj = [0,1]



for r in relsets:
    for s in sat:
        for b in beth:
            for d in disj:
                if d and not s:
                    continue
                # file name
                filename = "%s_%s_sat%s_beth%s_disj%s" %(options.method,r,s,b,d)
                if options.suffix:
                    model="%s_sat%s_beth%s_disj%s_%s" %(r,s,b,d,options.suffix)
                else:
                    model="%s_sat%s_beth%s_disj%s_models" %(r,s,b,d)
                if options.globalmodel:
                    if not(options.suffix): model += "_global"
                    filename += "_global"
                else:
                    if not(options.suffix): model += "_local"
                    filename +="_local"
                if options.resampled:
                    if not(options.suffix): model += "_resampled"
                    filename += "_resampled"
                if options.output_suffix:
                    filename += "_"+options.output_suffix
                # command 
                cmd = "python %s -r %s -l %s -o  %s/%s" %(TESTER,r,options.method,OUTPUTS,filename)
                cmd += " -m %s/%s "  %(MODEL_DIR,model)
                cmd += options.extra
                if options.annotation: cmd += " -a "
                cmd += " %s > %s/%s 2> %s/%s" %(FOLD_DIR,SCORES,filename,LOG_DIR,filename)
                print cmd
                if options.execute:
                    os.system( cmd )
                # reset
                filename = ""
                cmd = ""
    
