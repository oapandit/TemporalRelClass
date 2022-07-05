#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
#
# synthesis of autocomparison results
# takes a set of  results organised by column
# and produce something usable from spreadsheet
# by
# ok 1) filtering results by type
# ok 2) normalising the results with respect to size of each referent
#    (depends on the number of events ... to be done later)
# ok 3) summing evrything
# 4) weighting texts
# 5) filtering texts (eg on number of events)
# todo: script options



# - mettre des poids sur les textes
# - garder que les valeurs sur 10%, 20% , ...
# - rebin valeurs approch�es ex 0.19/0.20
# x- gaussian smoothing
# TODO: refactorisation
# Classes:
#    - data points
#    - measure(/text)
#    - TextStat = set of measures 


from apetite.graph_compare import table_row_labels, mesures_names
import random
import numpy
### This is the Gaussian data smoothing function I wrote ###  
def smooth_list_gaussian(list,degree=5):
    window=degree*2-1  
    weight=numpy.array([1.0]*window)  
    weightGauss=[]  
    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(numpy.exp((4*(frac))**2))  
        weightGauss.append(gauss)  
    weight=numpy.array(weightGauss)*weight  
    smoothed=[0.0]*(len(list)-window)  
    for i in range(len(smoothed)):  
        smoothed[i]=sum(numpy.array(list[i:i+window])*weight)/sum(weight)  
    return smoothed  



class TextStat:
    """ wrapper for stats collected on a given file"""
    def __init__(self,name):
        """name = filename"""
        self.name=name
        self.data={}
        self._size={}

        

    def measures(self):
        return self.data.keys()

    def size(self,type=None):
        if type==None:
            return self._size
        else:
            return self._size[type]

    
    def addMeasure(self,measure_name):
        self.data[measure_name]=[]

    def addValue(self,measure_name,value):
        self.data[measure_name].append(value)

        

class Report:
    """stats on a corpus """

    
        

if __name__=="__main__":

    import os, glob, sys
    #import pylab
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--mode",
                      dest="mode", default="remove",
                      help=" expe perte info (mode=remove) ou perturbation (mode=disturb) ")
    parser.add_option("-s","--sample",dest="sample",default=1000,
                      help="")
    parser.add_option("-e","--eventnb",dest="eventnb",default=None,
                      help="filter on number of events, with interval (min,max): not implmtd")
    parser.add_option("-w","--weight",dest="weight",action="store_true",default=False,
                      help="weight texts according to nb of relations")
    # type= precision ou recall
    parser.add_option("-t","--type",dest="type",default="recall",
                      help="type of measures: precision or recall measures")
    # to best see evolution of function (continuity especially)
    parser.add_option("-z","--remove-zero",dest="zero",action="store_true",
                      default=False,help="remove point 0,0")
    # remove (1,1)
    parser.add_option("-o","--remove-one",dest="one",action="store_true",
                      default=False,help="remove point 1,1")
    parser.add_option("-d","--display",dest="display",default="line",help="display mode : points, line, linear (pts+ linear fit), parabolic (pts+ parabolic fit)")
    parser.add_option("-l","--label",dest="label",default="",help="title for the figure (default=none)")
    parser.add_option("-b","--bin-number",dest="bin_nb",default=1,help="granluarity 1: all points, 10 = binning at 0.1 ") 
    parser.add_option("-c","--cut",dest="cut",type="float",default=0.,help="consider only points of abscisse>=thershold ") 
    parser.add_option("-a","--smooth",dest="smoothed",type="int",
                      default=None,help="gaussian smoothing of data, with given degree")

    # todo: select precision step
    # todo: select threshlod of nb of relations to focus on different texts sizes

    
    (options, args) = parser.parse_args()

    

    
    display=options.display
    bin_nb=float(options.bin_nb)
    # how many texts ?
    SAMPLE=200
    SAMPLE=int(options.sample)

    # which experiment ?
    if options.mode=="remove":
        MODE="left"
    else:
        MODE="undisturbed"

    TYPE=options.type
    
    WEIGHT=options.weight
    #WEIGHT=False
    
    if MODE=="left":# only recall matters
        #keep=['RECA','FINAllen','RECNOY','REC_PTGRAPH']
        keep=['RECA','FINAllen','REC_PTGRAPH']
        #label=dict(zip(keep,["simple recall","finesse",
        #                 "kernel recall","recall on endpoints"]))
        label=dict(zip(keep,["simple recall","gen. recall",
                             "recall on endpoints"]))
        suffix=".autocomp1.csv"
    else:
        if TYPE.startswith("p"):# precision measures
            keep=['PREC','COHAllen','PREC_PTGRAPH']
            lbl=["simple precision","gen. prec.",
                 "precision on endpoints"]
        if TYPE.startswith("rec"):# recall measure
            keep=['RECA','REC_PTGRAPH']
            lbl=["simple recall","recall on endpoints"]
        label=dict(zip(keep,lbl))
        suffix=".autocomp2.csv"
        
    row_idx=[(x,list(table_row_labels).index(x)) for x in keep]

    #print >> sys.stderr, keep
    #print >> sys.stderr, label

    texts={}

    
    if len(args)==0:# no argument, so take all timeml files in current dir
        list_filenames=glob.glob("*"+suffix)
        # during computation, get rid of potentially unfinished last treated file
        #list_filenames=random.sample(list_filenames[:-1],min(len(list_filenames)-1,SAMPLE))
    
    # or, argument= do it on a file 
    else:
        list_filenames=[args[0]]

    # poids sur textes: nb d'evt de la ref, nb d'evt du noyau, nb evt noyau modifi�

    nb_rel_tot={}
    weight={}
    # total is indexed by percent of kept information, and contain the total score
    # and the number of points contributing to the score
    total=dict(zip(keep,[{} for x in keep]))
    nb_files=0
    # bug: only the last file is kept (ok)
    for filename in list_filenames:
        print >> sys.stderr, "doing file", filename
        # first line is header, just keep numbers
        # todo: header choose be used, to avoid mistake and changes
        data=[x.strip().split() for x in open(filename).readlines()][1:]
        if data==[]:
            print >> sys.stderr, "file empy ??", filename
        else:
            nb_files +=1
            # ref: nb evt total pour chaque mesure
            for (name,idx) in row_idx:
                nb_rel_tot[name]=float(data[0][idx])
                if WEIGHT:
                    weight[name]=nb_rel_tot[name]
                else:
                    weight[name]=1
            # save evrything just in case (not used yet)
            texts[filename]=TextStat(filename)
            data_pt_nb=float(len(data)+1)
            for (name,idx) in row_idx:
                texts[filename].addMeasure(name)
                # total is indexed by percent of kept information, and contain the total score
                # and the number of points contributing to the score
                total[name][0.]=0.,nb_files,0.
                for i,line in enumerate(data):
                    try:
                        val=float(line[idx+2])
                    except:
                        print >> sys.stderr, "error reading value :", name,line,idx+2,filename
                        val=0
                    if val<0 or val>1:
                        print >> sys.stderr, "negative or excessive value for file (ignored):",val,name,filename
                    else:
                        texts[filename].addValue(name,float(val))
                        score,count,squares=total[name].get(1-i/data_pt_nb,(0.,0,0.))
                        new=float(val)*weight[name]
                        total[name][(1-i/data_pt_nb)]=(score+new,count+weight[name],squares+new*new)
                texts[filename].addValue(name,0.0)
    print >> sys.stderr, "supposed nb of files :", min(len(list_filenames),SAMPLE)
    #visualization
    if True:
        from math import sqrt
        average={}
        # now average each point, print out result table
        for name,idx in row_idx:
            #average[name]=[(pt,(x/y),sqrt(z/y-(x/y)**2)) for (pt,(x,y,z)) in total[name].items()]
            average[name]=[(pt,(x/y),sqrt(abs(z/y-(x/y)**2))) for (pt,(x,y,z)) in total[name].items() if (abs((pt*10)-round(pt*10))<1./bin_nb)]
            average[name].sort()
        # graphs !!
        import pylab
        symbols=["g>","kx","mp","bo","r+","vk"]
        symbols_line=["-","--","-.",".","_",","]
        pylab.subplot(2,1,1)
        for i,(name,idx) in enumerate(row_idx):
            # todo: force xrange[0,1]
            if options.cut>0:
                average[name]=[(x,y,z) for (x,y,z) in average[name] if x>=options.cut]
            abscisses,ordonnees,y_err=zip(*average[name])
            
            #if name=="REC_PTGRAPH":
            #    break#bugged measure
            #n, bins, patches = pylab.hist(abscisses, 100,label=label[name])
            if options.zero:
                # remove point (0,0)
                abscisses=abscisses[1:]
                ordonnees=ordonnees[1:]
            if options.one:
                # remove point (1,1)
                abscisses=abscisses[:-1]
                ordonnees=ordonnees[:-1]

            if options.smoothed:
                pylab.subplot(2,1,2)
                pylab.plot(abscisses,ordonnees,label=label[name],ls="-")
                degree=options.smoothed
                ordonnees=smooth_list_gaussian(ordonnees,degree=degree)
                pylab.subplot(2,1,1)
                abscisses=abscisses[degree-1:-(degree)]
            if display=="line":
                pylab.plot(abscisses,ordonnees,label=label[name],ls="--")
            else:
                pylab.plot(abscisses,ordonnees,symbols[i],label=label[name])
            if display=="linear":
                coeffs=pylab.polyfit(abscisses,ordonnees,1)
                pylab.plot(abscisses,pylab.polyval(coeffs,abscisses),label=label[name])
            elif display=="parabolic":
                coeffs=pylab.polyfit(abscisses,ordonnees,2)
                pylab.plot(abscisses,pylab.polyval(coeffs,abscisses),symbols_line[i],label='_nolegend_')#label=label[name])
            elif display.isdigit():
                coeffs=pylab.polyfit(abscisses,ordonnees,int(display))
                pylab.plot(abscisses,pylab.polyval(coeffs,abscisses),label=label[name])
            else:
                pass
        pylab.plot([0.,1.],[0.,1.],label="y=x")
        if options.smoothed:
            pylab.subplot(2,1,2)
            pylab.text(2,.3,"gaussian smoothed",fontsize=14)  
            pylab.plot([0.,1.],[0.,1.],label="y=x")
            pylab.xlabel("percent relation %s"%MODE)
            pylab.ylabel("score")
            pylab.subplot(2,1,1)
            pylab.text(2,.3,"raw",fontsize=14)  
        pylab.plot([0.,1.],[0.,0.])
        #pylab.errorbar(abscisses,ordonnees,yerr=None,label=label[name])
        pylab.xlabel("percent relation %s"%MODE)
        pylab.ylabel("score")
        pylab.title(options.label)
        pylab.legend(loc="best")
        pylab.show()
        
