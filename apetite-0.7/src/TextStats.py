#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""Module for handling of statistic about corpus of texts
and comparison of different runs of same algorithms
on a given text with some parameter variations

TODO:
   - 3d graphics with extra parameter (size of graph, vagueness degree, etc)
   - measures to watch as script options
   - smooth option to pass on to viz. function
   - type (prec/recall) option  to pass on to viz. function
   - skip saturation as option, to speed up viz. (when coherence not a factor ?)

supposed to be fed to some visualization, eg pylab, cf visu_*.py


mesures possibles:

FINAllen
COHAllen
FINAR
COHAR
PREC
RECA
PRECG
RECAG
JAQ_AR
FINAREE
COHAREE
COMPATAR
PRECEE
RECAEE
RECNOY
PRECNOY
PREC_PTGRAPH
REC_PTGRAPH 


"""
import sys
import pprint
from collections import defaultdict
from apetite.TimeMLDocument import Document
from apetite.graph_compare import read_graphe
from apetite.graph.Graph import allen_algebra
from apetite.utils import pts2matrix

from copy import copy

import numpy

try:
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import colorConverter
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
except:
    print >> sys.stderr, "no visualisation module"




class Ratio:
    """a class storing a count result as a triplet (n,N,n/N)"""
    
    def __init__(self,triplet):
        self._data=[float(x) for x in triplet]
    
    def value(self,check_negative=True):
        if check_negative and (self._data[-1]<0 or self._data[-1]>1.0 ):
            print >> sys.stderr, "negative or excessive (ignored):",self
            return None
        return self._data[-1]

    def __repr__(self):
        return "(%f,%f,%f)"%tuple(self._data)

class Point:
    """a point where a measurement is made
    should have at least two coordinates (x,y)
    y should have a method "value" returning a float
    """
 
    def __init__(self,data):
        self._data=data
        self._abs=float(data[0])
        self._ord=map(float,data[1])
        if len(data)>2:
            self._extra=data[2:]
        
    def abscisse(self):
        return self._abs
    
    def ordonnee(self):
        return self._ord
    
    def __repr__(self):
        return "(%s,%s)"%(`self.abscisse()`,`self.ordonnee()`)
    
    def norm(self,factor):
        self._abs=self._abs/float(factor)
    
class Measure:
    """
    A measurement on a text = a set of data points for a given measure
    """
    def __init__(self,name,data_pts):
        self._name=name
        self._data=[Point(x) for x in data_pts]

    def addPoint(self,x):
        self._data.append(Point(x))

    def name(self):
        return self._name
    
    def data(self,normalise=False):
        if normalise:
            norm=float(max([1]+[x.abscisse() for x in self._data]))    
        else:
            norm=1
        res=[]
        for x in self._data:
            newx=copy(x)
            newx.norm(norm)
            res.append(newx)
        return res
    
    
    def __repr__(self):
        return self.name()+": "+pprint.pformat(self.data())


class Experiment:
    """An experiment is a set of Measures
    on a given text
    """
    def __init__(self,name):
        self._name=name
        self._measures={}
        
    def name(self):
        return self._name
    
    def measures(self):
        return self._measures
    
    def addMeasure(self,name,data_pts):
        self._measures[name]=Measure(name, data_pts)
    
    def get_measure(self,name):
        return self._measures.get(name)
    
    def measure_names(self):
        return self._measures.keys()

    def parse(self,filename,constructor=lambda x:x):
        """reads a table from a file, for one experiment. 
        first line should be a header, with same name for points that belongs
        to the same measure. for each point, first two are then abscisse and ordonnee
        
        obsolete: ref_pointer indicates which column should be considered the reference point (if any)
        on the first line of data
        (default is ordonnee of first value = second column)
        
        constructor builds the data points
        """
        raw=open(filename).readlines()
        datapts=[x.strip().split() for x in raw]
        # each column is put in a list with its header
        datapts=zip(*datapts)
        # then we index by header name to put in a dictionary
        datapts=[(x[0],x[1:]) for x in datapts]
        d=defaultdict(list)
        for (lbl,pts) in datapts:
            d[lbl].append(pts)
        for lbl in d:
            self.addMeasure(lbl,[(i,constructor(x)) for (i,x) in enumerate(zip(*d[lbl]))])
        
            
        

    def __repr__(self):
        return self.name()+": "+pprint.pformat(self.measures())


class TextStat:
    """ wrapper for stats collected on a given file
    = set of Experiments, each being a set of measures (possibly many pts)
    extra is a set of meta-info provided at init time
    """
    def __init__(self,name,extra={},constructor=lambda x:x):
        """name = filename"""
        self._name=name
        self._expe={}
        self._extra=extra
        self._pt_constructor=constructor
        
    def extra(self,label):
        return self._extra[label]
    
    def addExperiment(self,name,filename):
        a=Experiment(name)
        try:
            a.parse(filename,constructor=self._pt_constructor)
            self._expe[name]=a  
        except:
            print >> sys.stderr, "pb with file ", filename
            #a.parse(filename,constructor=self._pt_constructor)
          
    def name(self):
        return self._name
    
    def experiments(self):
        return self._expe.items()
    
    def __repr__(self):
        return self.name()+": "+pprint.pformat(self.experiments())
   
    def extract(self,expe_name,selection,normalise=False):
        expe=self._expe.get(expe_name,None)
        if expe is None:
            return []
        datapts=[(x,expe.get_measure(x).data(normalise=normalise)) for x in expe.measures() if x in selection]
        return datapts
    
    def extract_referent(self,expe,mesure):
        """return maximal value for measure in experiment"""
        mes=self._expe.get(expe)
        if mes is not None:
            mes=mes.get_measure(mesure)
            if mes is not None:
                first_pt=mes.data()[0]
                # assumes data point returns (n,N,n/N))
                referent=first_pt.ordonnee()[1]
                return referent
        return 0
    
    def visualize(self,expe_name,selection,ref,idx=-1,smooth=None):
        """send curve to visualize module
        of a selection (list of measures) in a given experiment.
        since stats are integers count, provide a 'ref' to normalise (usually the relation count for recall,
        for prec, it's usually given by ref=1,idx=-1)
        and since each measure returns a tuple, select with idx which one is the actual result (defaut: last one, when ref=1)
        """
        datapts=self.extract(expe_name,selection)
        
        for meas,data in datapts: 
            abscisses=[x.abscisse() for x in data]
            maxi=float(max(abscisses))
            abscisses=[x/maxi for x in abscisses]
            ordonnees=[float(x.ordonnee()[idx])/ref for x in data]
            if smoothed:
                pylab.subplot(2,1,2)
                pylab.plot(abscisses,ordonnees,label="smoothed",ls="--")
                degree=smoothed
                ordonnees=smooth_list_gaussian(ordonnees,degree=degree)
                pylab.subplot(2,1,1)
                abscisses=abscisses[degree-1:-(degree)]
            
            pylab.plot(abscisses,ordonnees,label=meas)
        pylab.ylabel("score")
        pylab.xlabel("\% modified")
        pylab.title("experiment : %s"%expe_name)

class Report:
    """stats on a corpus = set of texts with stats
    """
    def __init__(self,name,constructor=lambda x:x):
        self._name=name
        self._texts={}
        self._pt_constructor=constructor
        self._expe=[]
        
    def name(self):
        return self._name
     
    def addFiles(self,filelist):
        self._files_names=filelist
        for one in filelist:
            self._texts[one]=TextStat(one,constructor=self._pt_constructor)

    def files(self):
        return self._files_names
    
    def expes(self):
        return self._expe
    
    def texts(self):
        return self._texts
    
    def getText(self,name):
        return self._texts[name]
    
    def addExperiment(self,name,file_suffix):
        """add an experiment whose results are in files of suffix file_suffix
        only files matching those listed during initialization of Report will be considered
        """
        for one in self.files():
            datafile=".".join((one,file_suffix))
            t=self.getText(one)
            t.addExperiment(name,datafile)
            self._expe.append(name)
    
    def __repr__(self):
        return self.name()+"\n"+pprint.pformat(self.texts())
        
        
class TemporalReport(Report):
    """class for temporal information analysis
    
    adds a few meta-information used wrt each text:
    - nb of events
    - total nb of relations in annotation
    - total nb of informative relations in transitive closure
    """
    symbols=["k.","bo","g>","m*","g>","r+","b>","yx","rp","mo","g+"]
    line_symbols=["-","--",":","-.","--","::","-."]

    labels={"FINAllen":"relaxed recall",
            "COHAllen":"relaxed precision",
            "FINAR":"tp rec/bruce",
            "COHAR":"tp prec/bruce",
            "PREC":"strict precision",
            "RECA":"strict recall",
            "PRECG":"??",
            "RECAG":"??",
            "JAQ_AR":"jacquard/bruce",
            "FINAREE":"relaxed rec/event-event",
            "COHAREE":"relaxed pr/event-event",
            "COMPATAR":"??",
            "PRECEE":"simple precision/event-event",
            "RECAEE":"simple recall/event-event",
            "RECNOY":"recall/core",
            "PRECNOY":"precision/core",
            "PREC_PTGRAPH":"precision/point graph",
            "REC_PTGRAPH" :"recall/point graph",
            "REC_PTGRAPH_SIMPLE" :"simple recall/point graph",
        }

    current=0
    norm={"PREC_PTGRAPH":0.5,
          "REC_PTGRAPH":0.5}
    
    def addFiles(self,filelist,xml=True,saturate=True):
        self._files_names=filelist
        for one in filelist:
            print >> sys.stderr, "doing file ",one
            if True:#try:
                if xml:
                    doc=Document(one)
                    g1=doc.get_graph()
                    flou=None# not computed on timeml TODO
                else:
                    g1=read_graphe(one,allen_algebra)
                    if g1._extra.has_key('f2'):
                        flou=float(g1._extra["f2"])
                    else:
                        flou=None
                rel_nb=0.5*len(g1.edges())
                event_nb=len(g1.nodes())
                if saturate:
                    human_coherent=g1.saturate()
                else:
                    human_coherent=True
                rel_nb_closure=0.5*len([x for x in g1.edges() if not(x==g1.algebra().universal())])
                
                self._texts[one]=TextStat(one,extra={'event_nb':event_nb,
                                                     'rel_nb':rel_nb,
                                                     'rel_nb_closure':rel_nb_closure,
                                                     'coherent':human_coherent,
                                                     'flou':flou})
                print >> sys.stderr, "done processing file ",one
            else:#except:
                print >> sys.stderr, "error processing file ",one
                self._files_names.remove(one)
                print  >> sys.stderr, sys.exc_info()
    
    def max_value_wrt_events(self,measure,visualize=True):
        data=[]
        one_expe=self.expes()[0]
        for one in self.texts():
            t=(self.getText(one))
            x=t.extra("event_nb")
            y=t.extract_referent(one_expe,measure)
            if y>0:
                data.append((x,y))
        if visualize:
            abs,ord=zip(*data)
            pylab.plot(abs,ord)
        return data
    

    def generate_floudata(self,expe_name,measure):
        """show results with disjunction level (fuzziness) as 2nd parameter
        todo: multiple subplots for a set of measures
        """
        # colormap
        cm=pylab.cm.gray
        # list of 3d points. x is level of fuzziness, y is parameter of experiment, z is measured result
        datapts=defaultdict(list)
        for one in self.texts():
            new=(self.getText(one).extract(expe_name,[measure],normalise=True))
            meas,data = new[0]
            for point in data:
                datapts[self.getText(one).extra("flou")].append((point.abscisse(),point.ordonnee()[-1]))
        return datapts
        

    def show3d(self,datapts,measure,fig=None):
        if fig is None:
            fig = pylab.figure()
        ax = Axes3D(fig)
        # plan y=x est en fait dans ces coordonnées z=1-x
        # 
        #matX0=numpy.arange(0,2,1)
        #matY0=numpy.arange(0,2,1)
        #matX, matY = numpy.meshgrid(matX0, matY0)
        #matZ=numpy.array([[1,0],[1,0]])
        #ax.plot_surface(matX,matY,matZ,color="k",alpha=0.2)
        # 
        verts = []
        xs=sorted(datapts.keys())
        for one in xs:
            if one>0:
                zs,ys=zip(*sorted(datapts[one]))
                ys=list(ys)
                # following line makes the surface go to axes zero, 
                # comment to have only the surface "around" the y=x plane
                #ys[0], ys[-1] = 0, 0
                verts.append(zip(zs,ys))
        
        poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
                                                   cc('y')]*len(datapts))
        #poly.set_alpha(0.9)
        ax.add_collection3d(poly, zs=xs, zdir='y')
        
        ax.set_xlabel("% removed")
        ax.set_ylabel("disjunction level")
        ax.set_zlabel(self.labels[measure])
        fig.show()  
        return ax,datapts

    def visualize_flou(self,expe_name,measure):
        datapts=self.generate_floudata(expe_name,measure)
        return self.show3d(datapts,measure)
        

    def visualize(self,expe_name,selection,smoothed=None,filter_func=(lambda x: True),max_abscisse=1.,regonly=True):
        """
        regonly: show only parabolic regression
        """
        datapts=defaultdict(list)
        for one in self.texts():
            if filter_func(self.getText(one)):
                new=(self.getText(one).extract(expe_name,selection,normalise=True))
                for meas,data in new:
                    if expe_name=="disturb":
                        datapts[meas].extend([x for x in data if x.abscisse()<=max_abscisse])
                    else:
                        datapts[meas].extend(data)
        #print datapts.items()
        for (i,meas) in enumerate(datapts):
            if meas.endswith('PTGRAPH'):
                #scale=0.5
                scale=1
            else:
                scale=1
            abscisses=[x.abscisse() for x in datapts[meas]]
            ordonnees=[float(x.ordonnee()[-1])*scale for x in datapts[meas]]
            

            if smoothed:# smooth to be redone. use it to display lines for now
                #pylab.subplot(2,1,2)
                pylab.plot(abscisses,ordonnees,self.symbols[i+1],label=self.labels[meas],lw=1,ls=self.line_symbols[i+1])
                #degree=smoothed
                #ordonnees=smooth_list_gaussian(ordonnees,degree=degree)
                #pylab.subplot(2,1,1)
                #abscisses=abscisses[degree-1:-(degree)]
            else:
                if not(regonly):
                    pylab.plot(abscisses,ordonnees,self.symbols[i],label=self.labels[meas])
            #coeffs=pylab.polyfit(abscisses,ordonnees,1)
            #pylab.plot(abscisses,pylab.polyval(coeffs,abscisses),label=meas)
            if not(smoothed): 
                if expe_name=="remove":
                    coeffs=pylab.polyfit(abscisses,ordonnees,2)
                else:
                    coeffs=pylab.polyfit(abscisses,ordonnees,1)
                interpolation=pylab.polyval(coeffs,abscisses)
                abscisses.sort()
                #=abscisses[1:]#[:11]
                interpolation=pylab.polyval(coeffs,abscisses)       
                # force interpolation to stay in [0,1] box      
                #viz=[(x,y) for (x,y) in zip(abscisses,interpolation)]# if y>=0.]
                #abscisses,interpolation=zip(*viz)
                pylab.plot(abscisses,interpolation,self.symbols[i],label=self.labels[meas],lw=2,ls=self.line_symbols[i])
                #print "==\n",abscisses,"\n",interpolation

        
        axes = pylab.gca()
        axes.set_ylim(bottom=-0.)
        #pylab.ylim(ymin=0,ymax= 1.0)
        pylab.axhline(color="black")
        pylab.ylabel("score")
        pylab.xlabel("% modified")
        pylab.title("experiment : %s"%expe_name)
        #pylab.draw()


    def compare_wrt_size(self,measures,style="-"):
        i=-1
        for measure1 in measures:
            i += 1
            data1=self.max_value_wrt_events(measure1,visualize=False)
            data1.sort()
            a1,o1=zip(*data1)
            pylab.plot(a1,[x*self.norm.get(measure1,1.) for x in o1],self.symbols[i],label=self.labels[measure1],ls=self.line_symbols[i])
        pylab.plot(a1,a1,ls="--",color="k",label="y=x")
        pylab.ylabel("size of reference")
        pylab.xlabel("nb of events")
        pylab.title("influence of size on reference graph size")
        

    
    def basic_stats(self):
        """lists basic stats about corpus: for each file:
        filename, nb events, nb timex, nb annotated relations, total nb of informative relations in transitive closure
        """
        header= "\t".join(["file","evt_nb","ann_rel_nb","closure_rel_nb"])
        all=[header]
        for one in self.texts():
            t=self.texts()[one]
            all.append("%s\t%d\t%d\t%d"%(one,t.extra("event_nb"),t.extra("rel_nb"),t.extra("rel_nb_closure")))
        return all


# gaussian smoothing, found somewhere (?)
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
 
    
# TODO: constructor is never called ?? check propagation 
if __name__=="__main__":
    import glob, random
    import optparse
    #filelist=["ea980120.1830.0071.tml.xml"]

    parser =  optparse.OptionParser()

    parser.add_option("-s", "--sample",type="int",default=None,
                      help="restreint a echantillon de taille donnee (defaut: non)")
    parser.add_option("-c", "--clip",type="float",default=1.0,
                      help="clip abscisse range to this maximum value (def 1.0)")
    parser.add_option("-t", "--size",default=False,action="store_true",
                      help="show size of reference wrt nb of temporal entities in the text (needs saturation =slow)")
    parser.add_option("-m", "--mode",
                      dest="mode", default="remove",
                      help=" expe perte info (mode=remove) ou perturbation (mode=disturb) ")
    parser.add_option("-w","--weight",dest="weight",action="store_true",default=False,
                      help="weight texts according to nb of relations (not implem.)")
    parser.add_option("-y","--type",dest="type",default="RECA,REC_PTGRAPH",
                      help="type of measures: list of measures among PREC,COHAllen,PRECNOY,PREC_PTGRAPH,  RECA, RECNOY,REC_PTGRAPH,RECAG, RECAEE, FINAllen, JAQ_AR")
    parser.add_option("-a","--smooth",dest="smoothed",type="int",
                      default=None,help="gaussian smoothing of data, with given degree"),
    parser.add_option("-r","--plain",default=False,action="store_true",help="read input in plain text format (not timeml)")
    parser.add_option("-f", "--flou",default=False,action="store_true",
                      help="show influence of disjunction level on measure (random graph for now)")
    parser.add_option("-p","--regonly",default=False,action="store_true",help="show only parabolic regression")
    

    (options, args) = parser.parse_args()
    if len(args)==0:
        filelist=glob.glob("*.xml")
        filelist.extend(glob.glob("*.tml"))
    else:
        filelist=args

    if options.sample is not None:
        filelist=random.sample(filelist,min(len(filelist),options.sample))


    all_measures="FINAllen COHAllen FINAR COHAR PREC RECA PRECG RECAG JAQ_AR FINAREE COHAREE COMPATAR PRECEE RECAEE RECNOY PRECNOY PREC_PTGRAPH REC_PTGRAPH"
    all_measures=set(all_measures.split())
    mes_types=options.type.split(",")


    expe={"remove":"autocomp1.csv",
          "disturb":"autocomp2.csv"
          }
    r=TemporalReport("EACL09",constructor=lambda x: Ratio(x))
    r.addFiles(filelist,saturate=options.size,xml=not(options.plain))
    

    if True:
        for one in expe:
            r.addExperiment(one,expe[one])
        #pprint.pprint(r)
        
        #t1=r.texts().values()[1]
        #print t1._extra
        #t1.visualize("remove",['RECA','REC_PTGRAPH'],t1.extra("rel_nb_closure"),idx=0)
        #t1.visualize("disturb",['RECA','RECNOY','REC_PTGRAPH'],t1.extra("rel_nb_closure"),idx=0)
        #t1.visualize("disturb",['PREC','COHAllen','PRECNOY','PREC_PTGRAPH'],1,idx=2)
        if options.size:
            #pylab.cla()
            #r.compare_wrt_size(["RECA","REC_PTGRAPH"])
            r.compare_wrt_size(mes_types)
        elif options.flou:
            ax,datapts=r.visualize_flou(options.mode,mes_types[0])
        elif options.mode=="remove":
            #r.visualize("remove",['REC_PTGRAPH',"RECAG",'RECNOY','RECA'],filter_func=lambda t: t.extra("event_nb")>0)
            r.visualize("remove",mes_types,filter_func=lambda t: t.extra("event_nb")>0,regonly=options.regonly,smoothed=options.smoothed)
        elif options.mode=="disturb":
            #r.visualize("disturb",['PREC','COHAllen','PRECNOY','PREC_PTGRAPH'],filter_func=lambda t: t.extra("event_nb")>0)
            r.visualize("disturb",mes_types,filter_func=lambda t: t.extra("event_nb")>0,regonly=options.regonly,max_abscisse=options.clip,smoothed=options.smoothed)
        if not(options.flou):
            pylab.plot([0,1],[1,0],ls="--",color="k",label="id")
        
        #pylab.axis([0,70,0,70])
        # todo: pt based ref should be / by 2
        

        pylab.legend(loc="best",fancybox=True,shadow=True)
        #pylab.legend(bbox_to_anchor=(1.05, 1), loc=2)

        pylab.show()
    else:
        print "\n".join(r.basic_stats())
