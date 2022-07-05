#! /usr/bin/python
# -*- coding: iso-8859-1 -*-

"""
Evaluation procedures on a whole corpus

run as main, script takes two directories,
 - one with the reference files in TimeML format (option: ascii simple format instead)
 - one with the system to be tested in ascii simple format

ex:

python ../src/Evaluation.py ../data/TimeBank1.1/docs output_baseline/ -s allen -m tempeval_recall,tempeval_prec -v > score_baseline_tempeval 2> logbaseline_tempeval

TODO:

   api evaluation par paires de graphe:
     du genre Evaluation.compare(gkey,gsys,mesure=["simple_prec","simple_recall"]) renvoie un dictionnaire de resultats par mesure

     à terme, devrait remplacer les procédures de graphe_compare ?
"""
import os.path
__author__="phil"
__date__ ="$21 déc. 2009 16:51:43$"

import glob
import os.path
import sys
import time

from apetite.TimeMLDocument import Document
from apetite.graph.Graph import CSPGraph, allen_algebra, Edge
from apetite.graph_compare import read_graphe, generic_measure, jaquard, arc_type, transform2pt,pt_recall,pt_precision
#from graph_compare import read_graphe, generic_measure, jaquard, arc_type


# A lot of different things could be tested here, in addition to measures in graph_compare
# for instance, precision here is only for edges where the systems takes a definite stand
# is it suggests a disjunction, the edge is not considered for precision, only for recall
# there could be a stricter version ?
# WARNING: only deals with event-event relations

# TODO: fonction de comparaison plus subtile:
#    - precision sur arcs simples : si disjonction qui inclut la ref, pas de pénalités
#    - precision sur arcs simples à la fois H et S
# TODO: matrice de confusion ?
def compare_verbose(r1,r2):
    """compare deux relations and outputs confusions on stderr for further analysis
    """
    if r1!=r2:
        if len(r1)==1 and len(r2)==1:
            print "err:",r1,r2
        return False
    else:
        return True


# TODO: deux fonctions, rappel et precision + options de tests



_measures_funcs={
        "simple_prec":{"compare":compare_verbose,"filtreH":(lambda x: False), 
                                "filtreS":(lambda x: (len(x.relation())==1) and arc_type(x,"event","event"))},
        "simple_recall":{"compare":compare_verbose,"filtreS":(lambda x: False), 
                                "filtreH":(lambda x: (len(x.relation())==1) and arc_type(x,"event","event"))},
        "tempeval_prec":{"compare":jaquard,"filtreH":(lambda x: False),
                                "filtreS":(lambda x: arc_type(x,"event","event"))},
        "tempeval_recall":{"compare":jaquard,"filtreS":(lambda x: False), 
                                "filtreH":(lambda x:  arc_type(x,"event","event"))},
#         results.append(generic_measure(g1,g2,g1.algebra(),type="ann",
#                                   filtreH=(lambda x: False),
#                                   filtreS=(lambda x: (len(x.relation().allen2bruce())==1))))
#    print >> sys.stderr, "PREC, prec. single ann. relations: %f/%f"%results[-1]
#    #rappel
#    results.append(generic_measure(g1,g2,g1.algebra(),type="ann",
#                                   filtreS=(lambda x: False),
#                                   filtreH=(lambda x: (len(x.relation().allen2bruce())==1))))
#    print >> sys.stderr, "RECA, recall single ann. relations: %f/%f"%results[-1]
        }


def graph_compare(gkey,gsys,measures=["simple_prec","simple_recall"],relation_set="allen"):
    result={}
    # print "Comparing graphs"
    for one in measures:
        print >> sys.stderr, one
        if one.startswith("tr_"):# transitive reduction based measures
            # print "Converting to point graph"
            gpt1,gmin1=transform2pt(gkey)
            gpt2,gmin2=transform2pt(gsys)
            # print "conversion complete"
            if one=="tr_prec":
                result[one]=pt_precision(gpt1,gpt2)
                # print "Precision calculated"
            else:
                result[one]=pt_recall(gpt1,gpt2)
                # print "recall calculated"
        else:#other measures
            measure_func=_measures_funcs.get(one)
            measure_func["type"]=relation_set
            # filters are based on allen algebra, not the algebra used for measurement
            # so edges have to be converted
            if relation_set!="allen":
                #conversion=(lambda x: Edge(x.node1(),x.node2(),(x.relation().other2allen(relation_set))))
                # TODO: tempeval in allen: change also tempeval_prec/rec
                if one=="simple_prec":
                    measure_func["filtreS"]=(lambda x: (len(x.relation().allen2other(relation_set))==1) 
                                             and arc_type(x,"event","event"))
                elif one=="simple_recall":
                    measure_func["filtreH"]=(lambda x: (len(x.relation().allen2other(relation_set))==1) 
                                             and arc_type(x,"event","event"))
                else:
                    pass
            result[one]=generic_measure(gkey,gsys,gkey.algebra(),**measure_func)
    return result

class Evaluation:
    """
    Base class for corpus evaluation
    """
    


    def __init__(self,key_directory,sys_directory,ascii_ref=False):
        """loads everything"""
        # key_files=glob.glob(os.path.join(key_directory,"*tml.xml"))
        # sys_files=glob.glob(os.path.join(sys_directory,"*tml.xml"))
        key_files=glob.glob(os.path.join(key_directory,"*.tml"))
        sys_files=glob.glob(os.path.join(sys_directory,"*.tml"))
        self._key={}
        self._measures={}
        self._ascii_ref=ascii_ref
        for one in key_files:
            filename=os.path.basename(one)
            self._key[filename]=one
            self._measures[filename]={}
        self._sys ={}
        for one in sys_files:
            filename=os.path.basename(one)
            self._sys[filename]=one
        self._global_micro_measures={}
        self._global_macro_measures={}
        self._incoherent_sys=0
        print key_files
        print sys_files

    def compute(self,measures=["simple_prec","simple_recall"],
                prediction="allen",
                relation_set="allen",
                verbose=False,
                cut=None,
                nosaturation=False):
        """ compute (micro) evaluation for measure
        prediction is the relation set of the predicted graph
        relation_set is the relation set of the evaluation
        nosaturation: to be able to evaluate the base classifier itself, use only predicted and annotated edges
        (do not saturate the graphs to be compared)

        TODO:
           x - call with set of measures, and factorize main + use graph_compare
           - factorise this mess

        """
        txt_avrg_subtotal={}
        ref_count={}
        ref_macro_count={}
        ref_macro_subtotal={}
        incoherent_sys=0
        total_sys=0

        for one in measures:
            txt_avrg_subtotal[one] = 0.
            ref_count[one]=0
            ref_macro_count[one]=0
            ref_macro_subtotal[one]=0

        reference=set(self._key.keys()) & set(self._sys.keys())
        if cut is not None:# obsolete , won't work
            # keep the last cut reference files
            cut=-cut
            reference=reference[cut:]

        for one in reference:
            print >> sys.stderr, "doing:",one
            if self._ascii_ref:
                g1 = read_graphe(self._key[one], allen_algebra)
            else:
                g1 = Document(self._key[one]).get_graph()
            g1.index_on_node1()
            t0=time.time()
            if nosaturation:
                ok1=True
            else:
                ok1=g1.saturate()
                t1=time.time()
                print >> sys.stderr, "saturation of key in %fs"%(t1-t0)
                #OP : Forcing to evaluate on inconsistent graph
                ok1 = True
            if not(ok1):
                print >> sys.stderr, "inconsistent reference; texte ignored"
                del self._measures[one]
            else:
                if self._sys.has_key(one):
                    if prediction=="allen":
                        conversion=lambda x : x
                    else:
                        conversion=lambda x : x.other2allen(prediction)
                    #OP : Changed TO READ SYSTEM TML FILE ALSO
                    # g2 = read_graphe(self._sys[one], allen_algebra,conversion=conversion)
                    if self._ascii_ref:
                        g2 = read_graphe(self._sys[one], allen_algebra,conversion=conversion)
                    else:
                        g2 = Document(self._sys[one]).get_graph()

                    g2.index_on_node1()
                    total_sys = total_sys + 1
                    t0=time.time()
                    if nosaturation:
                        ok2=True
                    else:    
                        try:
                             ok2=g2.saturate()
                        except (KeyboardInterrupt, SystemExit):
                            # if something takes too long, abort with control C and go on
                            print >> sys.stderr, "\nimpatient padawan ! saturation of graph stopped, this text will have messed up results"
                            ok2=False
                            g2 = CSPGraph(allen_algebra)
                        t1=time.time()
                        print >> sys.stderr, "saturation of system output in %fs"%(t1-t0)
                else:
                    print >> sys.stderr, "warning: no system output for text", one
                    g2 = CSPGraph(allen_algebra)
                    ok2=False

                #OP :Commented to run incoherent graph also
                # if not(ok2):
                #     print >> sys.stderr, "problem ? inconsistent system output all measures set to 0"
                #    the_eval={}
                #    for a_measure in measures:
                #        the_eval[a_measure]=(0,0)
                #else:
                #    the_eval=graph_compare(g1, g2, measures=measures,relation_set=relation_set)
                the_eval=graph_compare(g1, g2, measures=measures,relation_set=relation_set)

                for measure in measures:
                    self._measures[one][measure] = the_eval[measure]
                    # result is a couple expressing a ratio
                    tt,nn= the_eval[measure][0],the_eval[measure][1]
                    if nn==0:
                        pass
                    else:
                        ref_count[measure] += 1
                        ref_macro_count[measure] += nn
                        ref_macro_subtotal[measure] += tt
                        txt_avrg_subtotal[measure] += tt/float(nn)
                    if verbose:
                        if nn!=0:
                            print >> sys.stderr, ">>> %s:%s: %f=%f/%f"%(one,measure,tt/nn,tt,nn)
                        else:
                            print >> sys.stderr, ">>> %s:%s: 0=0/0"%(one,measure)
                #ok2=g2.saturate()
                if not(ok2):
                    incoherent_sys += 1

        self._incoherent_sys=incoherent_sys,total_sys


        for measure in measures:
            self._global_micro_measures[measure]=(txt_avrg_subtotal[measure],ref_count[measure])
            self._global_macro_measures[measure]=(ref_macro_subtotal[measure],ref_macro_count[measure])

        # for debugging purposes
        try:
            self._last=g1,g2
        except:
            self._last=None,None


    def report(self,measures=None,verbose=False):
        output=[]
        if measures is None:
            subset=self._global_micro_measures.keys()
        else:
            subset=measures

        scores = []
        for one in subset:
            total,nb_item=self._global_micro_measures[one]
            total_macro,nb_macro_item=self._global_macro_measures[one]
            if nb_item==0:
                print >> sys.stderr, ">> Global results: warning: no reference item for",one
            else:
                micro_score = total/float(nb_item)
                scores.append(micro_score)

                output.append(">> Global results for %s (micro):%f (%f/%f)"%(one,total/nb_item,total,nb_item))
                output.append(">> Global results for %s (macro):%f (%f/%f)"%(one,total_macro/nb_macro_item,total_macro,nb_macro_item))
                if verbose:
                    output.append(">>> Details ")
                    for onefile in self._measures:
                        t,n=self._measures[onefile].get(one,(0,0))
                        if n!=0:
                            output.append(">>> %s: %f=%f/%f"%(onefile,t/n,t,n))
                        else:
                            print >> sys.stderr, "warning! no prediction for file", onefile
                            output.append(">>> %s: 0=0/0"%(onefile,))
        isa, totsa = self._incoherent_sys
        if totsa!=0:
            output.append(">> Incoherent systems annotations : %f=%f/%f"%(isa/float(totsa),isa,totsa))
        if len(scores)==2:
            p = scores[0]
            r = scores[1]
            f =  2.0 * p * r / (p + r) if (p + r != 0) else 0
            scores.append(f)
        return "\n".join(output),scores


if __name__ == "__main__":
        """
        tests:
        python -s allen -m precision -v tests/test_ref tests/test_sys
        """
	from optparse import OptionParser

        usage="%prog [options] reference_directory system_output_directory"

	parser = OptionParser(usage=usage)
        parser.add_option("-s", "--relation_set",type="choice",choices=["allen","bruce","jurafsky","tempeval"],default="allen",
                          help="relation set in which the evaluation is done (default=allen=full timeml)")
        parser.add_option("-i", "--inputset",default=None,
                          help="relation set in which the predictions are expressed (default when None: evaluation relation set)")
	parser.add_option("-m", "--measure",default="simple_prec,simple_recall,tempeval_prec,tempeval_recall",
                          help="measure type (in ...)")
        parser.add_option("-a", "--ascii-format",action="store_true",default=False,
                          help="load reference graphs in ascii simple description")
	parser.add_option("-v", "--verbose",action="store_true",default=True,
                          help="detailed output per file")
        parser.add_option("-p", "--profile",action="store_true",default=False,
                          help="profiling run; only do first ten documents (default false)")
        
        parser.add_option("-b", "--nosaturation",action="store_true",default=False,
                          help="use only base predictions and reference, do not saturate the graphs before comparing")
                          

        (options, args) = parser.parse_args()

        implemented=set(["tr_recall","tr_prec","simple_prec","simple_recall","tempeval_prec","tempeval_recall"])

        key_directory=args[0]
        sys_directory=args[1]
        # measures=options.measure.split(",")
        measures = ["tr_recall","tr_prec"]
        print "GOLD annotation dir {0} and system pred dir {1}".format(key_directory,sys_directory)

        if not(set(measures).issubset(implemented)):
            raise Exception, "only implemented measures are:%s"%`implemented`

        if options.profile:
            print "Optins profile : yes"
            cut=10
            import cProfile
            import pstats
        else:
            cut=None

        if options.inputset is None:
            options.inputset=options.relation_set

        evaluation=Evaluation(key_directory,sys_directory,ascii_ref=options.ascii_format)
        #OP : chaning ascii_format to read tml file
        # evaluation = Evaluation(key_directory, sys_directory, ascii_ref=False)
        print "object created"
        if options.profile:
            cProfile.run('evaluation.compute(measures=measures,prediction=options.inputset,relation_set=options.relation_set,cut=cut,verbose=options.verbose)','/tmp/xeval.prof')
            p = pstats.Stats('/tmp/xeval.prof')
            p.strip_dirs()
            p.sort_stats('cumulative').print_stats(30)
        else:
            print "Optins profile : NO"
            evaluation.compute(measures=measures,prediction=options.inputset,relation_set=options.relation_set,verbose=options.verbose,nosaturation=options.nosaturation)
            print evaluation.report(verbose=options.verbose)
        #print >> sys.stderr, "measures computed"
