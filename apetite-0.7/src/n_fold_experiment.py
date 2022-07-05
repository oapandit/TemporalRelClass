#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


'''

Script for running n-fold cross validation experiments.

This script assumes that the fold split and model training have been
performed. See split_into_folds.py and inducer.py, respectively.

'''

import os
from apetite.TimeMLDocument import Document
from apetite.graph.Graph import merge_graphs, allen_algebra
from apetite.model import MegamClassifier
from apetite.linker import BaseLinker, BaseLinkerDisjunctive, ILPLinker, NRO_Linker, BFI_Linker, LocalSearchLinker, ZeroLinker
from apetite.utils import make_graph
#from distutils.log import threshold
from copy import deepcopy
import traceback

'''

TODO:
- create Experiment class
- massive refactoring
- parameter for eval measure?
'''

def run_one_fold( test_documents, classifier, linker_type='base', output_dir='./output', relset="allen", test_options={} ):

    ''' initialize linker '''
    if linker_type == 'zero':
        linker = ZeroLinker(classifier,
                            relset = relset,
                            test_options = test_options)
    elif linker_type == 'ilp':
        linker = ILPLinker(classifier,
                            relset = relset,
                            test_options = test_options)
    elif linker_type == 'nro':
        linker = NRO_Linker(classifier,
                            relset = relset,
                            test_options = test_options)
    elif linker_type == 'bfi':
        linker = BFI_Linker(classifier,
                              relset = relset,
                              test_options = test_options)
    elif linker_type== 'greedy' or linker_type=="hill":
        linker = LocalSearchLinker(classifier,
                                    relset = relset,
                                      test_options = test_options,
                                      mode = linker_type)
    elif linker_type== 'basedisjunct':
        linker = BaseLinkerDisjunctive(classifier,
                            relset=relset,
                            test_options=test_options)
    else:
        linker = BaseLinker(classifier,
                              relset = relset,
                              test_options = test_options)
        
    # initialize a base linker for comparison
    base_linker = BaseLinker(classifier,
                               relset = relset,
                               test_options = test_options)

    ''' collect existing predictions (for potential caching)'''
    existing_outputs = os.listdir( output_dir )

    ''' process test documents '''
    for doc in test_documents:

        # check that reference graph is consistent
        if test_options.get('consistent_gold'):
            if not doc.consistent_graph_saturation():
                print >> sys.stderr, "Annotation in doc %s are inconsistent. Skipping." %doc.id
                continue

        # limit on # of edges
        edge_ct = len(doc.get_graph().edges())
        if edge_ct > test_options.get('number_of_edges'):
            print >> sys.stderr, "Doc %s's reference graph has %s edges. Skipping." %(doc.id,edge_ct)
            continue

        # create output prediction file
        pred_file = os.path.basename( doc.id )
        pred_fh = open( os.path.join( output_dir, pred_file ),'w')

        # make sure we don't have results for this file already
        if pred_file in existing_outputs:
            print "Caching results for document %s" %doc.id
            continue

        # filter document whose # of events isn't within pre-defined bounds
        lb,ub = test_options['slice']
        evt_ct = len(doc.event_instances)
        if not lb <= evt_ct < ub:
            continue
        
        # link events in document
        print >> sys.stderr, "Linking document %s: " %doc.id,

        # run baseline linker
        base_predictions = base_linker.get_predictions( doc )
        print >> sys.stderr, "%s links" %len(base_predictions)
        if test_options.get('time_time'):
            # adding T-T and E-T relations (incl. those inferred via isodate normalization and local saturation)
            base_predictions.update( doc.get_tt_et_relations(relset=relset, isodate=True, loc_sat=True) )
        base_graph = make_graph( base_predictions )

        # perform linking with chosen linker
        predictions = []
        pred_graph = None
        
        if linker_type == 'base':
            predictions = base_predictions
            pred_graph = base_graph
        else:
            if test_options.get('inconsistent_base'):
                algebra=allen_algebra
                # only run linker if graph produced by base linker is inconsistent
                agraph = make_graph( base_predictions,algebra=algebra )
                if agraph.saturate():
                    print >> sys.stderr, "Base linker produces consistent graph. Base predictions returned."
                    write_predictions( base_predictions, pred_fh )
                    continue
                else:
                    print >> sys.stderr, "Base linker produces inconsistent graph, deferring to next linker (if any)"
            try:
            #if True:
                predictions = linker.get_predictions(doc)
                if test_options.get('time_time'):
                    # adding T-T and E-T relations (incl. those inferred via isodate normalization and local saturation)
                    predictions.update( doc.get_tt_et_relations(relset=relset, isodate=True, loc_sat=True) )
                    pred_graph = make_graph( predictions )
            except Exception, e:
            #else:
                print >> sys.stderr, "Linker crashed on this document because of error: %s. Base predictions returned." %e,
                predictions = base_predictions
                pred_graph = base_graph
                print >> sys.stderr, sys.exc_info()
                print >> sys.stderr,  traceback.print_tb(sys.exc_info()[2])

        # write predictions to output file
        write_predictions( predictions, pred_fh )

        # close file
        pred_fh.close()
    return 


def write_predictions( preds, fh ):
    for (e1, e2) in preds:
        print >> fh, preds[e1, e2], e1, e2


        

def cross_validate( folds, linker_type='base', output_dir='./output', relset="allen", test_options={} ):
    if test_options.get("increasing_size"):
        incremental_cross_validate( folds, linker_type=linker_type, output_dir=output_dir, relset=relset, test_options=test_options )
    else:
        standard_cross_validate( folds, linker_type=linker_type, output_dir=output_dir, relset=relset, test_options=test_options )
    return
    



def standard_cross_validate( folds, linker_type='base', output_dir='./output', relset="allen", test_options={} ):
    ''' fold by fold '''
    for run in folds:
        print "\nRUNNING FOLD %s..." %run
        test_files = folds[run]['test']
        model_file = folds[run]['model']
        # load test documents
        test_documents = [Document( f ) for f in test_files]
        # load model file
        model = MegamClassifier( paramfile=model_file )
        run_one_fold( test_documents,
                        model,
                        linker_type=linker_type,
                        output_dir=output_dir,
                        relset=relset,
                        test_options=test_options )
    return

    
def incremental_cross_validate( folds, linker_type='base', output_dir='./output', relset="allen", test_options={} ):
    ''' processing by document size (i.e., # of entities) '''
    test_documents = []
    doc2models = {}
    print "Loading documents and models..."
    for run in folds:
        test_files = folds[run]['test']
        model_file = folds[run]['model']
        model = MegamClassifier( paramfile=model_file )
        for f in test_files:
            doc = Document( f )    
            test_documents.append( doc )
            doc2models[doc] = model

    print "Sorting documents by graph size..."
    test_documents.sort(key=lambda doc: len(doc.events))
    # test_documents.sort(key=lambda doc: len(doc.temporal_entities))
    # test_documents.sort(key=lambda doc: len(doc.graph.nodes()))
    
    print "Linking documents..."
    for ct,doc in enumerate(test_documents):
        # print >> sys.stderr, "Document #%s: %s (%s nodes)" %(ct,doc.id,len(doc.graph.nodes()))
        run_one_fold( [doc],
                      doc2models[doc],
                      linker_type=linker_type,
                      output_dir=output_dir,
                      relset=relset,
                      test_options=test_options )
    return



def _test(relset="allen",debug=True,test_options={'annotations_only':True,"time_time":False}):
    import sys
    model_file="models/allen_sat0_beth1_disj0_local/3.megam"
    model = MegamClassifier( paramfile=model_file )
    #doc=Document("folds/tb_folds/0/test/wsj_0768_orig.tml.xml")
    
    # ce doc genere incohérence. 
    doc=Document("../data/otc251_5folds/3/test/wsj_0152_orig.tml.xml")
    graph = doc.get_graph()
    annotations = doc.get_event_event_relations()
    
    base_linker = BaseLinker(model,relset=relset,test_options=test_options)
    linker = NRO_Linker(model,relset=relset,test_options=test_options)
    if True:
    #try:
        predictions = linker.get_predictions( doc )
        pred_graph = make_graph( predictions )
        base_predictions= base_linker.get_predictions( doc )
        base_graph = make_graph( base_predictions )
        return linker,doc,predictions, pred_graph, base_predictions, base_graph
    else:
    #except:
        print >> sys.stderr, "pb with predictions, I return document and linker"
        print >> sys.stderr, sys.exc_info()
        
        return linker,doc
#################### main #####################################################



if __name__ == '__main__':

    import sys
    import os
    import optparse
    from collections import defaultdict
    
    usage = "usage: %prog [options] folds dir"
    parser = optparse.OptionParser(usage=usage)


    # general options
    parser.add_option("-r", "--relation_set", choices=["allen","bruce","tempeval","jurafsky"], default="allen", help="relation set (default: allen)")
    parser.add_option("-l", "--linker_type", choices=["base","basedisjunct","ilp","nro","bfi","greedy","hill","zero"], default="base", help="Type of linker: base,ilp,nro,bfi,greedy,hill,zero (default: base)")
    parser.add_option("-a", "--annotations_only", action="store_true", default=False, help="only use reference relations (default: False)")
    parser.add_option("-m", "--models", default='../models', help="path to model directory")
    parser.add_option("-o", "--output_dir", default='./output', help="path to output directory for experiment (default path: ./output)")
    parser.add_option("-c", "--caching", action="store_true", default=False, help="caching previous results in output_dir (default: False)")
    parser.add_option("-i", "--inconsistent_base_graphs", action="store_true", default=False, help="run experiment only on document yielding inconsistent graphs with baseline (default: False)")
    parser.add_option("-k", "--consistent_gold_graphs", action="store_true", default=False, help="skip documents whose reference is inconsistent (default: False)")
    parser.add_option("-t", "--time_time", action="store_true", default=False, help="use Time-Time and Event-Time reference relations (default: False)")
    parser.add_option("-e", "--number_of_edges", type=int, default=1000000, help="limit on # of edges for ILP solver (default: 1000000)")
    parser.add_option("-s", "--increasing_size", action="store_true", default=False, help="process documents by increasing size, instead than by fold (default: False)")
    
    # options for testing, most of which are ILP specific
    parser.add_option("-z", "--evt_nber_slice", default="0,10000", help="perform linking for documents that contains x events, where z1 <= x <= z2")
    parser.add_option("-d", "--time_limit", type=int, default=3600, help="time limit (in sec.) for ILP solver (default: 3600)")
    parser.add_option("-b", "--memory_limit", type=int, default=2000, help="memory limit (in Mb) for ILP solver (default: 2000)")
    parser.add_option("-u", "--trans_upper_limit", type=int, default=2000000, help="upper limit on number of transitivity constraints (default: 2000000)")
    parser.add_option("-x", "--threshold", type=float, default=0.0, help="threshold for some linker (eg BaseDisjunctive) (default: 0.0)")

    (options, args) = parser.parse_args()

    folds_dir = args[0]

    ############# print options
    print "\n"+"*"*100
    print "* EXPERIMENT SETTINGS"
    print "*"*100
    print "* Linker type:", options.linker_type
    print "* Relation set:", options.relation_set
    print "* Output directory:", options.output_dir
    print "* Caching old results:", options.caching
    
    ''' Test options (incl. ILP options) '''
    test_options = {
        'slice' : map( int, options.evt_nber_slice.split(',') ),
        'tmlim' : options.time_limit,
        'memlim' : options.memory_limit,
        'translim' : options.trans_upper_limit,
        'time_time' : options.time_time,
        'annotations_only': options.annotations_only,
        'inconsistent_base':options.inconsistent_base_graphs,
        'consistent_gold':options.consistent_gold_graphs,
        'number_of_edges':options.number_of_edges,
        'increasing_size':options.increasing_size,
        'threshold':options.threshold
        }

    print "* Test options:", ", ".join([":".join((str(o),str(v))) for (o,v) in test_options.items()])
    print "*"*100 

    ############# create output dir
    output_dir = options.output_dir
    if os.path.isfile(output_dir):
        raise OSError("A file with the same name as the desired dir, " \
                      "'%s', already exists." % output_dir)
    if os.path.isdir(output_dir):
        if options.caching:
            print "Caching existing results in %s" %output_dir
        else:
            print "Overriding previous results in directory %s (no caching)" %output_dir
            os.system("rm -f %s/*" %output_dir)
    else:
        os.makedirs(output_dir)


    ############ load model paths into dict indexed by fold #
    if not os.path.isdir( options.models ):
        raise OSError("Missing models directory! Provide correct path to models dir/ or use inducer.py to learn models.")
    models = {}
    for model_file in os.listdir( options.models ):
        fold_nb = int(model_file.split('.')[0])
        models[fold_nb] = os.path.join( options.models, model_file )
    
    ############ load test file paths into dict indexed by fold #
    test_files = {}
    for _dir in os.listdir( folds_dir ):
        if _dir.isdigit():
            fold_nb = int(_dir)
            test_dir = os.path.join( folds_dir, os.path.join(_dir,'test') )
            test_files[fold_nb] = [os.path.join(test_dir,f) for f in os.listdir(test_dir)
                                   if not f.startswith(".")]

    if set(models.keys()) <> set(test_files.keys()):
        sys.exit( "Error: %s models VS. %s test dirs" %(len(models),len(test_files)) )


    ########### match test files to models 
    folds = defaultdict(dict)
    for run in test_files:
        folds[run]['test'] = test_files[run]
        folds[run]['model'] = models[run]

    ############## run cross-validation experiment on folds_dir    
    cross_validate( folds,
                    linker_type = options.linker_type,
                    output_dir = output_dir,
                    relset = options.relation_set,
                    test_options = test_options)



    

    
