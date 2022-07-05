#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

'''

Class for inducing pairwise classifier models from timeML files


TODO:

- resampling
- options for type of model (E-E, E-T, T-T....)
- more training options: prior, etc.
- pickle model files

'''


import tempfile
import os
import apetite
BASE = apetite.__path__[0]
DATAPATH = os.path.join(BASE, "Config")
from TimeMLDocument import Document
from graph.Relation import Relation
from graph.Graph import allen_algebra
from instance import TimeMLEvtEvtInstance
from model import MegamClassifier




class Inducer:
    pass




class MaxEntInducer( Inducer ):


    def __init__(self, files, relset="allen", saturation=False, annotations_only=False,
                 bethard_rels={}, disjunction=True, local=False):
        self.files = files
        self.relset = relset
        self.disjunction = disjunction
        self.saturation = saturation
        self.bethard_rels = bethard_rels
        self.annotations_only = annotations_only
        self.local = local
        self.data_file = tempfile.mktemp(dir="/tmp")
        self.model = MegamClassifier()
        self.unk_rel = allen_algebra.universal()
        if relset == "bruce":
            self.unk_rel = self.unk_rel.allen2bruce()
        elif relset == "tempeval":
            self.unk_rel = self.unk_rel.allen2tempeval()
        elif relset == "jurafsky":
            self.unk_rel = "[jur_unk]" #self.unk_rel.allen2jurafsky()
        return


 
    def generate_instances(self, sampling=1000, data_file=tempfile.mktemp(dir="/tmp")):
        print >> sys.stderr, "Generating training instances: ",
        # build training instances and dump them to file
        tifh = open(self.data_file,'w')
        inst_ct = 0
        univ_inst_ct = 0
        for i,f in enumerate(self.files):
            # call Document constructor
            doc = Document( f )
            annot_rels =  doc.get_relations()
            print >> sys.stderr, i
            # print doc.id
            # compute E-E training instances for doc 
            relations = doc.get_event_event_relations(relset=self.relset,
                                                      bethard_rels = self.bethard_rels,
                                                      glob_sat=self.saturation,
                                                      nodisjunction = not self.disjunction)
            entities = doc.get_events() # only events for now
            # compute instance for each ordered event-event pair
            entity_index = entities.keys()
            for i in xrange(1,len(entity_index)):
                e2 = entity_index[i]
                ent2 = entities[e2]
                for j in xrange(i-1,0,-1): # right to left
                    e1 = entity_index[j]
                    ent1 = entities[e1]
                    if self.annotations_only:
                        if (e1,e2) not in annot_rels:
                            continue
                    if self.local:
                        if (e1,e2) not in relations:
                            continue
                    rel = relations.get((e1,e2), self.unk_rel)
                    # resampling of vague relation
                    if not rel.is_simple() or rel == "[jur_unk]":
                        if i-j > sampling: # max of N events back
                            continue
                    train_inst = TimeMLEvtEvtInstance(rel,ent1,ent2)
                    print >> tifh, train_inst
                    inst_ct += 1
                    # print train_inst.label
                    print >> sys.stderr, "==inst", doc.path.split("/")[-3], doc.path.split("/")[-1], train_inst.label
        print >> sys.stderr, "%s instances dumped in %s" %(inst_ct,self.data_file)
        tifh.close()
        return


    def train_classifier(self, path="model.megam", verbose=False):
        self.model.train( self.data_file, paramfile=path, quiet=not(verbose) )
        # os.unlink( self.data_file )
        return 




def extract_bethard_relations():
    path = os.path.join(DATAPATH,'bethard-timebank-verb-clause.txt')
    bethard_rels = {} # NB: indexed by filepath (i.e. filepath -> relation dict)
    bethard_file = open(path,'r')
    for l in bethard_file:
        l = l.strip()
        if l.startswith('#'):
            continue
        fileprefix, ei1, ei2, rel = l.split()
        filename = fileprefix+'_orig.tml.xml'
        rel = Relation(rel.lower().encode()) 
        rel_dict = { (ei1,ei2): rel }
        bethard_rels[filename] = bethard_rels.get(filename,{})
        bethard_rels[filename].update(rel_dict)
    bethard_file.close()
    return bethard_rels



if __name__ == '__main__':

    import sys
    import os
    import optparse
    from collections import defaultdict

    usage = "usage: %prog [options] folds/ dir"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-r", "--relation_set", choices=["allen","bruce","tempeval","jurafsky"], default="allen", help="relation set (default: allen 13 relations)")
    parser.add_option("-s", "--saturate_graph", action="store_true", default=False, help="use saturated graphes (default: False)")
    parser.add_option("-e", "--resampling", type=int, default=1000, help="resample training instances by going back only N events back (default: 1000)")
    parser.add_option("-b", "--bethard_relations", action="store_true", default=False, help="use additional relations from Bethard (default: False)")
    parser.add_option("-d", "--disjunction", action="store_true", default=False, help="use disjunctive relations (default: False)")
    parser.add_option("-a", "--annotations_only", action="store_true", default=False, help="only use reference relations (default: False)")
    parser.add_option("-p", "--path", type=str, default="../models", help="path where to dump models (default: ../models)")
    parser.add_option("-v", "--verbose", action="store_true", default=False, help="verbose mode")
    parser.add_option("-l", "--local", action="store_true", default=False, help="local models (do not use universal rel. for unrelated pairs)")
    
    (options, args) = parser.parse_args()
    _dir = args[0]

    print >> sys.stderr, options

    # create model directory
    model_dir = options.path
    print >> sys.stderr, ">>> Creating model directory %s/" %model_dir
    if os.path.isdir(model_dir):
        os.system("rm -rf %s" % model_dir)
    elif os.path.isfile(model_dir):
        raise OSError("A file with the same name as the desired dir, " \
                      "'%s', already exists." % model_dir)
    os.makedirs(model_dir)

    # bethard relations
    bethard_relations = {}
    if options.bethard_relations:
        bethard_relations = extract_bethard_relations()
    
    # create models for different folds
    fold_dirs = [os.path.join( _dir, d) for d in os.listdir( _dir )
                 if not d.startswith(".")]



    for d in fold_dirs:
        print >> sys.stderr, ">>> Building model for fold", d
        model_name = os.path.basename(d)+".megam"
        train_subdir = os.path.join( d, 'train' )
        train_files = [os.path.join( train_subdir, f ) for f in os.listdir( train_subdir )
                       if not f.startswith(".")]
        inducer = MaxEntInducer( train_files,
                                 relset = options.relation_set,
                                 disjunction = options.disjunction,
                                 saturation=options.saturate_graph,
                                 bethard_rels=bethard_relations,
                                 annotations_only = options.annotations_only,
                                 local = options.local)
        inducer.generate_instances( sampling=options.resampling )
        inducer.train_classifier( path=os.path.join(model_dir,model_name) )
    





