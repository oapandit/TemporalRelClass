#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

'''

Class and functions for performing accuracy and f-score evaluation:
they take a list of ( (e1, e2), relation) tuples as parameters

'''

import os
import sys



class ResultSink:

    def __init__(self):
        # document counters
        self.doc_correct = 0
        self.doc_total = 0
        # global counters
        self.doc_ct = 0
        self.correct = 0
        self.total = 0
        self.correct_by_rel = {}
        self.gold_by_rel = {}
        self.pred_by_rel = {}
        return

    def update(self, predictions, annotations):
        self.doc_ct += 1
        # reset local counters
        self.doc_total = 0
        self.doc_correct = 0
        # only use predictions that correspond to annotations
        predictions = dict([(pair,predictions[pair]) for pair in predictions
                            if pair in annotations])
        # compare predictions against annotations
        for (e1,e2),rel in predictions.items():
            self.pred_by_rel[str(rel)] = self.pred_by_rel.get(str(rel),0) + 1
        for (e1,e2),rel in annotations.items():
            self.gold_by_rel[str(rel)] = self.gold_by_rel.get(str(rel),0) + 1
            if rel == predictions.get((e1,e2),None):
                self.doc_correct += 1
                self.correct_by_rel[str(rel)] = self.correct_by_rel.get(str(rel),0) + 1
        self.doc_total = len(annotations)
        self.total += self.doc_total
        self.correct += self.doc_correct
        return 

    def accuracy(self):
        if self.total == 0:
            return 0.0
        return round(self.correct/float(self.total),3)

    def doc_accuracy(self):
        if self.doc_total == 0:
            return 0.0
        return round(self.doc_correct/float(self.doc_total),3)

    def recall(self, reltype):
        if not reltype in self.gold_by_rel:
            return 0.0
        return self.correct_by_rel.get(reltype,0)/float(self.gold_by_rel[reltype])

    def precision(self, reltype):
        if not reltype in self.pred_by_rel:
            return 0.0
        return self.correct_by_rel.get(reltype,0)/float(self.pred_by_rel[reltype])

    def fscore(self, rec, prec):
        if prec+rec == 0:
            return 0.0
        return 2*prec*rec/float((prec+rec))

    def rpf_str(self):
        lines = []
        lines.append("")
        lines.append("-"*94)        
        lines.append("| %20s | %10s %10s %10s | %10s %10s %10s |" \
          %("RelType", "Rec", "Prec", "F", "# preds", "# correct", "# gold"))
        lines.append("-"*94)
        for rel in self.gold_by_rel:
            rec = self.recall(rel)    
            prec = self.precision(rel)
            f = self.fscore(rec,prec)
            lines.append("| %20s | %10s %10s %10s | %10s %10s %10s |" \
              %(rel,round(rec,3),round(prec,3),round(f,3), self.pred_by_rel.get(rel,0), self.correct_by_rel.get(rel,0), self.gold_by_rel.get(rel,0)))
        lines.append("-"*94)
        return "\n".join(lines)
    




def readin_relations( _file ):
    relations = {}
    for l in file(_file):
        l = l.strip()
        if l.startswith('#'):
            continue
        e1, e2, rel = l.split()
        relations[e1,e2] = rel
    return relations



def score_dir( output_dir, file_ext=".pred" ):

    pred_files = dict([(f[:-len(file_ext)],f) for f in os.listdir(output_dir)
                       if f.endswith(file_ext) and not f.startswith('.')])
    gold_files = dict([(f[:-len(".gold")],f) for f in os.listdir(output_dir)
                       if f.endswith(".gold") and not f.startswith('.')])
    if set(pred_files.keys()) <> set(gold_files.keys()):
        sys.exit("Error: %s predictions files vs. %s gold files" %(len(pred_files),len(gold_files)))

    sink = ResultSink()
    print "-"*57
    print "| %-30s | %-20s |" %( "Document", "Acc." )
    print "-"*57
    for f in pred_files:
        pred_relations = readin_relations( os.path.join( output_dir, pred_files[f]) )
        gold_relations = readin_relations( os.path.join( output_dir, gold_files[f]) )
        sink.update( pred_relations, gold_relations )
        print "| %-30s | %-20s |" %( f, "%s (%s/%s)" %(sink.doc_accuracy(),sink.doc_correct,sink.doc_total) )
    print "-"*57   
    print "| %-30s | %-20s |" %("AVG. on %s docs" %sink.doc_ct, sink.accuracy() )
    print "-"*57+"\n"  
    print sink.rpf_str()

    return




if __name__ == '__main__':

    import sys
    import optparse

    usage = "usage: %prog [options] dir"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-x", "--file_extension", type=str, default='.pred', help="file extension for file selection (default: '.pred')")
    (options, args) = parser.parse_args()

    _dir = args[0]    
    score_dir( _dir, file_ext=options.file_extension )
    
