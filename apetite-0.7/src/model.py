#!/usr/bin/python

import sys
import os
import math
import tempfile
import datasource


class MegamClassifier:

    def __init__( self, paramfile=None ):
        self.paramfile = paramfile
        self.classes = []
        self.weights = {} 
        self.LABELS = "***NAMEDLABELSIDS***" 
        self.BIAS = "**BIAS**"
        self.bias_weights = []
        if self.paramfile:
            self.load_model()
        return

    def flush_model_file( self ):
        os.unlink( self.paramfile)
        return

    def load_model(self):
        """ the paramfile is a sequence of whitespace-separated lines
        the column column is a string feature label, while subsequent
        columns are the weight for that feature for class. The first
        line is a map of class *names* to *column positions* for
        example: ***NAMEDLABELSIDS*** O B I
        """
        print >> sys.stderr, "Reading parameters file: %s" %self.paramfile
        
        for l in open( self.paramfile, 'r' ):
            l = l.strip()
            parts = l.split()
            feature = intern( parts[0] )  # first column is the feature name
            if feature == self.LABELS :
                self.classes = map( str, parts[1:] )
            elif feature == self.BIAS :
                self.bias_weights = map( float, parts[1:] )
            else :
                self.weights[feature] = map( float, parts[1:] )  # next are the weights -- convert all to floats

        if self.bias_weights == []:
            self.bias_weights = [0.0 for i in xrange(len(self.weights))]
            
        #print >> sys.stderr, "Model classes (%s): %s" %(len(self.classes),self.classes)
        #print >> sys.stderr, "# of features: %s" %(len(self.weights))
        return


    def train( self, datafile, paramfile=tempfile.mktemp(dir="/tmp"), \
               prior=1, repeat=4, maxit=100, quiet=False ):
        """ simple call to megam executable for multiclass
        classification with some relevant options:
        
        -prior: precision of Gaussian prior (megam default:1)
        -repeat: repeat optimization <int> times (megam default:1)
        -maxit: max # of iterations (megam default:100)
                    
        """
        print ">>> Training Megam classifier..."
            
        self.paramfile = paramfile
        # build process command
        megam_exec = 'megam.opt'
        cmd = '%s -nc -repeat %s -lambda %s -maxi %s multiclass %s 1> %s' \
              %(megam_exec,repeat,prior,maxit,datafile,self.paramfile)
        if quiet:
            cmd += " 2> /tmp/null"
        # run command
        print >> sys.stderr, cmd
        rc = os.system( cmd )
        if rc != os.EX_OK:
            sys.exit("Error: Something went wrong with megam training. Make sure is in PATH!")
        print >> sys.stderr, "Megam parameters dumped into file %s" %self.paramfile
        return


    def test( self, datafile ):
        """ categorize feature vectors in datafile and compute
        pointwise accuracy"""
        source = datasource.SimpleClassifierSource( datafile )
        correct = 0
        total = 0
        for ( gold_cl, features ) in source:
            pred_cl = self.categorize( features )
            total += 1
            if pred_cl == gold_cl:
                correct += 1
        return correct / float(total)


    def categorize( self, features ):
        """ sum over feature weight and return class that receives
        highest overall weight 
        """
        # print >> sys.stderr, "event: %s" % features
        weights = self.bias_weights
        for f in features :
            try:
                # get weight vector for feature f
                fweights = self.weights[f]
                # addin corresponding bias weights
                weights = map( sum, zip( weights, fweights ))
            except Exception, e:
                # print >> sys.stderr, "Dereferenced unknown feature: %s" % f
                pass
        # find highest weight sum
        best_weight = max( weights )
        # return class corresponding to highest weight sum 
        return self.classes[weights.index(best_weight)]


    def class_distribution( self, features ):
        """ probability distribution over the different classes
        """
        # print >> sys.stderr, "event: %s" % features
        weights = self.bias_weights
        for f in features :
            try:
                # get weight vector for feature f
                fweights = self.weights[f]
                # add in corresponding bias weights
                weights = map( sum, zip( weights, fweights ))
            except Exception, e:
                # print >> sys.stderr, "Dereferenced unknown feature: %s" % f
                pass
        # exponentiation of weight sum
        scores = map( math.exp, weights )
        # compute normalization constant Z
        z = sum( scores )
        # compute probabilities
        probs = [ s/z for s in scores ]
        # return class/prob map
        return zip( self.classes, probs )


    def get_uniform_distribution( self, features ):
        return zip( self.classes, [1.0/len(self.classes) for c in self.classes] )

    


