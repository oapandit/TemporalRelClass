#!/usr/bin/python

"""API for Instance objects."""


class Instance:
    '''An instance is a particular data object, represented as a set of features'''
    pass


class ClassifierInstance( Instance ):
    '''A ClassifierInstance is an Instance that is used for classification.
It contains a label for the instance, a set of feature/value pairs.'''
    def __init__(self):
        self.label = None
        self.fv = []
        return

    def add(self,name,value):
        try:
            self.fv.append( ("%s=%s" %(name,stringify(value))) )
        except IndexError:
            pass
        return

    def __repr__(self):
        # TODO: normaliser l'affichage: espace, ', ", % => _ avec maketrans ? et aussi virgules bizarres
        # label    fv1 fv2 ... fvn
        return str("%s %s" %(self.label," ".join(self.fv)))



class TimeMLInstance( ClassifierInstance ):
    
    def __init__(self, label, entity1, entity2):
        ClassifierInstance.__init__(self)
        self.label = label
        self.compute_features(entity1, entity2)
        return

    
    
class TimeMLEvtEvtInstance( TimeMLInstance ):

    def __init__(self, label, entity1, entity2):
        TimeMLInstance.__init__(self, label, entity1, entity2)
        return

    def compute_features(self, ei1, ei2):
        ''' event1 specific '''
        self.add('e1_str', ei1.event.text)
        self.add('e1_class', ei1.event.cat)
        self.add('e1_tns', ei1.tense)
        self.add('e1_asp', ei1.aspect)
        self.add('e1_mod', ei1.modality)
        self.add('e1_pol', ei1.polarity)
        
        ''' event2 specific '''
        self.add('e2_str', ei2.event.text)
        self.add('e2_class', ei2.event.cat)
        self.add('e2_tns', ei2.tense)
        self.add('e2_asp', ei2.aspect)
        self.add('e2_mod', ei2.modality)
        self.add('e2_pol', ei2.polarity)

        ''' relational features '''
        ## self.add('str_pair', ei1.event.text+'_&_'+ei2.event.text)
        ## pairs of attributes
        self.add('class_pair', ei1.event.cat+'_&_'+ei2.event.cat)
        self.add('tns_pair', ei1.tense+'_&_'+ei2.tense)
        self.add('asp_pair', ei1.aspect+'_&_'+ei2.aspect)
        self.add('mod_pair', ei1.modality+'_&_'+ei2.modality)
        self.add('pol_pair', ei1.polarity+'_&_'+ei2.polarity)
        ## same X
        self.add('same_str', ei1.event.text == ei2.event.text)
        self.add('same_class', ei1.event.cat == ei2.event.cat)
        self.add('same_tns', ei1.tense == ei2.tense)
        self.add('same_asp', ei1.aspect == ei2.aspect)
        self.add('same_mod', ei1.modality == ei2.modality)
        self.add('same_pol', ei1.polarity == ei2.polarity)

        ## distance: in # of separating sentences and events
        e1_s = ei1.event.sentence; e2_s = ei2.event.sentence
        if e1_s and e2_s:
            self.add('sdist', discr(abs(e1_s.index-e2_s.index)))
        self.add('edist', discr(abs(ei1.event.index-ei2.event.index)))
                
        return



def stringify(_str):
    _str = str(_str)
    _str = _str.replace(' ','_')
    return _str



def discr(u):
    '''Map an integer to one of a fixed number of bins (0, 1, 2, 3-5, 6-10, 10+)'''
    v = u
    # values 0, 1, 2, 3-5, 6-10, 10+
    if 3 <= v <= 5:
        v = '3-5'
    elif 6 <= v <= 10:
        v = '6-10'
    elif 10 < v:
        v = '10+'
    return v





