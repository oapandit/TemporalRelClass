#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


'''

ILP classes, sub-classing from Pulp classes 

'''

from pulp import *
import signal
from collections import defaultdict
from copy import deepcopy
from graph.Relation import Relation
from graph.Graph import Edge, CSPGraph, AllenGraph, allen_algebra, merge_graphs
from graph_compare import ptAlgebra, ptAlgebra_relset, ptAlgebra_ext_relset, ptAlgebra_inv_explicit, ptAlgebra_compo_explicit, interval2point, point2interval2 as point2interval, _conversion as int2ptrel
from subprocess import Popen



class ILPTempGraphProblem(LpProblem):
    ''' Sub-class of LP Problem for temporal graph'''

    def __init__(self, graph, local_probs, tt_et_rels={}):
        self.entities = graph.nodes().keys()
        self.tt_et_rels = tt_et_rels
        self.local_probs = local_probs
        self.var_map = {}
        self.inv_var_map = {}
        self.var_ct = 0
        LpProblem.__init__(self, sense = LpMaximize)
        return

    def set(self, trans_limit=20e6):
        self.addVariables()
        self.setObjFcn()
        self.addConstraints( trans_limit=trans_limit )
        return


    def addVar(self, _tuple, lb, ub, vtype=LpInteger):
        varName = "x%s" %self.var_ct
        self.var_map[_tuple] = LpVariable( varName, lb, ub, vtype )
        self.inv_var_map[varName] = _tuple
        self.var_ct += 1
        return


    def addVariables(self):
        raise NotImplementedError("Super class method to be overidden") 
        return

    
    def setObjFcn(self):
        raise NotImplementedError("Super class method to be overidden") 
        return


    def addConstraints(self,trans_limit=20e6):
        raise NotImplementedError("Super class method to be overidden") 
        return


    def positive_assignments(self):
        assignments = []
        for var in self.variables():
            val = var.varValue
            # test that values are integers!
            if 0 < val < 1:
                raise LinkerError("ILP assigns value %s (non an integer) to %s" %(val,var))
            # only take 1-assignemnts
            if var.varValue == 1:
                assignments.append( var )
        return assignments


    def extract_relations(self):
        raise NotImplementedError("Super class method to be overidden") 
        return 


    def robust_solve(self, time_limit=1800, mem_limit=2000):
        ''' modification of Pulp GLPK_CMD '''
        # solver exec path
        path = "glpsol"
        # solver options
        solver_options = ['--tmlim', str(time_limit),
                          '--memlim', str(mem_limit)]
        # solver files
        pid = os.getpid()
        tmpLp = os.path.join("/tmp", "%d-pulp.lp" % pid)
        self.writeLP(tmpLp, writeSOS = 0)
        tmpSol = os.path.join("/tmp", "%d-pulp.sol" % pid)        
        # solver process
        process = ["glpsol", "--cpxlp", tmpLp, "-o", tmpSol]
        process.extend(solver_options)        
        # if quiet:
        #    solver_options = ['1>', '/tmp/null']
        # print >> sys.stderr, process

        # set the alarm signal handler, and set the alarm to tmlim
        signal.signal(signal.SIGALRM, alarmHandler)
        signal.alarm( time_limit )
        
        # solve
        try:
            #rc = os.spawnvp(os.P_WAIT, path, process)
            #rc = call( process )
            pop = Popen( process )
            # print >> sys.stderr, pop.pid
            pop.wait()
            if not os.path.exists(tmpSol):
                raise Exception("Error while executing %s. No solution file" %path)
            self.status, values = self.readsol(tmpSol)    
            self.assignVarsVals(values)
            try: os.remove(tmpLp)
            except: pass
            try: os.remove(tmpSol)
            except: pass
        except TimeOutError,e:
            print >> sys.stderr, "Killing glpsol!"
            try:
                pop.kill() # new in Python 2.6
            except AttributeError: 
                os.kill(pop.pid,signal.SIGKILL)
            raise Exception("Solver killed after time out")
        return
    

    def readsol(self,filename):
        """Read a GLPK solution file (from Pulp)"""
        f = file(filename)
        f.readline()
        rows = int(f.readline().split()[1])
        cols = int(f.readline().split()[1])
        f.readline()
        statusString = f.readline()[12:-1]
        glpkStatus = {
            "INTEGER OPTIMAL":LpStatusOptimal,
            "OPTIMAL":LpStatusOptimal,
            "INFEASIBLE (FINAL)":LpStatusInfeasible,
            "INTEGER UNDEFINED":LpStatusUndefined,
            "UNBOUNDED":LpStatusUnbounded,
            "UNDEFINED":LpStatusUndefined,
            "INTEGER EMPTY":LpStatusInfeasible
            }
        if statusString not in glpkStatus:
            raise ValueError, "Unknow status returned by GLPK"
        status = glpkStatus[statusString]
        isInteger = statusString in ["INTEGER OPTIMAL","INTEGER UNDEFINED"]
        values = {}
        for i in range(4): f.readline()
        for i in range(rows):
            line = f.readline().split()
            if len(line) ==2: f.readline()
        for i in range(3):
            f.readline()
        for i in range(cols):
            line = f.readline().split()
            name = line[1]
            if len(line) ==2: line = [0,0]+f.readline().split()
            if isInteger:
                if line[2] == "*": value = int(line[3])
                else: value = float(line[2])
            else:
                value = float(line[3])
            values[name] = value
        return status, values
    




class ILPTempIntervalGraphProblem(ILPTempGraphProblem):
    ''' Sub-class of LP Probem for temporal graph'''

    def __init__(self, graph, local_probs, tt_et_rels):
        ILPTempGraphProblem.__init__(self, graph, local_probs, tt_et_rels)
        return


    def addVariables(self):
        """ create 0-1 variables for each pair of temporal intervals
        and relation"""
        for (e1,e2) in self.local_probs:
            distrib = dict(self.local_probs[e1,e2])
            for rel in distrib:
                self.addVar( (e1,e2,rel), 0, 1 )
        print "%s LP variables" %self.var_ct
        return
    

    def setObjFcn(self):
        """ Obj function is: sum( p_ij * x_ij ) """
        terms = []
        for (e1,e2,rel) in self.var_map:
            cost = dict(self.local_probs[e1,e2])[rel]
            terms.append( cost*self.var_map[e1,e2,rel] )
        self += lpSum( terms ), 'obj'
        return


    def addConstraints(self,trans_limit=20e6):
        self.addConnectednessConstraints()
        # self.addSymmetryConstraints()
        # self.addTransitivityConstraints()
        return


    def addConnectednessConstraints(self):
        ''' at most one non-disj. relation between entities '''
        for (e1,e2) in self.local_probs:
            terms = [self.var_map[(e1,e2,rel)] for rel in dict(self.local_probs[e1,e2])]
            self += lpSum( terms ) <= 1
            # exactly one non-disj relation:
            # self += lpSum( terms ) >= 1
        return


    def addSymmetryConstraints(self):
        """ R(e1,e2) <-> R-1(e1,e3) """
        for (e1,e2,rel) in self.var_map:
            var1 = self.var_map[(e1,e2,rel)]
            inv_rel = str(allen_algebra.inverse(Relation(rel)))
            var2 = self.var_map[(e2,e1,inv_rel)]
            self += var1 <= var2
            # self += var2 <= var1 not needed since pairs not ordered
        return

    def addTransitivityConstraints(self):
        """ R(e1,e2) & R'(e2,e3) -> R o R'(e1,e3) """
        raise NotImplementedError("No compo table yet!") 
        return


    def extract_relations(self):
        relations = {}
        for var in self.positive_assignments():
            (e1,e2,rel) = self.inv_var_map[var.name]
            rel = Relation(rel)
            # keep more specific relation 
            if (e1,e2) in relations:
                relations[(e1,e2)] = relations[(e1,e2)] & rel 
            else:
                relations[(e1,e2)] = rel
        return relations 






class ILPTempPointGraphProblem(ILPTempGraphProblem):
    ''' Sub-class of ILPTempGraphProblem using conversion to point algebra'''

    def __init__(self, graph, local_probs, tt_et_rels):
        ILPTempGraphProblem.__init__(self, graph, local_probs, tt_et_rels)
        # points
        self.points = []
        for e in self.entities:
            self.points.extend([e+'_left',e+'_right'])
        # point pairs
        self.pt_pairs = []
        for p1 in self.points:
            for p2 in self.points:
                if p1 == p2:
                    continue
                self.pt_pairs.append( (p1,p2) )
        # ordered point triples
        self.pt_triples = []
        for i in xrange(len(self.points)):
            for j in xrange(i+1,len(self.points)):
                for k in xrange(j+1,len(self.points)):
                    p1,p2,p3 = self.points[i],self.points[j],self.points[k]
                    self.pt_triples.append( (p1,p2,p3) )
        self.set_pt_local_probs()
        return


    def set_pt_local_probs(self):
        print "Point conversion..."
        print "# of entity pairs probs:", len(self.local_probs)
        """ convert interval relations to point relations and
        propagate probabilities """
        self.pt_local_probs = defaultdict(dict)
        for (e1,e2) in self.local_probs:
            distrib = dict(self.local_probs[(e1,e2)])
            for one_rel in distrib: # distrib over the 13 Allen relations
                one_pr = distrib[one_rel]
                # generate pt relations from interval relation
                pt_relations = int2ptrel[ list(Relation(one_rel))[0] ] 
                # extend point graph with all basic relations
                pt_relations = [("%s_%s"%(str(e1),x[0]),"%s_%s"%(str(e2),x[1]),x[2]) \
                                 for x in pt_relations]
                # for each pt rel, keep max prob. from corresponding interval prob. 
                for (p1,p2,rel) in pt_relations:
                    if rel in self.pt_local_probs[p1,p2]:
                        self.pt_local_probs[p1,p2][rel] = max(one_pr,self.pt_local_probs[p1,p2][rel])
                    else:
                        self.pt_local_probs[p1,p2][rel] = one_pr
            # normalize to get prob distrib over pt rels
            for (p1,p2) in self.pt_local_probs:
                z = sum(self.pt_local_probs[p1,p2].values())
                for rel in self.pt_local_probs[p1,p2]:
                    self.pt_local_probs[p1,p2][rel] = self.pt_local_probs[p1,p2][rel]/float(z)
        # print self.pt_local_probs
        print '# of endpoint pairs probs:',len(self.pt_local_probs)
        return



    # def set_pt_local_probs(self):
#         """ convert interval relations to point relations and
#         propagate probabilities """
#         self.pt_local_probs = {}
#         for (e1,e2) in self.local_probs:
#             derived_probs = {}
#             distrib = dict(self.local_probs[(e1,e2)])
#             for one_rel in distrib: # distrib over the 13 Allen relations
#                 one_pr = distrib[one_rel]
#                 # generate pt relations from interval relation
#                 pt_relations = int2ptrel[ list(Relation(one_rel))[0] ]
#                 # extend point graph with all basic relations
#                 pt_relations = [("%s_%s"%(str(e1),x[0]),"%s_%s"%(str(e2),x[1]),x[2]) \
#                                  for x in pt_relations]
#                 # store pt relations with their prob. derived from int. rel. prob. 
#                 for (p1,p2,rel) in pt_relations:
#                     derived_probs.setdefault((p1,p2),[]).append((rel,one_pr))
#             # for each pair of points, get prob. distrib. over {<,>,==}
#             for (p1,p2) in derived_probs:
#                 # sum up probs derived from interval relations (no
#                 # longer a prob. distrib!) and normalize
#                 rel_prob_list = derived_probs[p1,p2]
#                 self.pt_local_probs[(p1,p2)] = {}
#                 z = 0
#                 for r,pr in rel_prob_list:
#                     self.pt_local_probs[(p1,p2)][r] = self.pt_local_probs[(p1,p2)].get(r,0.0) + pr
#                     z += pr
#                 # normalize to get real prob. distrib over {<,>,==},
#                 # assuming independence 
#                 for r in self.pt_local_probs[(p1,p2)]:
#                     self.pt_local_probs[(p1,p2)][r] = self.pt_local_probs[(p1,p2)][r]/float(z)
#         # print pt_local_probs
#         print '# of endpoint pairs for which we have probs:',len(self.pt_local_probs)
#         return
    


    def addVariables(self):
        """ create variables for triples (p1,p2,rel), where (p1,p2) is
        a pair of temporal points and rel is a pt rel (<,>,==,<=,>=)"""
        for (p1,p2) in self.pt_pairs:
            e1,e1_side = p1.split('_')
            e2,e2_side = p2.split('_')
            # relations in same interval
            if e1 == e2:
                if e1_side == "left":
                    # p1 R p2
                    self.addVar( (p1,p2,'<'), 1, 1 )
                    self.addVar( (p1,p2,'>'), 0, 0 )
                    self.addVar( (p1,p2,'=='), 0, 0 )
                    self.addVar( (p1,p2,'<='), 1, 1 )
                    self.addVar( (p1,p2,'>='), 0, 0 )
                elif e1_side == "right":
                    # p2 R p1
                    self.addVar( (p1,p2,'<'), 0, 0 )
                    self.addVar( (p1,p2,'>'), 1, 1 )
                    self.addVar( (p1,p2,'=='), 0, 0 )
                    self.addVar( (p1,p2,'<='), 0, 0 )
                    self.addVar( (p1,p2,'>='), 1, 1 )
            # know T-T and E-T relations 
            elif (p1,p2) in self.tt_et_rels:
                relset = set(self.tt_et_rels[p1,p2])
                # p1 < p2
                if relset == set(['<']):
                    self.addVar( (p1,p2,'<'), 1, 1 )
                    self.addVar( (p1,p2,'>'), 0, 0 )
                    self.addVar( (p1,p2,'=='), 0, 0 )
                    self.addVar( (p1,p2,'<='), 1, 1 )
                    self.addVar( (p1,p2,'>='), 0, 0 )
                # p1 == p2
                elif relset == set(['==']):
                    self.addVar( (p1,p2,'<'), 0, 0 )
                    self.addVar( (p1,p2,'>'), 0, 0 )
                    self.addVar( (p1,p2,'=='), 1, 1 )
                    self.addVar( (p1,p2,'<='), 1, 1 )
                    self.addVar( (p1,p2,'>='), 1, 1 )
                # p1 > p2
                elif relset == set(['>']):
                    self.addVar( (p1,p2,'<'), 0, 0 )
                    self.addVar( (p1,p2,'>'), 1, 1 )
                    self.addVar( (p1,p2,'=='), 0, 0 )
                    self.addVar( (p1,p2,'<='), 0, 0 )
                    self.addVar( (p1,p2,'>='), 1, 1 )
                # p1 <= p2
                elif relset == set(['<','==']):
                    self.addVar( (p1,p2,'<'), 0, 1 )
                    self.addVar( (p1,p2,'>'), 0, 0 )
                    self.addVar( (p1,p2,'=='), 0, 1 )
                    self.addVar( (p1,p2,'<='), 1, 1 )
                    self.addVar( (p1,p2,'>='), 0, 1 )
                # p1 >= p2
                elif relset == set(['>','==']):
                    self.addVar( (p1,p2,'<'), 0, 0 )
                    self.addVar( (p1,p2,'>'), 0, 1 )
                    self.addVar( (p1,p2,'=='), 0, 1 )
                    self.addVar( (p1,p2,'<='), 0, 1 )
                    self.addVar( (p1,p2,'>='), 1, 1 )
                # p1 <=> p2
                else:
                    for rel in ptAlgebra_ext_relset: 
                        self.addVar( (p1,p2,rel), 0, 1 )
            # remaining E-E relations (across intervals)
            else:
                for rel in ptAlgebra_ext_relset: 
                    self.addVar( (p1,p2,rel), 0, 1 )
        print "%s LP variables" %self.var_ct
        return



    def setObjFcn(self):
        """ Obj function is: sum( p_ij * x_ij ) """
        terms = []
        for (p1,p2,rel) in self.var_map:
            # add only those variables that have an assignment cost
            if rel in self.pt_local_probs[p1,p2]: 
                cost = dict(self.pt_local_probs[p1,p2])[rel]
                terms.append( cost*self.var_map[p1,p2,rel] )
        self += lpSum( terms ), 'obj'
        return


    def addConstraints(self,trans_limit=20e6):
        self.addConnectednessConstraints()
        self.addDisjunctiveConstraints()
        self.addSymmetryConstraints()
        self.addTransitivityConstraints(trans_limit)
        return


    def addConnectednessConstraints(self):
        ''' at most ONE NON-DISJ. relation between points '''
        for (p1,p2) in self.pt_pairs:
            terms = [self.var_map[(p1,p2,rel)] for rel in ptAlgebra_relset]
            self += lpSum( terms ) <= 1
            # exactly one non-disj relation:
            # self += lpSum( terms ) >= 1
        return



    def addDisjunctiveConstraints(self):
        '''
        IMPLICATIONS
        pt1 = pt2 or pt1 > pt2  ---> pt1 >= pt2   
        pt1 = pt2 or pt1 < pt2  ---> pt1 <= pt2
        EXCLUSIONS
        pt1 >= pt2 <--> not (pt1 < pt2)
        pt1 <= pt2 <--> not (pt1 > pt2)
        '''
        ct = 0
        for (p1,p2) in self.pt_pairs:
            # = or >   -->    >=
            self += lpSum([self.var_map[(p1,p2,'>')],self.var_map[(p1,p2,'==')]]) <= self.var_map[(p1,p2,'>=')]
            # = or <   -->    <= 
            self += lpSum([self.var_map[(p1,p2,'<')],self.var_map[(p1,p2,'==')]]) <= self.var_map[(p1,p2,'<=')] 
            # >=  -->   not <
            self += lpSum([self.var_map[(p1,p2,'>=')],self.var_map[(p1,p2,'<')]]) <= 1
            self += lpSum([self.var_map[(p1,p2,'>=')],self.var_map[(p1,p2,'<')]]) >= 1
            # <=  -->   not >
            self += lpSum([self.var_map[(p1,p2,'<=')],self.var_map[(p1,p2,'>')]]) <= 1
            self += lpSum([self.var_map[(p1,p2,'<=')],self.var_map[(p1,p2,'>')]]) >= 1
            ct += 6
        print "# of disj. constraints:", ct



    def addSymmetryConstraints(self):
        """ R(e1,e2) <-> R-1(e1,e3) """
        ct = 0
        for (e1,e2,rel) in self.var_map:
            var1 = self.var_map[(e1,e2,rel)]
            inv_rel = ptAlgebra_inv_explicit[rel]
            var2 = self.var_map[(e2,e1,inv_rel)]
            self += var1 <= var2
            # self += var2 <= var1 # not needed since pairs are not ordered
            ct += 1
        print "# of symmetry constraints:", ct
        return



    def addTransitivityConstraints(self,trans_limit):
        """ R(e1,e2) & R'(e2,e3) -> R o R'(e1,e3) """
        ct = 0
        # given that we have symmetry, we can use ordered triples, instead of PxPxP
        for (p1,p2,p3) in self.pt_triples:
            # print p1,p2,p3
            for r1 in ptAlgebra_compo_explicit:
                for r2 in ptAlgebra_compo_explicit[r1]:
                    r3 = ptAlgebra_compo_explicit[r1][r2]
                    var1 = self.var_map[(p1,p2,r1)]
                    var2 = self.var_map[(p2,p3,r2)]
                    var3 = self.var_map[(p1,p3,r3)]
                    # print "%s & %s => %s" %((p1,p2,r1),(p2,p3,r2),(p1,p3,r3))
                    self += lpSum([var1,var2,-var3]) <= 1
                    ct += 1
                    if ct > trans_limit:
                        raise TransLimitError("# of transivity exceeded (max:%s)" %trans_limit)
        print "# of transitivity constraints:", ct
        return


    def pt2int_conversion(self, relations ):
        # create point graph from relations
        pt_graph = CSPGraph(ptAlgebra)
        for (p1,p2) in relations:
            pt_rel = relations[p1,p2]
            pt_graph.addNode(p1)
            pt_graph.addNode(p2)
            pt_graph.addEdge( Edge(p1,p2,pt_rel) )
        # convert to interval graph
        # print pt_graph
        int_graph = point2interval( pt_graph )
        return dict([((a.node1(),a.node2()),a.rel()) for a in int_graph.edges().values()]) 



    def extract_relations(self):
        relations = {}
        for var in self.positive_assignments():
            (e1,e2,rel) = self.inv_var_map[var.name]
            # "unpack" disjunctions
            if rel == '<=':
                rel = ['<','==']
            if rel == '>=':
                rel = ['>','==']
            rel = Relation(rel)
            # keep more specific relation 
            if (e1,e2) in relations:
                relations[(e1,e2)] = relations[(e1,e2)] & rel
            else:
                relations[(e1,e2)] = rel
        return self.pt2int_conversion( relations )
    




## code to implement brut force timeout! 
def alarmHandler(signum, frame):
    raise TimeOutError("Timing out the ILP solver...")


class TimeOutError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class TransLimitError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def make_graph( relations ):
    g = AllenGraph()
    for (e1,e2) in relations:
        g.addNode(e1)
        g.addNode(e2)
        rel = relations[e1,e2]
        g.addEdge( Edge(e1,e2,Relation(rel)) )
    return g
    
    



if __name__ == '__main__':

    from TimeMLDocument import Document
    from model import MegamClassifier
    from instance import TimeMLEvtEvtInstance
    from evaluation import ResultSink

    # doc = Document("../data/aquaint_timeml_1.0/data/APW19980807.0261.tml.xml")
    # doc = Document("../data/aquaint_timeml_1.0/data/APW19980810.0907.tml.xml")
    # doc = Document("../data/aquaint_timeml_1.0/data/APW20000128.0316.tml.xml")
    doc = Document("../data/aquaint_timeml_1.0/data/XIE19980808.0060.tml.xml")
    #doc = Document("../data/aquaint_timeml_1.0/data/APW19980826.0389.tml.xml")
    # doc = Document("../data/OTC/wsj_0158_orig.tml.xml")
    
    model = MegamClassifier(paramfile="../models/aquaint-base-models/1.megam")
    #model = MegamClassifier(paramfile="../models/otc-base-models/1.megam")
    local_probs = {}
    entities = doc.event_instances

    for e1 in entities:
        for e2 in entities:
            if e1 == e2:
                continue
            if (e1,e2) not in doc.get_event_event_relations():
                continue
            test_inst = TimeMLEvtEvtInstance('?',entities[e1],entities[e2])
            prob_distrib = model.class_distribution( test_inst.fv )
            local_probs[e1,e2] = prob_distrib
            
    #for p in local_probs:
    #    print p, local_probs[p]
    graph = doc.get_event_event_graph()
    tt_et_graph = doc.get_tt_et_graph()
    unsat_graph = deepcopy(tt_et_graph)
    cons = tt_et_graph.saturate()
    if not cons:
        tt_et_graph = unsat_graph
    tt_et_rels = dict([((a.node1(),a.node2()),a.rel()) for a in tt_et_graph.edges().values()])
    pt_tt_et_graph = interval2point( tt_et_graph )
    pt_tt_et_rels = dict([((a.node1(),a.node2()),a.rel()) for a in pt_tt_et_graph.edges().values()])
    graph = merge_graphs([graph,tt_et_graph])
    
    subgraphs = graph.decompose()
    print len(subgraphs), "connex graphs"

    # intervals
    int_sols = {}
    ct = 1
    for graph in subgraphs:
        print "graph %s: %s nodes" %(ct,len(graph.nodes()))
        ct += 1      
        int_lp_prob = ILPTempIntervalGraphProblem( graph, local_probs, tt_et_rels=tt_et_rels )
        int_lp_prob.set()
        # int_lp_prob.writeLP('toto.cpxlp')
        # int_lp_prob.writeLPSOLVE('toto.lp')
        # int_lp_prob.writeMPS('toto.mps')
        int_lp_prob.robust_solve()
        int_sols.update( int_lp_prob.extract_relations() )

    g = make_graph( int_sols )
    print "consistent output graph: %s" %(g.saturate())
    sink = ResultSink()
    sink.update( int_sols, doc.get_event_event_relations() )
    print ">>>> ACC: %s (%s/%s)" %(sink.accuracy(),sink.correct,sink.total)


    # points
    pt_sols = {}
    ct = 1
    for graph in subgraphs:
        print "graph %s: %s nodes" %(ct,len(graph.nodes()))
        ct += 1

        pt_lp_prob = ILPTempPointGraphProblem( graph, local_probs, tt_et_rels=pt_tt_et_rels )
        pt_lp_prob.set()
        # pt_lp_prob.writeLP('toto.cpxlp')
        # pt_lp_prob.writeLPSOLVE('toto.lp')
        # pt_lp_prob.writeMPS('toto.mps')
        pt_lp_prob.robust_solve()
        pt_sols.update( pt_lp_prob.extract_relations() )

    g = make_graph( pt_sols )
    print "consistent output graph: %s" %(g.saturate())
    sink = ResultSink()
    sink.update( pt_sols, doc.get_event_event_relations() )
    print ">>>> ACC: %s (%s/%s)" %(sink.accuracy(),sink.correct,sink.total)

    # compare
    print int_sols == pt_sols
    print len(int_sols),len(pt_sols)
    agr = 0
    for e1,e2 in int_sols:
        if int_sols[e1,e2] == pt_sols[e1,e2]:
            agr += 1
            print "YES", e1,e2,int_sols[e1,e2], pt_sols[e1,e2]
        else:
            print "NOPE", e1,e2,int_sols[e1,e2], pt_sols[e1,e2]
    print "%s/%s" %(agr,len(int_sols))


