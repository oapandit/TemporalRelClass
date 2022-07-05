'''
Utils
'''


from apetite.graph.Relation import Relation
from apetite.graph.Graph import AllenGraph, CSPGraph, Graph, Edge



def make_graph( relations,algebra=None,index=True):
    """
    make a graph from a dict of (e1,e2):relation description
    TODO ? move: should be a graph method, with an explicit algebra parameter
    """
    if algebra:
        g= CSPGraph(algebra,index=index)
    else:
        g = Graph()
    for (e1,e2) in relations:
        g.addNode(e1)
        g.addNode(e2)
        rel = relations[e1,e2]
        g.addEdge( Edge(e1,e2,Relation(rel)) )
    return g

def make_graph_from_rels( relations,algebra=None,index=True):
    """
    make a graph from a dict of (e1,e2):relation
    TODO ? move: should be a graph method, with an explicit algebra parameter
    """
    if algebra:
        g= CSPGraph(algebra,index=index)
    else:
        g = Graph()
    for (e1,e2) in relations:
        g.addNode(e1)
        g.addNode(e2)
        rel = relations[e1,e2]
        g.addEdge(rel)
    return g



import numpy 

def pts2matrix(ptslist):
    """ list of 3d points -> matrix usable by matplotlib"""
    x,y,z=zip(*ptslist)
    
    minx,maxx= min(x),max(x)
    miny,maxy= min(y),max(y)

    

    sx=(maxx-minx)/len(x)
    sy=(maxy-miny)/len(y)
    
    X, Y = numpy.meshgrid(numpy.arange(minx, maxx+sx, sx), numpy.arange(miny,maxy+sy, sy))
    
    refX=dict([(str(u),i) for (i,u) in enumerate(numpy.arange(minx, maxx+sx, sx))])
    refY=dict([(str(u),i) for (i,u) in enumerate(numpy.arange(miny, maxy+sy, sy))])
    refY["-0.0"]=refY.get("0.0","Nan")
    
    
    Z = numpy.zeros(X.shape, 'Float32')
    for d in ptslist:
        x1, y1, z1 = d
        ix = refX[str(x1)]#,"Nan")
        iy = refY[str(y1)]#,"Nan")
        Z[iy,ix] = z1
    return X,Y,Z




class ProbDist(dict):
    """ dict subclass for set of probability distribution
    """

    def update(self,rel_dict):
        """add a dict of certain information
        as relations with probability 1.
        each relation must be a simple relation otherwise
        it is ignored
        """
        for one in rel_dict:
            rel=rel_dict[one]
            if len(rel)==1:
                self[one]=[(list(rel)[0],1.0)]

    def get_sure(self):
        """return all certain information, ie keys for which self[one]=(r,1.0)"""
        res=[]
        for one in self:
            if len(self[one])==1 and self[one][0][1]==1.0:
                res.append(one)
        return res


    def cut_threshold(self,threshold,cumulative=False):
        """cut all probas below threshold to speed up search
        if no relation above threshold, keep only the best one
        cumulative: cut all probas after first selected relations when probas of selected is more than threshold
        """
        for one in self:
            probs=self[one]
            cumul=0
            new=[]
            for (x,y) in probs:
                cumul += y
                if cumulative:
                    new.append((x,y))
                    if cumul>=threshold:
                        break
                elif y>threshold:
                    new.append((x,y))
            if new==[]:
                bp,br=max([(y,x) for (x,y) in probs])
                new=[(br,bp)]
            self[one]=new
            
        
   
    def max(self,one):
        """return best relation and its probability for key: one """
        bp,br= max([(y,x) for (x,y) in self[one]])
        return br,bp

    def subset(self,one,threshold,convex=False):
        """returns minimal subset of relation above threshold cumulative probability
        if convex is True, returns a convex subset
        """
        pass


