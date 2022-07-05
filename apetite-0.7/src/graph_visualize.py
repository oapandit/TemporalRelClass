#!/bin/env python
# -*- coding: iso-8859-1 -*-
#

""" visualization of temporal graphs (using igraph module)
to analyse connexity of annotations
and various stuff

TODO:
      - API has changed for document
           x - new fields
           x - no need for preprocessor
    x - load ascii format
    x - load graph in any relation scheme (for ascii format)
      - show graph in any relation scheme (allen;bruce;tempeval) independent of input format
      - option: show only event-event relations
    x - options for saturation, bethard relations
    x - options for explicit time-time relations
    / - add node name + relation label on edges
    x - outputs self-connected components (-> merge this and Graphe maybe)
    x - export latex
      - use decompose(minelements=3) retourne copies des subgraph a au moins 3 elts
      - show equivalent point-based graph (transitive reduction ?)


ex usage:

 graph_visualize.py -e -a -t -s  -b /home/phil/Devel/PreApetite/data/bethard-timebank-verb-clause.txt  -g 

"""


import sys
try:
    import igraph
    import igraph.drawing as draw
except:
    print >> sys.stderr, "igraph needed for visualization"


from apetite.TimeMLDocument import Document
from apetite.inducer import extract_bethard_relations
from apetite.graph.Graph import allen_algebra, AllenGraph, Edge,  igraph2allen, _color
from apetite.graph.Relation import Relation
from apetite.graph_compare import read_graphe, transform2pt, interval2point



if __name__=="__main__":

    import time, glob, os.path
    import optparse
    usage = "usage: %prog [options] (file|directory)"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-e", "--latex", default=False, action="store_true",
                      help="export a latex file showing relations -- need the preprocessed timeml file\
                      (not implem.)")
    parser.add_option("-g", "--gold", default=False, action="store_true",
                      help="show only gold relations")
    parser.add_option("-s", "--saturate", default=False, action="store_true",
                      help="saturate the graph selected")
    parser.add_option("-t", "--time-time", dest="tt", default=False, action="store_true",
                      help="compute time-time relations")
    parser.add_option("-b", "--bethard",default=False, action="store_true",
                      help="add bethard relations")
    parser.add_option("-d","--directory", default=False, action="store_true",
                      help="to process a directory instead of a single file")
    parser.add_option("-a","--all-non-trivial",dest="allrels", default=False, action="store_true",
                      help="add an edge for all non-trivial relations (default: only simple relation)")
    parser.add_option("-p","--ascii", default=False, action="store_true",
                      help="load temporal graph in plain ascii format (beware of bad interaction with latex option)")
    parser.add_option("-f","--format", default="allen", type="choice",choices=["allen","bruce","tempeval"],
                      help="input relation format for ascii files (allen,bruce,tempeval)")
    parser.add_option("-r","--restrict",default=False, action="store_true",
                      help="show only event-event relations (not implem yet)")
 
    parser.add_option("-u","--undirected", default=False, action="store_true",
                      help="undirected graphs: no arrow shown, but self-connected components are right")
    parser.add_option("-l", "--layout", default="fr", help="graph layout method\n\
    * circle, circular: circular layout\n\
    * drl: DrL layout for large graphs \n\
    * fr, fruchterman_reingold: Fruchterman-Reingold layout \n\
    * fr_3d, fr3d, fruchterman_reingold_3d: 3D Fruchterman-Reingold layout\n\
    * graphopt: the graphopt algorithm \n\
    * gfr, grid_fr, grid_fruchterman_reingold: grid-based Fruchterman-Reingold layout\n\
    * kk, kamada_kawai: Kamada-Kawai layout \n\
    * kk_3d, kk3d, kamada_kawai_3d: 3D Kamada-Kawai layout \n\
    * lgl, large, large_graph: Large Graph Layout \n\
    * random: random layout \n\
    * random_3d: random 3D layout \n\
    * rt, tree, reingold_tilford: Reingold-Tilford tree layout \n\
    * sphere, spherical, circle_3d, circular_3d: spherical layout \n\
    ")

    parser.add_option("-x","--redux", default=False, action="store_true",
                      help="show transitive reduction of point-based translation; useful for incoherent graphs")
    parser.add_option("-c","--batch", default=False, action="store_true",
                      help="batch mode")


    (options, args) = parser.parse_args()

    
    if options.directory:
        documents=glob.glob(args[0]+"/*.tml*")
    else:
        documents=[args[0]]

    # stats on size/nb of components
    avg_size=0
    total_comp=0

    for onefile in documents:
        if options.ascii:
            if options.format=="allen":
                conversion=lambda x : x
            else:
                conversion=lambda x : x.other2allen(options.format)
            allen_graph = read_graphe(onefile, allen_algebra,conversion=conversion)
        else:
            try:
                doc=Document(onefile)#,do_prep=options.latex)
                if options.bethard:
                    # 
                    bethard_relations = extract_bethard_relations()
                else:
                    bethard_relations={}
                allen_graph=doc.get_graph(relset="allen", bethard_rels=bethard_relations, isodate=options.tt, saturation=options.saturate)
            except:
                print >> sys.stderr, "pb reading graph from", onefile
                allen_graph=AllenGraph()
        #if options.gold:
        #    allen_graph=doc.make_graph(doc.relations)
        
        #if not(isConsistent):
        #        print >> sys.stderr, "inconsistent annotation"
        if options.allrels:
            display="all"
        else:
            display="default"
        #try:
        if True:
            if options.redux:
                gpt,gmin=transform2pt(allen_graph)
                g=gmin.export2igraph(display=display,directed=not(options.undirected))
                # cycles ? -> g.topological_sorting()
                g2=gpt.export2igraph(display=display,directed=not(options.undirected))
            else:
                g=allen_graph.allen2igraph(display=display,directed=not(options.undirected))
                g2=None
            #print onefile,g.summary()
            comp=g.components()
            all_lengths=[len(x) for x in comp]
            avg_size +=sum(all_lengths)
            total_comp +=len(comp)
            if options.saturate:
                print onefile, g.vcount(), g.ecount(), len(comp), max(all_lengths), allen_graph.saturate() 
            else:
                print onefile, g.vcount(), g.ecount(), len(comp), max(all_lengths)
        #except:
        else:
            print onefile, ": failure, no edges ?"
        if not(options.directory) and not(options.batch):
            draw.plot(g,bbox=(0,0,1000,800),edge_label='test',layout=options.layout,vertex_size=20,vertex_label_dist=0,vertex_shape="hidden")
            if g2 is not None:
                draw.plot(g2,bbox=(0,0,1000,800),edge_label='test',layout=options.layout,vertex_size=20,vertex_label_dist=0,vertex_shape="hidden")
        if options.latex:
            # different names ... todo
            if True:#try:
                import codecs
                latexout=codecs.open(onefile+".tex","w",encoding="utf8")
                color=_color
                #color.update([])
                output=doc.to_latex( relset="allen",nopubtime=False,large=True,bethard_rels=bethard_relations, isodate=options.tt, saturation=options.saturate,color=color)
                latexout.write(output)
                latexout.close()
            else:
                print >> sys.stderr, "problem with file: ", onefile

    #print "moyenne taille composants>",  avg_size/float(total_comp)
