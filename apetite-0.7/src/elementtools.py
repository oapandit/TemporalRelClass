# -*- coding: iso-8859-1 -*-
# modules de fonctions utiles ajoutées a ElementTree
# 

import sys

try:
    import cElementTree as ElementTree
except:
    try:
        import ElementTree
    except:
        import xml.etree.ElementTree as ElementTree


# probleme d'elementtree: 
# assure passage code accents html -> unicode
# manque oe ~
# 
# probleme "
h2u={"#235":"ë",
     "#234":"ê",
     "#233":"é",
     "#232":"è",
     "#231":"ç",
     "#238":"î",
     "#239":"i",
     "#224":"à",
     "#226":"â",
     "#244":"ô",
     "#246":"ö",
     "#251":"û",
     "#252":"ü",
     "#249":"ù",
     "#176":"°",
     #"#156":"~",
     #"quot":'',
     "apos":"'",
     '#230':"æ",
     '#171':"«",
     '#187':"»",
     }



def html2unicode(str):
    for code in h2u.keys():
        str=str.replace("&%s;"%code,h2u[code])
    return str



def parent_map(tree):
    return dict([(c,p) for p in tree.getiterator() for c in p])


# recupere seulement les fils ayant un tag dans taglist
def filter_nodes(tree,taglist):
    res=[x for x in tree if x.tag in taglist]
    return

# applique une fonction a chaque noeud ayant un tag
# dans taglist
# ?? peut modifier l'arbre sur place
#ex: affiche toutes les balises mots du tlf
#tools.apply(tree,(lambda x: sys.stdout.write(ElementTree.tostring(x)+"\n")),["mot"])
def apply(tree,func,taglist=None):
    #res=[]
    for x in tree.getiterator():
        if taglist==None or x.tag in taglist:
            #print func(x)
            #x.text=func(x)
            func(x)
        else:
            pass
    #return res

        

# get positions of the node in the text
# returns (number of first token, number of last token)
def get_location(node):
    poslist=collect(node,"pos")
    poslist= [x for x in poslist if x!="extra"]
    if poslist==[]:
        sys.stderr.write(ElementTree.tostring(node)+"\n")
        start=end=None
    else:
        start=poslist[0].split("-")[0]
        end=poslist[-1].split("-")[-1]
    return start,end

# renvoie liste des valeurs de l'attribut tag contenu dans les descendanst
# de node
def collect(node,tag):
    res=[node.attrib.get(tag)]    
    for x in node:
        res.extend(collect(x,tag))
    return [x for x in res if x!=None]


# collecte le texte contenu dans un noeud et ses descendants
def text_span(entity):
    res=entity.text
    if res==None:
        res=""
    for x in entity:
        res=res+text_span(x)
        if x.tail is not None:
            res=res+x.tail
    res=" ".join(res.split())
    return res

