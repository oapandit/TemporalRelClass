#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# reproduit la tokenisation treetagger (corrig�e, cf CRTDtreetagger)
# necessaire pour s'y retrouver dans les fichiers annot�s en xml

# A SEULEMENT BESOIN DE PRESERVER LES COMPTAGES
# pas necessairement la tokenisation exacte (cf erreurs plus bas)
# et doit reproduire les eventuelles erreurs ...
# ex: extension en .nap. ->tokenise correct mais pas par treetagger...
#( baldwin train/14)


import sys, re
import locale
locale.setlocale(locale.LC_ALL, '')


# x= chaine 
# renvoie la liste des tokens+ponctuations
# 

# a faire:
#    ok - noms propres americains ex J.R.R. Tolkien = sigles avec "."
#         marche pas pour initiales simples (pb si c'est M.<->Monsieur)
#       - "..." a garder comme un signe de ponctuation
#    - sortie xml en option <token></token>
# gros bordel: quand tout marchera, refactoriser
#    - autres erreurs: 45�, n�1, mots a orthographe parenth�s�s  prenez-le(s)
# 
#
#  - utiliser maketrans/translate pour les conv. de caracteres
#    cf html2txt
#  - parametrage des conv. plus claire


# attention pas le meme que elementtools a cause "
h2u={"#235":u"�",
     "#234":u"�",
     "#233":u"�",
     "#232":u"�",
     "#231":u"�",
     "#238":u"�",
     "#239":u"i",
     "#224":u"�",
     "#226":u"�",
     "#244":u"�",
     "#246":u"�",
     "#251":u"�",
     "#252":u"�",
     "#249":u"�",
     "#176":u"�",
     #"#156":u"~",
     "quot":'"',
     "apos":u"'",
     '#230':u"�",
     '#171':u"�",
     '#187':u"�",
     }


# TODO: perf. amelior�e avec regexp compil� ?
def html2unicode(str):
    # jamais trop prudent
    str=unicode(str,"latin-1")
    for code in h2u.keys():
        str=str.replace("&%s;"%code,h2u[code])
    return str


# attention a la def de re.LOCALE pour les coupures de mots/lettres (\b\w)
# ici ca marche parce que pas d'accents aux endroits cruciaux, mais dangereux

def tokenise(x,html=True):
    # normalise la chaine et les separateurs (necessaire ?)
    res=" ".join(x.split())
    if html:
        res=html2unicode(res)
    # cas de MM. et M. -> Monsieur (attention a \s apres M.)
    res=re.sub(r"\bM?M\.\s","Mr ",res)
    # english initials -> marked with a following + 
    res=re.sub(r"\b(?P<i>[A-Z])\.\s","\g<i>+ ",res)
    # a refaire avec une regexp
    res=res.replace("--"," - ")
    res=re.sub(r"(?<!\s)-\s"," - ",res)
    res=re.sub(r"\s-(?!\s)"," - ",res)
    # range 1800-1900 -> 3 items
    res=re.sub(r"(?<=\d)-(?=\d)"," - ",res)
    res=res.replace("..."," ... ")
    # cas de a-t-il, a-t-on <=> xx-t-yy -> xx + -t-yy
    # et de pronoms rattach�es -le (avouons-le, est-il)
    res=res.replace("-t-"," -t-")
    # pronom rattach� mais pas apres "-t-" (+imperatifs mefie-toi)
    # un peu fucke because "-" est considere comme un separateur wrt \b
    # manque "allez vous-en" ..?
    # a simplifier avec forward/backward non consuming exp.
    res=re.sub(r"(?P<avant>[^-]\w)-(?P<pro>(le|la|les|y|je|ce|on|nous|vous|toi|le|il|elle|ils|elles|leur|lui|ci))\b","\g<avant> -\g<pro>",res)
    # -meme adverbe sauf pronom (lui-meme/elle-meme/...)
    res=res.replace(u"-m�me",u" -m�me")
    res=re.sub(r"\b(?P<pro>soi|vous|moi|toi|lui|elle|eux|elles|nous) -m�me",
               u"\g<pro>-m�me",res)
    # jusque-l�, ce moment-l�
    res=res.replace(u"-l�",u" -l�")
    # cas des guillemets
    # a faire: mettre � et � pour retrouver la trace apres
    # '" ' -> "� "/ ' "' -> " �"
    # TODO: regexp (plus efficace)
    res=res.replace('"',' " ')
    res=res.replace(u"�",u" � ")
    res=res.replace(u"�",u" � ")
    # incorrect (ex: URL) mais s'aligne sur Treetagger
    res=res.replace(":"," :")
    # rare mais 
    res=res.replace("))",") )")
    # s�pare la ponctuation des mots
    res=re.sub(r'[,.;:?](\s|\n|$)'," \g<0> ",res)
    res=re.sub(r'[)][,;?:.]?(\s|\n|$)'," \g<0> ",res)
    # cas des apostrophes (reste coll� au mot gauche)
    # exceptions : d'ailleurs pas general (venu d'ailleurs vs d'ailleurs)
    # mais treetagger se plante aussi
    # d'apres doit rest� coll� (d'apr�s lui) pour coller a treetagger (->ADV)
    # mais l'ann�e d'apr�s ?
    exceptions=["d'ailleurs","D'ailleurs","Aujourd'hui","aujourd'hui","d'autant","D'autant",u"d'apr�s",u"D'apr�s"]
    for one in exceptions:
        res=res.replace(one,one.replace("'",u"�"))
    res=res.replace("'","' ")
    res=res.replace(u"�","'")
    # moved/cas des guillemets
    #res=res.replace('"',' " ')
    # parenthese ouvrants, pr�c�d� d'un espace
    res=res.replace("("," ( ")
    res=res.replace(")"," ) ")
    res=res.replace("]"," ] ")
    res=res.replace("["," [ ")
    res=res.replace("{"," } ")
    res=res.replace("{"," } ")

    #
    # treetagger supprime certains espaces:
    # 20 %-> 20% / 700 000 -> 700000,  5 256 -> 5256, etc
    # peut d�passer le million
    res=re.sub(r"(?P<nb>\d+)\s+(?P<suite>((\d\d\d)|%))\s+(?P<encore>\d\d\d)","\g<nb>\g<suite>\g<encore>",res)
    res=re.sub(r"(?P<nb>\d+)\s+(?P<suite>((\d\d\d)|%))","\g<nb>\g<suite>",res)
    # rattrape les sigles decoupes
    res=re.sub(r"(?P<sigle>([A-Z]\.)+[A-Z]) \.","\g<sigle>.",res)
    # recollage divers
    recolle=[("Inc","."),
             ("-","ci"),           
             ("etc","."), # erreur de treetagger ... que faire
             ("..","."),
             ("rendez","-vous"),
             ("cessez","-le-feu"),
             ("c'",u"est-�-dire"),
             ("d'","abord ")]#unsafe: d'abord difficile
    for (a,b) in recolle:
        res=res.replace(a+" "+b,a+b)
    # ad hoc
    res=res.replace("..."," ... ")  
    # decoupage en plus 
    #decoupe=[("jusque-l�","jusque -l�")]
    #for a,b in decoupe:
    #    res=res.replace(a,b)
    #
    res=res.split()
    #
    return res

# inverse
# pas trop test�
# pas fini, manque les "" (comment faire ?)
# manque raccords certains "-" (aurions -nous pu)
def reformat(text):
    res=text
    res=text.replace(" -t-","-t-")
    res=res.replace(" -m�me","-m�me")
    res=res.replace(" -l�","-l�")
    res=re.sub(" -(?P<pro>(la|les|le))(?P<pun>[\s.,;:])","-\g<pro>\g<pun>",res)
    res=re.sub(r" -(?P<pro>(y|je|ce|on|nous|vous|toi|le|il|elle|ils|elles|leur|lui|ci))\b","-\g<pro>",res)
    # annuler les decoupage en plus
    #decoupe=[("jusque-l�","jusque -l�")]
    #for a,b in decoupe:
    #    res=res.replace(b,a)
    res=res.replace("' ","'")
    res=res.replace(" ,",",")
    res=res.replace(" .",".")
    res=res.replace("( ","(").replace(" )",")")
    res=re.sub("\s(?P<pun>[,.:;])","\g<pun>",res)
    res=re.sub(r"\bMr\b","M.",res)
    # initials (risky !) Georges W. Bush -> W+ -> W.
    res=re.sub(r"\b(?P<init>[A-Z])\+","\g<init>.",res)
    return res


if __name__=="__main__":
    if len(sys.argv)>1:
        base=open(sys.argv[1])
    else:
        base=sys.stdin
    base=base.read().decode("utf8")

    #list_lines=base.split("\n")
    res=tokenise(base,html=False)
    for x in res:
        print x.encode("utf8")
        
