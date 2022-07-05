# PuLP : Python LP Modeler
# Version 1.20

# Copyright (c) 2002-2005, Jean-Sebastien Roy (js@jeannot.org)
# Modifications Copyright (c) 2007- Stuart Anthony Mitchell (s.mitchell@auckland.ac.nz)
# $Id: pulp.py 1791 2008-04-23 22:54:34Z smit023 $

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
PuLP: An LP modeler in Python

PuLP is an LP modeler written in python. PuLP can generate MPS or LP files
and call GLPK, COIN CLP/CBC, CPLEX and XPRESS to solve linear problems.

A comprehensive wiki can be found at http://130.216.209.237/engsci392/pulp/OptimisationWithPuLP

Use LpVariable() to create new variables. ex:
x = LpVariable("x", 0, 3)
to create a variable 0 <= x <= 3

Use LpProblem() to create new problems. ex:
prob = LpProblem("myProblem", LpMinimize)

Combine variables to create expressions and constraints and add them to the
problem. ex:
prob += x + y <= 2
If you add an expression (not a constraint, f.e. prob += 4*z + w), it will
become the objective.

Choose a solver and solve the problem. ex:
prob.solve(GLPK())

You can get the value of the variables using value(). ex:
value(x)

Exported Classes:
    - LpProblem -- Container class for a Linear programming problem
    - LpVariable -- Variables that are added to constraints in the LP
    - LpConstraint -- A constraint of the general form 
      a1x1+a2x2 ...anxn (<=, =, >=) b 
    - LpConstraintVar -- Used to construct a column of the model in column-wise 
      modelling

Exported Functions:
    - value() -- Finds the value of a variable or expression
    - lpSum() -- given a list of the form [a1*x1, a2x2, ..., anxn] will construct 
      a linear expression to be used as a constraint or variable
    - lpDot() --given two lists of the form [a1, a2, ..., an] and 
      [ x1, x2, ..., xn]will construct a linear epression to be used 
      as a constraint or variable
"""

import types
import string
import itertools
from constants import *
from solvers import *


def setConfigInformation(**keywords):
    """
    set the data in the configuration file
    at the moment will only edit things in [locations] 
    the keyword value pairs come from the keywords dictionary
    """
    #TODO: extend if we ever add another section in the config file
    #read the old configuration
    config = ConfigParser.SafeConfigParser()
    config.read(config_filename)
    #set the new keys
    for (key,val) in keywords.items():
        config.set("locations",key,val)
    #write the new configuration
    fp = open(config_filename,"w")
    config.write(fp)
    fp.close()
    

# Default solver selection
if CPLEX_MEM().available():
    LpSolverDefault = CPLEX_MEM()
elif COIN_MEM().available():
    LpSolverDefault = COIN_MEM()
elif COINMP_DLL().available():
    LpSolverDefault = COINMP_DLL()
elif GLPK_MEM().available():
    LpSolverDefault = GLPK_MEM()
elif CPLEX_CMD().available():
    LpSolverDefault = CPLEX_CMD()
elif COIN_CMD().available():
    LpSolverDefault = COIN_CMD()
elif GLPK_CMD().available():
    LpSolverDefault = GLPK_CMD()
else:
    LpSolverDefault = None

class LpElement(object):
    """Base class for LpVariable and LpConstraintVar 
    """
    #to remove illegal characters from the names
    trans = string.maketrans("-+[] ","_____")
    def setname(self,name):
        if name:
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None
    def getname(self):
        return self.__name
    name = property(fget = getname,fset = setname)
    
    def __init__(self, name):
        self.name = name
         # self.hash MUST be different for each variable
        # else dict() will call the comparison operators that are overloaded
        self.hash = id(self)
        self.modified = True

    def __hash__(self):
        return self.hash

    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

    def __neg__(self):
        return - LpAffineExpression(self)
        
    def __pos__(self):
        return self

    def __nonzero__(self):
        return 1

    def __add__(self, other):
        return LpAffineExpression(self) + other

    def __radd__(self, other):
        return LpAffineExpression(self) + other
        
    def __sub__(self, other):
        return LpAffineExpression(self) - other
        
    def __rsub__(self, other):
        return other - LpAffineExpression(self)

    def __mul__(self, other):
        return LpAffineExpression(self) * other

    def __rmul__(self, other):
        return LpAffineExpression(self) * other
        
    def __div__(self, other):
        return LpAffineExpression(self)/other

    def __rdiv__(self, other):
        raise TypeError, "Expressions cannot be divided by a variable"

    def __le__(self, other):
        return LpAffineExpression(self) <= other

    def __ge__(self, other):
        return LpAffineExpression(self) >= other

    def __eq__(self, other):
        return LpAffineExpression(self) == other

    def __ne__(self, other):
        if isinstance(other, LpVariable):
            return self.name is not other.name
        elif isinstance(other, LpAffineExpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1

  
class LpVariable(LpElement):
    """A LP variable"""
    def __init__(self, name, lowBound = None, upBound = None,
                  cat = LpContinuous, e = None):
        """
        Creates an LP variable
        
        This function creates an LP Variable with the specified associated parameters
            
        Inputs:
            - name: The name of the variable used in the output .lp file
            - lowbound -- Optional: The lower bound on this variable's range. Default is negative infinity
            - upBound -- Optional: The upper bound on this variable's range. Default is positive infinity
            - cat -- Optional: The category this variable is in, Integer or Continuous(default)
            - e -- Optional: Used for column based modelling: relates to the variable's existence in the objective function and constraints
                     
        Returns:
            - An LP variable
        """
        LpElement.__init__(self,name)
        self.lowBound = lowBound
        self.upBound = upBound
        self.cat = cat
        self.varValue = None
        self.init = 0
        #code to add a variable to constraints for column based 
        # modelling
        if e:
            self.add_expression(e)

    def add_expression(self,e):
        self.expression = e
        self.addVariableToConstraints(e)

    def matrix(self, name, indexs, lowBound = None, upBound = None, cat = 0, indexStart = []):
        if not isinstance(indexs, tuple): indexs = (indexs,)
        if "%" not in name: name += "_%d" * len(indexs)

        index = indexs[0]
        indexs = indexs[1:]
        if len(indexs) == 0:
            return [LpVariable(name % tuple(indexStart + [i]), lowBound, upBound, cat) for i in index]
        else:
            return [LpVariable.matrix(name, indexs, lowBound, upBound, cat, indexStart + [i]) for i in index]
    matrix = classmethod(matrix)

    def dicts(self, name, indexs, lowBound = None, upBound = None, cat = 0, indexStart = []):
        """
        Creates a dictionary of LP variables
        
        This function creates a dictionary of LP Variables with the specified associated parameters.
            
        Inputs:
            - name: The prefix to the name of each LP variable created
            - indexs: A list of strings of the keys to the dictionary of LP variables, and the main part of the variable name itself
            - lowbound -- Optional: The lower bound on these variables' range. Default is negative infinity
            - upBound -- Optional: The upper bound on these variables' range. Default is positive infinity
            - cat -- Optional: The category these variables are in, Integer or Continuous(default)
                                 
        Returns:
            - A dictionary of LP Variables
        """
        if not isinstance(indexs, tuple): indexs = (indexs,)
        if "%" not in name: name += "_%s" * len(indexs)

        index = indexs[0]
        indexs = indexs[1:]
        d = {}
        if len(indexs) == 0:
            for i in index:
                d[i] = LpVariable(name % tuple(indexStart + [str(i)]), lowBound, upBound, cat)
        else:
            for i in index:
                d[i] = LpVariable.dicts(name, indexs, lowBound, upBound, cat, indexStart + [i])
        return d
    dicts = classmethod(dicts)

    def dict(self, name, indexs, lowBound = None, upBound = None, cat = 0):
        if not isinstance(indexs, tuple): indexs = (indexs,)
        if "%" not in name: name += "_%s" * len(indexs)

        lists = indexs

        if len(indexs)>1:
            # Cartesian product
            res = []
            while len(lists):
                first = lists[-1]
                nres = []
                if res:
                    if first:
                        for f in first:
                            nres.extend([[f]+r for r in res])
                    else:
                        nres = res
                    res = nres
                else:
                    res = [[f] for f in first]
                lists = lists[:-1]
            index = [tuple(r) for r in res]
        elif len(indexs) == 1:
            index = indexs[0]
        else:
            return {}

        d = {}
        for i in index:
         d[i] = self(name % i, lowBound, upBound, cat)
        return d
    dict = classmethod(dict)

    def bounds(self, low, up):
        self.lowBound = low
        self.upBound = up

    def positive(self):
        self.lowBound = 0
        self.upBound = None

    def value(self):
        return self.varValue
    
    def round(self, epsInt = 1e-5, eps = 1e-7):
        if self.varValue is not None:
            if self.upBound != None and self.varValue > self.upBound and self.varValue <= self.upBound + eps:
                self.varValue = self.upBound
            elif self.lowBound != None and self.varValue < self.lowBound and self.varValue >= self.lowBound - eps:
                self.varValue = self.lowBound
            if self.cat == LpInteger and abs(round(self.varValue) - self.varValue) <= epsInt:
                self.varValue = round(self.varValue)
    
    def roundedValue(self, eps = 1e-5):
        if self.cat == LpInteger and self.varValue != None \
            and abs(self.varValue - round(self.varValue)) <= eps:
            return round(self.varValue)
        else:
            return self.varValue
        
    def valueOrDefault(self):
        if self.varValue != None:
            return self.varValue
        elif self.lowBound != None:
            if self.upBound != None:
                if 0 >= self.lowBound and 0 <= self.upBound:
                    return 0
                else:
                    if self.lowBound >= 0:
                        return self.lowBound
                    else:
                        return self.upBound
            else:
                if 0 >= self.lowBound:
                    return 0
                else:
                    return self.lowBound
        elif self.upBound != None:
            if 0 <= self.upBound:
                return 0
            else:
                return self.upBound
        else:
            return 0

    def valid(self, eps):
        if self.varValue == None: return False
        if self.upBound != None and self.varValue > self.upBound + eps:
            return False
        if self.lowBound != None and self.varValue < self.lowBound - eps:
            return False
        if self.cat == LpInteger and abs(round(self.varValue) - self.varValue) > eps:
            return False
        return True

    def infeasibilityGap(self, mip = 1):
        if self.varValue == None: raise ValueError, "variable value is None"
        if self.upBound != None and self.varValue > self.upBound:
            return self.varValue - self.upBound
        if self.lowBound != None and self.varValue < self.lowBound:
            return self.varValue - self.lowBound
        if mip and self.cat == LpInteger and round(self.varValue) - self.varValue != 0:
            return round(self.varValue) - self.varValue
        return 0

    def isBinary(self):
        return self.cat == LpInteger and self.lowBound == 0 and self.upBound == 1

    def isFree(self):
        return self.lowBound == None and self.upBound == None

    def isConstant(self):
        return self.lowBound != None and self.upBound == self.lowBound

    def isPositive(self):
        return self.lowBound == 0 and self.upBound == None

    def asCplexLpVariable(self):
        if self.isFree(): return self.name + " free"
        if self.isConstant(): return self.name + " = %.12g" % self.lowBound
        if self.lowBound == None:
            s= "-inf <= "
        # Note: XPRESS and CPLEX do not interpret integer variables without 
        # explicit bounds
        elif (self.lowBound == 0 and self.cat == LpContinuous):
            s = ""
        else:
            s= "%.12g <= " % self.lowBound
        s += self.name
        if self.upBound != None:
            s+= " <= %.12g" % self.upBound
        return s

    def asCplexLpAffineExpression(self, name, constant = 1):
        return LpAffineExpression(self).asCplexLpAffineExpression(name, constant)

    def __ne__(self, other):
        if isinstance(other, LpElement):
            return self.name is not other.name
        elif isinstance(other, LpAffineExpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1
        
    def addVariableToConstraints(self,e):
        """adds a variable to the constraints indicated by
        the LpConstraintVars in e
        """
        for constraint, coeff in e.items():
            constraint.addVariable(self,coeff)
            
    def setInitialValue(self,val):
        """sets the initial value of the Variable to val
        may of may not be supported by the solver
        """ 
        


class LpAffineExpression(dict):
    """A linear combination of LP variables"""
    #to remove illegal characters from the names
    trans = string.maketrans("-+[] ","_____")
    def setname(self,name):
        if name:
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None
        
    def getname(self):
        return self.__name
        
    name = property(fget = getname,fset = setname)

    def __init__(self, e = None, constant = 0, name = None):
        self.name = name
        #TODO remove isinstance usage
        if e is None:
            e = {}
        if isinstance(e,LpAffineExpression):
            # Will not copy the name
            self.constant = e.constant
            dict.__init__(self, e)
        elif isinstance(e,dict):
            self.constant = constant
            dict.__init__(self, e)
        elif isinstance(e,LpElement):
            self.constant = 0
            dict.__init__(self, {e:1})
        else:
            self.constant = e
            dict.__init__(self)

    # Proxy functions for variables

    def isAtomic(self):
        return len(self) == 1 and self.constant == 0 and self.values()[0] == 1

    def isNumericalConstant(self):
        return len(self) == 0

    def atom(self):
        return self.keys()[0]

    # Functions on expressions

    def __nonzero__(self):
        return float(self.constant) != 0 or len(self)

    def value(self):
        s = self.constant
        for v,x in self.iteritems():
            if v.varValue is None:
                return None
            s += v.varValue * x
        return s
        
    def valueOrDefault(self):
        s = self.constant
        for v,x in self.iteritems():
            s += v.valueOrDefault() * x
        return s
        
    def addterm(self, key, value):
            y = self.get(key, 0)
            if y:
                y += value
                if y: self[key] = y
                else: del self[key]
            else:
                self[key] = value

    def emptyCopy(self):
        return LpAffineExpression()
        
    def copy(self):
        """Make a copy of self except the name which is reset"""
        # Will not copy the name
        return LpAffineExpression(self)
        
    def __str__(self, constant = 1):
        s = ""
        for v in self:
            val = self[v]
            if val<0:
                if s != "": s += " - "
                else: s += "-"
                val = -val
            elif s != "": s += " + "
            if val == 1: s += str(v)
            else: s += str(val) + "*" + str(v)
        if constant:
            if s == "":
                s = str(self.constant)
            else:
                if self.constant < 0: s += " - " + str(-self.constant)
                elif self.constant > 0: s += " + " + str(self.constant)
        elif s == "":
            s = "0"
        return s
        
    def __repr__(self):
        l = [str(self[v]) + "*" + str(v) for v in self]
        l.append(str(self.constant))
        s = " + ".join(l)
        return s

    def asCplexLpAffineExpression(self, name, constant = 1):
        # Ugly.
        s = ""
        sl = name + ":"
        notFirst = 0
        for v,val in self.iteritems():
            if val<0:
                ns = " - "
                val = -val
            elif notFirst:
                ns = " + "
            else:
                ns = " "
            notFirst = 1
            if val == 1: ns += v.name
            else: ns += "%.12g %s" % (val, v.name)
            if len(sl)+len(ns) > LpCplexLPLineSize:
                s += sl+"\n"
                sl = ns
            else:
                sl += ns
        if not self:
            ns = " " + str(self.constant)
        else:
            ns = ""
            if constant:
                if self.constant < 0: ns = " - " + str(-self.constant)
                elif self.constant > 0: ns = " + " + str(self.constant)
        if len(sl)+len(ns) > LpCplexLPLineSize:
            s += sl+"\n"+ns+"\n"
        else:
            s += sl+ns+"\n"
        return s

    def addInPlace(self, other):
        if other is 0: return self
        if isinstance(other,LpElement):
            self.addterm(other, 1)
        elif (isinstance(other,list) 
              or isinstance(other,types.GeneratorType)):
           for e in other:
                self.addInPlace(e)
        elif isinstance(other,LpAffineExpression):
            self.constant += other.constant
            for v,x in other.iteritems():
                self.addterm(v, x)
        elif isinstance(other,dict):
            for e in other.itervalues():
                self.addInPlace(e)
        else:
            self.constant += other
        return self

    def subInPlace(self, other):
        if other is 0: return self
        if isinstance(other,LpElement):
            self.addterm(other, -1)
        elif (isinstance(other,list) 
              or isinstance(other,types.GeneratorType)):
            for e in other:
                self.subInPlace(e)
        elif isinstance(other,LpAffineExpression):
            self.constant -= other.constant
            for v,x in other.iteritems():
                self.addterm(v, -x)
        elif isinstance(other,dict):
            for e in other.itervalues():
                self.subInPlace(e)
        else:
            self.constant -= other
        return self
        
    def __neg__(self):
        e = self.emptyCopy()
        e.constant = - self.constant
        for v,x in self.iteritems():
            e[v] = - x
        return e
        
    def __pos__(self):
        return self

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)
        
    def __sub__(self, other):
        return self.copy().subInPlace(other)
        
    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __mul__(self, other):
        e = self.emptyCopy()
        if isinstance(other,LpAffineExpression):
            e.constant = self.constant * other.constant
            if len(other):
                if len(self):
                    raise TypeError, "Non-constant expressions cannot be multiplied"
                else:
                    c = self.constant
                    if c != 0:
                        for v,x in other.iteritems():
                            e[v] = c * x
            else:
                c = other.constant
                if c != 0:
                    for v,x in self.iteritems():
                        e[v] = c * x
        elif isinstance(other,LpVariable):
            return self * LpAffineExpression(other)
        else:
            if other != 0:
                e.constant = self.constant * other
                for v,x in self.iteritems():
                    e[v] = other * x
        return e

    def __rmul__(self, other):
        return self * other
        
    def __div__(self, other):
        if isinstance(other,LpAffineExpression) or isinstance(other,LpVariable):
            if len(other):
                raise TypeError, "Expressions cannot be divided by a non-constant expression"
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v,x in self.iteritems():
            e[v] = x / other
        return e

    def __rdiv__(self, other):
        e = self.emptyCopy()
        if len(self):
            raise TypeError, "Expressions cannot be divided by a non-constant expression"
        c = self.constant
        if isinstance(other,LpAffineExpression):
            e.constant = other.constant / c
            for v,x in other.iteritems():
                e[v] = x / c
        else:
            e.constant = other / c
        return e

    def __le__(self, other):
        return LpConstraint(self - other, LpConstraintLE)

    def __ge__(self, other):
        return LpConstraint(self - other, LpConstraintGE)

    def __eq__(self, other):
        return LpConstraint(self - other, LpConstraintEQ)

class LpConstraint(LpAffineExpression):
    """An LP constraint"""
    def __init__(self, e = None, sense = LpConstraintEQ,
                  name = None, rhs = None):
        LpAffineExpression.__init__(self, e, name = name)
        if rhs is not None:
            self.constant = - rhs
        self.sense = sense

    def __str__(self):
        s = LpAffineExpression.__str__(self, 0)
        if self.sense:
            s += " " + LpConstraintSenses[self.sense] + " " + str(-self.constant)
        return s

    def asCplexLpConstraint(self, name):
        # Immonde.
        s = ""
        sl = name + ":"
        notFirst = 0
        for v,val in self.iteritems():
            if val<0:
                ns = " - "
                val = -val
            elif notFirst:
                ns = " + "
            else:
                ns = " "
            notFirst = 1
            if val == 1: ns += v.name
            else: ns += "%.12g %s" % (val , v.name)
            if len(sl)+len(ns) > LpCplexLPLineSize:
                s += sl+"\n"
                sl = ns
            else:
                sl += ns
        if not self: sl += "0"
        c = -self.constant
        if c == 0: c = 0 # Supress sign
        ns = " %s %.12g" % (LpConstraintSenses[self.sense], c)
        if len(sl)+len(ns) > LpCplexLPLineSize:
            s += sl + "\n" + ns + "\n"
        else:
            s += sl + ns + "\n"
        return s

    def __repr__(self):
        s = LpAffineExpression.__repr__(self)
        if self.sense is not None:
            s += " " + LpConstraintSenses[self.sense] + " 0"
        return s
        
    def copy(self):
        """Make a copy of self"""
        return LpConstraint(self, self.sense)
        
    def emptyCopy(self):
        return LpConstraint(sense = self.sense)

    def addInPlace(self, other):
        if isinstance(other,LpConstraint):
            if self.sense * other.sense >= 0:
                LpAffineExpression.addInPlace(self, other)  
                self.sense |= other.sense
            else:
                LpAffineExpression.subInPlace(self, other)  
                self.sense |= - other.sense
        elif isinstance(other,list):
            for e in other:
                self.addInPlace(e)
        else:
            LpAffineExpression.addInPlace(self, other)
            #raise TypeError, "Constraints and Expressions cannot be added"
        return self

    def subInPlace(self, other):
        if isinstance(other,LpConstraint):
            if self.sense * other.sense <= 0:
                LpAffineExpression.subInPlace(self, other)  
                self.sense |= - other.sense
            else:
                LpAffineExpression.addInPlace(self, other)  
                self.sense |= other.sense
        elif isinstance(other,list):
            for e in other:
                self.subInPlace(e)
        else:
            LpAffineExpression.addInPlace(self, other)
            #raise TypeError, "Constraints and Expressions cannot be added"
        return self
        
    def __neg__(self):
        c = LpAffineExpression.__neg__(self)
        c.sense = - c.sense
        return c

    def __add__(self, other):
        return self.copy().addInPlace(other)
        
    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __mul__(self, other):
        if isinstance(other,LpConstraint):
            c = LpAffineExpression.__mul__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return LpAffineExpression.__mul__(self, other)
        
    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other,LpConstraint):
            c = LpAffineExpression.__div__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return LpAffineExpression.__mul__(self, other)

    def __rdiv__(self, other):
        if isinstance(other,LpConstraint):
            c = LpAffineExpression.__rdiv__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return LpAffineExpression.__mul__(self, other)

    def valid(self, eps = 0):
        val = self.value()
        if self.sense == LpConstraintEQ: return abs(val) <= eps
        else: return val * self.sense >= - eps

class LpConstraintVar(LpElement):
    """A Constraint that can be treated as a variable when constructing
    a LpProblem by columns
    """
    def __init__(self, name = None ,sense = None, 
                 rhs = None, e = None):
        LpElement.__init__(self,name)
        self.constraint = LpConstraint(name = self.name, sense = sense,
                                       rhs = rhs , e = e)
        
    def addVariable(self, var, coeff):
        """Adds more a varible to the constraint with the
        activity coeff
        """
        self.constraint.addterm(var, coeff)
        
    def value(self):
        return self.constraint.value()

class LpProblem:
    """An LP Problem"""
    def __init__(self, name = "NoName", sense = LpMinimize):
        """
        Creates an LP Problem
        
        This function creates a new LP Problem  with the specified associated parameters
            
        Inputs:
            - name -- The name of the problem used in the output .lp file
            - sense(optional) -- The LP problem objective: LpMinimize(default) or LpMaximise
                    
        Returns:
            - An LP Problem
        """
        self.objective = None
        self.constraints = {}
        self.name = name
        self.sense = sense
        self.sos1 = {}
        self.sos2 = {}
        self.status = LpStatusNotSolved
        self.noOverlap = 1
        self.solver = None
        self.initialValues = {}
        self.modifiedVariables = []
        self.modifiedConstraints = []
        self.resolveOK = False

        
        # locals
        self.lastUnused = 0

    def __repr__(self):
        string = self.name+":\n"
        if self.sense == 1:
            string += "MINIMIZE\n"
        else:
            string += "MAXIMIZE\n"
        string += repr(self.objective) +"\n"

        if self.constraints:
            string += "SUBJECT TO\n"
            for n, c in self.constraints.iteritems():
                string += c.asCplexLpConstraint(n) +"\n"
        string += "VARIABLES\n"
        for v in self.variables():
            string += v.asCplexLpVariable() + " " + LpCategories[v.cat] + "\n"
        return string

    def copy(self):
        """Make a copy of self. Expressions are copied by reference"""
        lpcopy = LpProblem(name = self.name, sense = self.sense)
        lpcopy.objective = self.objective
        lpcopy.constraints = self.constraints.copy()
        lpcopy.sos1 = self.sos1.copy()
        lpcopy.sos2 = self.sos2.copy()
        return lpcopy

    def deepcopy(self):
        """Make a copy of self. Expressions are copied by value"""
        lpcopy = LpProblem(name = self.name, sense = self.sense)
        if lpcopy.objective != None:
            lpcopy.objective = self.objective.copy()
        lpcopy.constraints = {}
        for k,v in self.constraints.iteritems():
            lpcopy.constraints[k] = v.copy()
        lpcopy.sos1 = self.sos1.copy()
        lpcopy.sos2 = self.sos2.copy()
        return lpcopy

    def normalisedNames(self):
        constraintsNames = {}
        i = 0
        for k in self.constraints:
            constraintsNames[k] = "C%07d" % i
            i += 1
        variablesNames = {}
        i = 0
        for k in self.variables():
            variablesNames[k.name] = "X%07d" % i
            i += 1
        return constraintsNames, variablesNames, "OBJ"

    def isMIP(self):
        for v in self.variables():
            if v.cat == LpInteger: return 1
        return 0

    def roundSolution(self, epsInt = 1e-5, eps = 1e-7):
        """
        Rounds the lp variables
        
        Inputs:
            - none
        
        Side Effects:
            - The lp variables are rounded
        """
        for v in self.variables():
            v.round(epsInt, eps)

    def unusedConstraintName(self):
        self.lastUnused += 1
        while 1:
            s = "_C%d" % self.lastUnused
            if s not in self.constraints: break
            self.lastUnused += 1
        return s

    def valid(self, eps = 0):
        for v in self.variables():
            if not v.valid(eps): return False
        for c in self.constraints.itervalues():
            if not c.valid(eps): return False
        else:
            return True

    def infeasibilityGap(self, mip = 1):
        gap = 0
        for v in self.variables():
            gap = max(abs(v.infeasibilityGap(mip)), gap)
        for c in self.constraints.itervalues():
            if not c.valid(0):
                gap = max(abs(c.value()), gap)
        return gap

    def variables(self):
        """
        Returns a list of the problem variables
        
        Inputs:
            - none
        
        Returns:
            - A list of the problem variables
        """
        variables = {}
        if self.objective:
            variables.update(self.objective)
        for c in self.constraints.itervalues():
            variables.update(c)
        return variables.keys()

    def variablesDict(self):
        variables = {}
        if self.objective:
            for v in self.objective:
                variables[v.name] = v
        for c in self.constraints.values():
            for v in c:
                variables[v.name] = v
        return variables
    
    def add(self, constraint, name = None):
        self.addConstraint(constraint, name)

    def addConstraint(self, constraint, name = None):
        if not isinstance(constraint, LpConstraint):
            raise TypeError, "Can only add LpConstraint objects"
        if name:
            constraint.name = name 
        try:
            if constraint.name:
                name = constraint.name
            else:
                name = self.unusedConstraintName()
        except AttributeError:
            raise TypeError, "Can only add LpConstraint objects"            
            #removed as this test fails for empty constraints
#        if len(constraint) == 0:
#            if not constraint.valid():
#                raise ValueError, "Cannot add false constraints"
        if name in self.constraints:
            if self.noOverlap:
                raise "overlapping constraint names: " + name
            else:
                print "Warning: overlapping constraint names:", name
        self.constraints[name] = constraint
        self.modifiedConstraints.append(constraint)

    def setObjective(self,obj):
        """
        Sets the input variable as the objective function. Used in Columnwise Modelling
        
        Inputs:
            - obj -- the objective function of type LpConstraintVar
            
        Side Effects:
            - The objective function is set
        """
        try:
            obj = obj.constraint
            name = obj.name
        except AttributeError:
            name = None
        self.objective = obj
        self.objective.name = name
    
    def __iadd__(self, other):
        if isinstance(other, tuple):
            other, name = other
        else:
            name = None
        if other is True:
            return self
        if isinstance(other, LpConstraintVar):
            self.addConstraint(other.constraint)
        elif isinstance(other, LpConstraint):
            self.addConstraint(other, name)
        elif isinstance(other, LpAffineExpression):
            self.objective = other
            self.objective.name = name
        elif isinstance(other, LpVariable) or type(other) in [int, float]:
            self.objective = LpAffineExpression(other)
            self.objective.name = name
        else:
            raise TypeError, "Can only add LpConstraintVar, LpConstraint, LpAffineExpression or True objects"
        return self
    
    def extend(self, contraintes):
        if isinstance(contraintes, dict):
            for name in contraintes:
                self.constraints[name] = contraintes[name]
        else:
            for c in contraintes:
                if isinstance(c,tuple):
                    name = c[0]
                    c = c[1]
                else:
                    name = None
                if not name: name = c.name
                if not name: name = self.unusedConstraintName()
                self.constraints[name] = c

    def coefficients(self, translation = None):
        coefs = []
        if translation == None:
            for c in self.constraints:
                cst = self.constraints[c]
                coefs.extend([(v.name, c, cst[v]) for v in cst])
        else:
            for c in self.constraints:
                ctr = translation[c]
                cst = self.constraints[c]
                coefs.extend([(translation[v.name], ctr, cst[v]) for v in cst])
        return coefs
        
    def writeMPS(self, filename, mpsSense = 0, rename = 0, mip = 1):
        wasNone, dummyVar = self.fixObjective()
        f = file(filename, "w")
        if mpsSense == 0: mpsSense = self.sense
        cobj = self.objective
        if mpsSense != self.sense:
            n = cobj.name
            cobj = - cobj
            cobj.name = n
        if rename:
            constraintsNames, variablesNames, cobj.name = self.normalisedNames()
        f.write("*SENSE:"+LpSenses[mpsSense]+"\n")
        n = self.name
        if rename: n = "MODEL"
        f.write("NAME          "+n+"\n")
        vs = self.variables()
        # constraints
        f.write("ROWS\n")
        objName = cobj.name
        if not objName: objName = "OBJ"
        f.write(" N  %s\n" % objName)
        mpsConstraintType = {LpConstraintLE:"L", LpConstraintEQ:"E", LpConstraintGE:"G"}
        for k,c in self.constraints.iteritems():
            if rename: k = constraintsNames[k]
            f.write(" "+mpsConstraintType[c.sense]+"  "+k+"\n")
        # matrix
        f.write("COLUMNS\n")
        # Creation of a dict of dict:
        # coefs[nomVariable][nomContrainte] = coefficient       
        coefs = {}
        for k,c in self.constraints.iteritems():
            if rename: k = constraintsNames[k]
            for v in c:
                n = v.name
                if rename: n = variablesNames[n]
                if n in coefs:
                    coefs[n][k] = c[v]
                else:
                    coefs[n] = {k:c[v]}
        
        for v in vs:
            if mip and v.cat == LpInteger:
                f.write("    MARK      'MARKER'                 'INTORG'\n")
            n = v.name
            if rename: n = variablesNames[n]
            if n in coefs:
                cv = coefs[n]
                # Most of the work is done here
                for k in cv: f.write("    %-8s  %-8s  % .5e\n" % (n,k,cv[k]))

            # objective function
            if v in cobj: f.write("    %-8s  %-8s  % .5e\n" % (n,objName,cobj[v]))
            if mip and v.cat == LpInteger:
                f.write("    MARK      'MARKER'                 'INTEND'\n")
        # right hand side
        f.write("RHS\n")
        for k,c in self.constraints.iteritems():
            c = -c.constant
            if rename: k = constraintsNames[k]
            if c == 0: c = 0
            f.write("    RHS       %-8s  % .5e\n" % (k,c))
        # bounds
        f.write("BOUNDS\n")
        for v in vs:
            n = v.name
            if rename: n = variablesNames[n]
            if v.lowBound != None and v.lowBound == v.upBound:
                f.write(" FX BND       %-8s  % .5e\n" % (n, v.lowBound))
            elif v.lowBound == 0 and v.upBound == 1 and mip and v.cat == LpInteger:
                f.write(" BV BND       %-8s\n" % n)
            else:
                if v.lowBound != None:
                    # In MPS files, variables with no bounds (i.e. >= 0)
                    # are assumed BV by COIN and CPLEX.
                    # So we explicitly write a 0 lower bound in this case.
                    if v.lowBound != 0 or (mip and v.cat == LpInteger and v.upBound == None):
                        f.write(" LO BND       %-8s  % .5e\n" % (n, v.lowBound))
                else:
                    if v.upBound != None:
                        f.write(" MI BND       %-8s\n" % n)
                    else:
                        f.write(" FR BND       %-8s\n" % n)
                if v.upBound != None:
                    f.write(" UP BND       %-8s  % .5e\n" % (n, v.upBound))
        f.write("ENDATA\n")
        f.close()
        self.restoreObjective(wasNone, dummyVar)
        # returns the variables, in writting order
        if rename == 0:
            return vs
        else:
            return vs, variablesNames, constraintsNames, cobj.name
        
    def writeLP(self, filename, writeSOS = 1, mip = 1):
        """
        Write the given Lp problem to a .lp file.
        
        This function writes the specifications (objective function,
        constraints, variables) of the defined Lp problem to a file.
        
        Inputs:
            - filename -- the name of the file to be created.          
                
        Side Effects:
            - The file is created.
        """
        f = file(filename, "w")
        f.write("\\* "+self.name+" *\\\n")
        if self.sense == 1:
            f.write("Minimize\n")
        else:
            f.write("Maximize\n")
        wasNone, dummyVar = self.fixObjective()
        objName = self.objective.name
        if not objName: objName = "OBJ"
        f.write(self.objective.asCplexLpAffineExpression(objName, constant = 0))
        f.write("Subject To\n")
        ks = self.constraints.keys()
        ks.sort()
        for k in ks:
            f.write(self.constraints[k].asCplexLpConstraint(k))
        vs = self.variables()
        vs.sort()
        # Bounds on non-"positive" variables
        # Note: XPRESS and CPLEX do not interpret integer variables without 
        # explicit bounds
        if mip:
            vg = [v for v in vs if not (v.isPositive() and v.cat == LpContinuous) \
                and not v.isBinary()]
        else:
            vg = [v for v in vs if not v.isPositive()]
        if vg:
            f.write("Bounds\n")
            for v in vg:
                f.write("%s\n" % v.asCplexLpVariable())
        # Integer non-binary variables
        if mip:
            vg = [v for v in vs if v.cat == LpInteger and not v.isBinary()]
            if vg:
                f.write("Generals\n")
                for v in vg: f.write("%s\n" % v.name)
            # Binary variables
            vg = [v for v in vs if v.isBinary()]
            if vg:
                f.write("Binaries\n")
                for v in vg: f.write("%s\n" % v.name)
        # Special Ordered Sets
        if writeSOS and (self.sos1 or self.sos2):
            f.write("SOS\n")
            if self.sos1:
                for sos in self.sos1.itervalues():
                    f.write("S1:: \n")
                    for v,val in sos.iteritems():
                        f.write(" %s: %.12g\n" % (v.name, val))
            if self.sos2:
                for sos in self.sos2.itervalues():
                    f.write("S2:: \n")
                    for v,val in sos.iteritems():
                        f.write(" %s: %.12g\n" % (v.name, val))
        f.write("End\n")
        f.close()
        self.restoreObjective(wasNone, dummyVar)
        
    def assignVarsVals(self, values):
        variables = self.variablesDict()
        for name in values:
            try:
                variables[name].varValue = values[name]
            except KeyError:
                print >> sys.stderr, "Pulp Warning: var %s not found!" %name
            
    def assignVarsDj(self,values):
        variables = self.variablesDict()
        for name in values:
            variables[name].dj = values[name]
            
    def assignConsPi(self, values):
        for name in values:
            self.constraints[name].pi = values[name]
            
    def assignConsSlack(self, values):
        for name in values:
            self.constraints[name].slack = float(values[name])

    def fixObjective(self):
        if self.objective is None:
            self.objective = 0
            wasNone = 1
        else:
            wasNone = 0
        if not isinstance(self.objective, LpAffineExpression):
            self.objective = LpAffineExpression(self.objective)
        if self.objective.isNumericalConstant():
            dummyVar = LpVariable("__dummy", 0, 0)
            self.objective += dummyVar
        else:
            dummyVar = None
        return wasNone, dummyVar

    def restoreObjective(self, wasNone, dummyVar):
        if wasNone:
            self.objective = None
        elif not dummyVar is None:
            self.objective -= dummyVar

    def solve(self, solver = None):
        """
        Solve the given Lp problem.
        
        This function changes the problem to make it suitable for solving
        then calls the solver.actualSolve() method to find the solution
        
        Inputs:
            - solver -- Optional: the specific solver to be used, defaults to the 
              default solver.
                
        Side Effects:
            - The attributes of the problem object are changed in 
              solver.actualSolve() to reflect the Lp solution
        """
        
        if not(solver): solver = self.solver
        if not(solver): solver = LpSolverDefault
        wasNone, dummyVar = self.fixObjective()
        #time it
        self.solutionTime = -clock()
        status = solver.actualSolve(self)
        self.solutionTime += clock()
        self.restoreObjective(wasNone, dummyVar)
        self.solver = solver
        return status
    
    def resolve(self, solver = None):
        """ resolves an Problem using the same solver as previously
        """
        if not(solver): solver = self.solver
        if self.resolveOK:
            return self.solver.actualResolve(self)
        else:
            return self.solve(solver = solver)
        
    def setSolver(self,solver = LpSolverDefault):
        """Sets the Solver for this problem useful if you are using 
        resolve
        """
        self.solver = solver
    
    def setInitial(self,values):
        self.initialValues = values
    


class LpVariableDict(dict):
    """An LP variable generator"""
    def __init__(self, name, data = {}, lowBound = None, upBound = None, cat = 0):
        self.name = name
        dict.__init__(self, data)
        
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            self[key] = LpVariable(name % key, lowBound, upBound, cat)
            return self[key]

# Utility fonctions

def lpSum(vector):
    """
    Calculate the sum of a list of linear expressions
    
    Inputs:
         - vector -- A list of linear expressions
    """
    return LpAffineExpression().addInPlace(vector)

def lpDot(v1, v2):
    """Calculate the dot product of two lists of linear expressions"""
    if not isiterable(v1) and not isiterable(v2):
        return v1 * v2
    elif not isiterable(v1):
        return lpDot([v1]*len(v2),v2)
    elif not isiterable(v2):
        return lpDot(v1,[v2]*len(v1))
    else:
        return lpSum([lpDot(e1,e2) for e1,e2 in zip(v1,v2)])

def isNumber(x):
    """Returns true if x is an int of a float"""
    return type(x) in [int, float]

def value(x):
    """Returns the value of the variable/expression x, or x if it is a number"""
    if isNumber(x): return x
    else: return x.value()

def valueOrDefault(x):
    """Returns the value of the variable/expression x, or x if it is a number
    Variable wihout value (None) are affected a possible value (within their 
    bounds)."""
    if isNumber(x): return x
    else: return x.valueOrDefault()

def combination(orgset, k = None):
    """
    returns an iterator that lists the combinations of orgset of 
    length k
    
    @param orgset: the list to be iterated
    @param k: the cardinality of the subsets
    
    @return: an iterator of the subsets
    
    example:
    >>> c = combination([1,2,3,4],2)
    >>> for s in c:
    ...     print s
    [1, 2]
    [1, 3]
    [1, 4]
    [2, 3]
    [2, 4]
    [3, 4]
    """
    try:
        import probset
        return probset.Combination(orgset,k)
    except(ImportError):
        return __combination(orgset,k)

def __combination(orgset,k):
    """
    fall back if probset is not installed note it is GPL so cannot
    be included
    """
    if k == 1:
        for i in orgset:
            yield [i]
    elif k>1:
        for i,x in enumerate(orgset):
            #iterates though to near the end
            for s in __combination(orgset[i+1:],k-1):
                s.insert(0,x)
                yield s

def permutation(orgset, k = None):
    """
    returns an iterator that lists the permutations of orgset of 
    length k
    
    @param orgset: the list to be iterated
    @param k: the cardinality of the subsets
    
    @return: an iterator of the subsets
    
    example:
    >>> c = permutation([1,2,3,4],2)
    >>> for s in c:
    ...     print s
    [1, 2]
    [1, 3]
    [1, 4]
    [2, 1]
    [2, 3]
    [2, 4]
    [3, 1]
    [3, 2]
    [3, 4]
    [4, 1]
    [4, 2]
    [4, 3]
    """
    try:
        import probset
        return probset.Permutation(orgset, k)
    except(ImportError):
        return __permutation(orgset, k)
        
def __permutation(orgset, k):
    """
    fall back if probset is not installed note it is GPL so cannot
    be included
    """
    if k == 1:
        for i in orgset:
            yield [i]
    elif k>1:
        for i,x in enumerate(orgset):
            #iterates though to near the end
            for s in __permutation(orgset[:i] + orgset[i+1:],k-1):
                s.insert(0,x)
                yield s

def allpermutations(orgset,k):
    """
    returns all purmutations of orgset with up to k items
    @param orgset: the list to be iterated
    @param k: the maxcardinality of the subsets
    
    @return: an iterator of the subsets
    
    example:
    >>> c = allpermutations([1,2,3,4],2)
    >>> for s in c:
    ...     print s
    [1]
    [2]
    [3]
    [4]
    [1, 2]
    [1, 3]
    [1, 4]
    [2, 1]
    [2, 3]
    [2, 4]
    [3, 1]
    [3, 2]
    [3, 4]
    [4, 1]
    [4, 2]
    [4, 3]
    """
    return itertools.chain(*[permutation(orgset,i) for i in range(1,k+1)])

def allcombinations(orgset,k):
    """
    returns all purmutations of orgset with up to k items
    @param orgset: the list to be iterated
    @param k: the maxcardinality of the subsets
    
    @return: an iterator of the subsets
    
    example:
    >>> c = allcombinations([1,2,3,4],2)
    >>> for s in c:
    ...     print s
    [1]
    [2]
    [3]
    [4]
    [1, 2]
    [1, 3]
    [1, 4]
    [2, 3]
    [2, 4]
    [3, 4]
    """
    return itertools.chain(*[combination(orgset,i) for i in range(1,k+1)])

def makeDict(headers, array, default = None):
    """
    makes a list into a dictionary with the headings given in headings
    headers is a list of header lists
    array is a list with the data
    """
    result, defdict = __makeDict(headers, array, default)
    return result

def __makeDict(headers, array, default = None):
    #this is a recursive function so end the recursion as follows
    result ={}
    returndefaultvalue = None
    if len(headers) == 1:
        result.update(dict(zip(headers[0],array)))
        defaultvalue = default
    else:
        for i,h in enumerate(headers[0]):
            result[h],defaultvalue = __makeDict(headers[1:],array[i],default)          
    if default != None:
        f = lambda :defaultvalue
        defresult = collections.defaultdict(f)
        defresult.update(result)
        result = defresult
        returndefaultvalue = collections.defaultdict(f)
    return result, returndefaultvalue
        
def splitDict(Data):
    """
    Split a dictionary with lists as the data, into smaller dictionaries
    
    Inputs:
        - Data: A dictionary with lists as the values
        
    Returns:
        - A tuple of dictionaries each containing the data separately, 
          with the same dictionary keys
    """
    # find the maximum number of items in the dictionary
    maxitems = max([len(values) for values in Data.values()])
    output =[dict() for i in range(maxitems)]
    for key, values in Data.items():
        for i, val in enumerate(values):
            output[i][key] = val
             
    return tuple(output)

def configSolvers():
    """
    Configure the path the the solvers on the command line
    
    Designed to configure the file locations of the solvers from the 
    command line after installation
    """
    configlist = [(cplex_dll_path,"cplexpath","CPLEX: "),
                  (coinMP_path, "coinmppath","CoinMP dll (windows only): ")]
    print ("Please type the full path including filename and extension \n" +
           "for each solver available")
    configdict = {}
    for (default, key, msg) in configlist:
        value = raw_input(msg + "[" + str(default) +"]")
        if value:
            configdict[key] = value
    setConfigInformation(**configdict)
    
# Tests

def pulpTestCheck(prob, solver, okstatus, sol = {},
                   reducedcosts = None,
                   duals = None,
                   slacks = None,
                   eps = EPS):
    status = prob.solve(solver)
    if status not in okstatus:
        prob.writeLP("debug.lp")
        prob.writeMPS("debug.mps")
        print "Failure: status ==", status, "not in", okstatus
        raise "Tests failed for solver ", solver
    if sol:
        for v,x in sol.iteritems():
            if abs(v.varValue - x) > eps:
                prob.writeLP("debug.lp")
                prob.writeMPS("debug.mps")
                print "Test failed: var", v, "==", v.varValue, "!=", x
                raise "Tests failed for solver ", solver
    if reducedcosts:
        for v,dj in reducedcosts.iteritems():
            if abs(v.dj - dj) > eps:
                prob.writeLP("debug.lp")
                prob.writeMPS("debug.mps")
                print "Test failed: var.dj", v, "==", v.dj, "!=", dj
                raise "Tests failed for solver ", solver
    if duals:
        for cname,p in duals.iteritems():
            c = prob.constraints[cname]
            if abs(c.pi - p) > eps:
                prob.writeLP("debug.lp")
                prob.writeMPS("debug.mps")
                print "Test failed: constraint.pi", cname , "==", c.pi, "!=", p
                raise "Tests failed for solver ", solver
    if slacks:
        for cname,slack in slacks.iteritems():
            c = prob.constraints[cname]
            if abs(c.slack - slack) > eps:
                prob.writeLP("debug.lp")
                prob.writeMPS("debug.mps")
                print ("Test failed: constraint.slack", cname , "==",
                        c.slack, "!=", slack)
                raise "Tests failed for solver ", solver
        
def pulpTest1(solver):
    # Continuous
    prob = LpProblem("test1", LpMinimize)
    x = LpVariable("x", 0, 4)
    y = LpVariable("y", -1, 1)
    z = LpVariable("z", 0)
    w = LpVariable("w", 0)
    prob += x + 4*y + 9*z, "obj"
    prob += x+y <= 5, "c1"
    prob += x+z >= 10, "c2"
    prob += -y+z == 7, "c3"
    prob += w >= 0, "c4"
    print "\t Testing continuous LP solution"
    pulpTestCheck(prob, solver, [LpStatusOptimal], {x:4, y:-1, z:6, w:0})

def pulpTest2(solver):
    # MIP
    prob = LpProblem("test2", LpMinimize)
    x = LpVariable("x", 0, 4)
    y = LpVariable("y", -1, 1)
    z = LpVariable("z", 0, None, LpInteger)
    prob += x + 4*y + 9*z, "obj"
    prob += x+y <= 5, "c1"
    prob += x+z >= 10, "c2"
    prob += -y+z == 7.5, "c3"
    if solver.__class__ in [COIN_CMD]:
        # COIN_CMD always return LpStatusUndefined for MIP problems
        pulpTestCheck(prob, solver, [LpStatusUndefined], {x:3, y:-0.5, z:7})
    else:
        print "\t Testing MIP solution"
        pulpTestCheck(prob, solver, [LpStatusOptimal], {x:3, y:-0.5, z:7})

def pulpTest3(solver):
    # relaxed MIP
    prob = LpProblem("test3", LpMinimize)
    x = LpVariable("x", 0, 4)
    y = LpVariable("y", -1, 1)
    z = LpVariable("z", 0, None, LpInteger)
    prob += x + 4*y + 9*z, "obj"
    prob += x+y <= 5, "c1"
    prob += x+z >= 10, "c2"
    prob += -y+z == 7.5, "c3"
    solver.mip = 0
    print "\t Testing MIP relaxation"
    pulpTestCheck(prob, solver, [LpStatusOptimal], {x:3.5, y:-1, z:6.5})


def pulpTest4(solver):
    # Feasibility only
    prob = LpProblem("test4", LpMinimize)
    x = LpVariable("x", 0, 4)
    y = LpVariable("y", -1, 1)
    z = LpVariable("z", 0, None, LpInteger)
    prob += x+y <= 5, "c1"
    prob += x+z >= 10, "c2"
    prob += -y+z == 7.5, "c3"
    if solver.__class__ in [COIN_CMD]:
        # COIN_CMD always return LpStatusUndefined
        pulpTestCheck(prob, solver, [LpStatusUndefined])
        if x.varValue is None or x.varValue is None or x.varValue is None:
            raise "Tests failed for solver ", solver
    else:
        print "\t Testing feasibility problem (no objective)"
        pulpTestCheck(prob, solver, [LpStatusOptimal])


def pulpTest5(solver):
    # Infeasible
    prob = LpProblem("test5", LpMinimize)
    x = LpVariable("x", 0, 4)
    y = LpVariable("y", -1, 1)
    z = LpVariable("z", 0, 10)
    prob += x+y <= 5.2, "c1"
    prob += x+z >= 10.3, "c2"
    prob += -y+z == 17.5, "c3"
    if solver.__class__ is GLPK_CMD:
        # GLPK_CMD return codes are not enough informative
        pulpTestCheck(prob, solver, [LpStatusUndefined])
    elif solver.__class__ is CPLEX_MEM:
        # CPLEX_MEM returns InfeasibleOrUnbounded
        pulpTestCheck(prob, solver, [LpStatusInfeasible, LpStatusUndefined])
        print "\t Testing an infeasible problem"
    elif solver.__class__ is CPLEX_DLL:
        # CPLEX_DLL Does not solve the problem
        print "\t Testing an infeasible problem"
        pulpTestCheck(prob, solver, [LpStatusNotSolved])
    else:
        print "\t Testing an infeasible problem"
        pulpTestCheck(prob, solver, [LpStatusInfeasible])

def pulpTest6(solver):
    # Integer Infeasible
    prob = LpProblem("test6", LpMinimize)
    x = LpVariable("x", 0, 4, LpInteger)
    y = LpVariable("y", -1, 1, LpInteger)
    z = LpVariable("z", 0, 10, LpInteger)
    prob += x+y <= 5.2, "c1"
    prob += x+z >= 10.3, "c2"
    prob += -y+z == 7.4, "c3"
    if solver.__class__ in [GLPK_MEM, COIN_CMD]:
        # GLPK_CMD return codes are not enough informative
        # GLPK_MEM integer return codes seems wrong
        # COIN_CMD integer return code is always LpStatusUndefined
        pulpTestCheck(prob, solver, [LpStatusUndefined])
    elif solver.__class__ in [CPLEX_MEM, GLPK_CMD]:
        # CPLEX_MEM returns InfeasibleOrUnbounded
        print "\t Testing an integer infeasible problem"
        pulpTestCheck(prob, solver, [LpStatusInfeasible, LpStatusUndefined])
    elif solver.__class__ in [COINMP_DLL]:
        #Currently there is an error in COINMP for problems where
        #presolve elimiates too many variables
        print "\t Testing an integer infeasible problem (Error to be fixed)"
        pulpTestCheck(prob, solver, [LpStatusOptimal])
    else:
        print "\t Testing an integer infeasible problem"
        pulpTestCheck(prob, solver, [LpStatusInfeasible])

def pulpTest7(solver):
    #Column Based modelling of PulpTest1
    prob = LpProblem("test7", LpMinimize)
    obj = LpConstraintVar("obj")
    # constraints
    a = LpConstraintVar("C1", LpConstraintLE, 5)
    b = LpConstraintVar("C2", LpConstraintGE, 10)
    c = LpConstraintVar("C3", LpConstraintEQ, 7)

    prob.setObjective(obj)
    prob += a
    prob += b
    prob += c
    # Variables
    x = LpVariable("x", 0, 4, LpContinuous, obj + a + b)
    y = LpVariable("y", -1, 1, LpContinuous, 4*obj + a - c)
    z = LpVariable("z", 0, None, LpContinuous, 9*obj + b + c)
    print "\t Testing column based modelling"
    pulpTestCheck(prob, solver, [LpStatusOptimal], {x:4, y:-1, z:6})

def pulpTest75(solver):
    #Column Based modelling of PulpTest1 with empty constraints
    prob = LpProblem("test7", LpMinimize)
    obj = LpConstraintVar("obj")
    # constraints
    a = LpConstraintVar("C1", LpConstraintLE, 5)
    b = LpConstraintVar("C2", LpConstraintGE, 10)
    c = LpConstraintVar("C3", LpConstraintEQ, 7)

    prob.setObjective(obj)
    prob += a
    prob += b
    prob += c
    # Variables
    x = LpVariable("x", 0, 4, LpContinuous, obj + b)
    y = LpVariable("y", -1, 1, LpContinuous, 4*obj - c)
    z = LpVariable("z", 0, None, LpContinuous, 9*obj + b + c)
    if solver.__class__ in [CPLEX_DLL, CPLEX_CMD, COINMP_DLL]:
        print "\t Testing column based modelling with empty constraints"
        pulpTestCheck(prob, solver, [LpStatusOptimal], {x:4, y:-1, z:6})

def pulpTest8(solver):
    """
    Test the reporting of dual variables slacks and reduced costs
    """
    prob = LpProblem("test8", LpMinimize)
    x = LpVariable("x", 0, 4)
    y = LpVariable("y", -1, 1)
    z = LpVariable("z", 0)
    c1 = x+y <= 5
    c2 = x+z >= 10
    c3 = -y+z == 7
    
    prob += x + 4*y + 9*z, "obj"
    prob += c1, "c1"
    prob += c2,"c2"
    prob += c3,"c3"
    
    if solver.__class__ in [CPLEX_DLL, CPLEX_CMD, COINMP_DLL]:
        print "\t Testing dual variables and slacks reporting"
        pulpTestCheck(prob, solver, [LpStatusOptimal],
                  sol = {x:4, y:-1, z:6}, 
                  reducedcosts = {x:0, y:12, z:0},
                  duals = {"c1":0, "c2":1, "c3":8}, 
                  slacks = {"c1":2, "c2":0, "c3":0})

def pulpTest9(solver):
    #Column Based modelling of PulpTest1 with a resolve
    prob = LpProblem("test9", LpMinimize)
    obj = LpConstraintVar("obj")
    # constraints
    a = LpConstraintVar("C1", LpConstraintLE, 5)
    b = LpConstraintVar("C2", LpConstraintGE, 10)
    c = LpConstraintVar("C3", LpConstraintEQ, 7)

    prob.setObjective(obj)
    prob += a
    prob += b
    prob += c
    
    prob.setSolver(solver)# Variables
    x = LpVariable("x", 0, 4, LpContinuous, obj + a + b)
    y = LpVariable("y", -1, 1, LpContinuous, 4*obj + a - c)
    prob.resolve()
    z = LpVariable("z", 0, None, LpContinuous, 9*obj + b + c)
    print "\t Testing resolve of problem"
    prob.resolve()
    #difficult to check this is doing what we want as the resolve is 
    #over ridden if it is not implemented
    #pulpTestCheck(prob, solver, [LpStatusOptimal], {x:4, y:-1, z:6})

def pulpTestSolver(solver):
    tests = [pulpTest1, pulpTest2, pulpTest3, pulpTest4,
        pulpTest5, pulpTest6, pulpTest7, pulpTest75,
        pulpTest8, pulpTest9]
    for t in tests:
        t(solver(msg=0))

def pulpTestAll():
    solvers = [CPLEX_MEM, CPLEX_DLL, CPLEX_CMD,
        COIN_MEM, COIN_CMD, COINMP_DLL,
        GLPK_MEM, GLPK_CMD,
        XPRESS]

    for s in solvers:
        if s().available():
            try:
                pulpTestSolver(s)
                print "* Solver", s, "passed."
            except:
                print "* Solver", s, "failed."
        else:
            print "Solver", s, "unavailable."


if __name__ == '__main__':
    # Tests
    pulpTestAll()
    import doctest
    doctest.testmod()
