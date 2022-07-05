# PuLP : Python LP Modeler
# Version 1.20

# Copyright (c) 2002-2005, Jean-Sebastien Roy (js@jeannot.org)
# Modifications Copyright (c) 2007- Stuart Anthony Mitchell (s.mitchell@auckland.ac.nz)
# $Id:solvers.py 1791 2008-04-23 22:54:34Z smit023 $

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
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

"""
This file contains the solver classes for PuLP
Note that the solvers that require a compiled extension may not work in
the current version
"""

import os
import sys
from time import clock
import ConfigParser
import sparse
import collections
from tempfile import mktemp
from constants import *

#import configuration information
def initialize(file = None):
    """ reads the configuration file to initialise the module"""
    if not file:
        file = __file__
    config = ConfigParser.SafeConfigParser()
    config.read(file)
    cplex_dll_path = config.get("locations","CplexPath")
    coinMP_path = config.get("locations","CoinMPPath").split()
    for i,path in enumerate(coinMP_path):
        if not os.path.dirname(path):
            #if no pathname is supplied assume the file is in the same directory
            coinMP_path[i] = os.path.join(os.path.dirname(config_filename),path) 
    return cplex_dll_path, coinMP_path

if __name__ != '__main__':
    config_filename = os.path.join(os.path.dirname(__file__),
                                   "pulp.cfg")
else: #run as a script
    from pulp import __file__ as fname
    config_filename = os.path.join(os.path.dirname(fname),
                                   "pulp.cfg")
cplex_dll_path,coinMP_path = initialize(config_filename)

# See later for LpSolverDefault definition
class LpSolver:
    """A generic LP Solver"""

    def __init__(self, mip = 1, msg = 1, options = []):
        self.mip = mip
        self.msg = msg
        self.options = options

    def available(self):
        """True if the solver is available"""
        raise NotImplementedError

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        raise NotImplementedError

    def actualResolve(self,lp):
        """
        uses existing problem information and solves the problem
        If it is not implelemented in the solver
        just solve again
        """
        self.actualSolve(lp)

    def copy(self):
        """Make a copy of self"""
        
        aCopy = self.__class__()
        aCopy.mip = self.mip
        aCopy.msg = self.msg
        aCopy.options = self.options
        return aCopy

    def solve(self, lp):
        """Solve the problem lp"""
        # Always go through the solve method of LpProblem
        return lp.solve(self)
    
    #TODO: Not sure if this code should be here or in a child class
    def getCplexStyleArrays(self,lp,
                       senseDict={LpConstraintEQ:"E", LpConstraintLE:"L", LpConstraintGE:"G"},
                       LpVarCategories = {LpContinuous: "C",LpInteger: "I"},
                       LpObjSenses = {LpMaximize : -1,
                                      LpMinimize : 1},
                       infBound =  1e20 
                       ):
        """returns the arrays suitable to pass to a cdll Cplex
        or other solvers that are similar

        Copyright (c) Stuart Mitchell 2007
        """
        rangeCount = 0
        variables=lp.variables()
        numVars = len(variables)
        #associate each variable with a ordinal
        self.v2n=dict(((variables[i],i) for i in range(numVars)))
        self.vname2n=dict(((variables[i].name,i) for i in range(numVars)))            
        self.n2v=dict((i,variables[i]) for i in range(numVars))
        #objective values
        objSense = LpObjSenses[lp.sense]
        NumVarDoubleArray = ctypes.c_double * numVars
        objectCoeffs=NumVarDoubleArray()
        #print "Get objective Values"
        for v,val in lp.objective.iteritems():
            objectCoeffs[self.v2n[v]]=val
        #values for variables
        objectConst = ctypes.c_double(0.0)
        NumVarStrArray = ctypes.c_char_p * numVars
        colNames = NumVarStrArray()
        lowerBounds = NumVarDoubleArray()
        upperBounds = NumVarDoubleArray()
        initValues = NumVarDoubleArray()
        if self.debug: print "Get Variable information"
        for v in lp.variables():
            colNames[self.v2n[v]] = str(v.name)
            initValues[self.v2n[v]] = 0.0
            if v.lowBound != None:
                lowerBounds[self.v2n[v]] = v.lowBound
            else:
                lowerBounds[self.v2n[v]] = -infBound
            if v.upBound != None:
                upperBounds[self.v2n[v]] = v.upBound
            else:
                upperBounds[self.v2n[v]] = infBound
        #values for constraints
        numRows =len(lp.constraints)
        NumRowDoubleArray = ctypes.c_double * numRows
        NumRowStrArray = ctypes.c_char_p * numRows
        NumRowCharArray = ctypes.c_char * numRows
        rhsValues = NumRowDoubleArray()
        rangeValues = NumRowDoubleArray()
        rowNames = NumRowStrArray()
        rowType = NumRowCharArray()
        self.c2n = {}
        self.n2c = {}
        i = 0
        if self.debug: print "Get constraint information"
        for c in lp.constraints:
            rhsValues[i] = -lp.constraints[c].constant
            #for ranged constraints a<= constraint >=b
            rangeValues[i] = 0.0
            rowNames[i] = str(c)
            rowType[i] = senseDict[lp.constraints[c].sense]
            self.c2n[c] = i
            self.n2c[i] = c
            i = i+1
        #return the coefficient matrix as a series of vectors
        coeffs = lp.coefficients()
        sparseMatrix = sparse.Matrix(range(numRows), range(numVars))
        for var,row,coeff in coeffs:
            sparseMatrix.add(self.c2n[row], self.vname2n[var], coeff)
        (numels, mystartsBase, mylenBase, myindBase, 
         myelemBase) = sparseMatrix.col_based_arrays() 
        elemBase = ctypesArrayFill(myelemBase, ctypes.c_double)
        indBase = ctypesArrayFill(myindBase, ctypes.c_int)
        startsBase = ctypesArrayFill(mystartsBase, ctypes.c_int)
        lenBase = ctypesArrayFill(mylenBase, ctypes.c_int)
        #MIP Variables
        NumVarCharArray = ctypes.c_char * numVars
        columnType = NumVarCharArray()
        if lp.isMIP():
            for v in lp.variables():
                columnType[self.v2n[v]] = LpVarCategories[v.cat]
        self.addedVars = numVars
        self.addedRows = numRows
        return  (numVars, numRows, numels, rangeCount, 
            objSense, objectCoeffs, objectConst, 
            rhsValues, rangeValues, rowType, startsBase, lenBase, indBase, 
            elemBase, lowerBounds, upperBounds, initValues, colNames, 
            rowNames, columnType, self.n2v, self.n2c)


class LpSolver_CMD(LpSolver):
    """A generic command line LP Solver"""
    def __init__(self, path = None, keepFiles = 0, mip = 1, msg = 1, options = []):
        LpSolver.__init__(self, mip, msg, options)
        if path is None:
            self.path = self.defaultPath()
        else:
            self.path = path
        self.keepFiles = keepFiles
        self.setTmpDir()

    def copy(self):
        """Make a copy of self"""
        
        aCopy = LpSolver.copy(self)
        aCopy.path = self.path
        aCopy.keepFiles = self.keepFiles
        aCopy.tmpDir = self.tmpDir
        return aCopy

    def setTmpDir(self):
        """Set the tmpDir attribute to a reasonnable location for a temporary
        directory"""
        if os.name != 'nt':
            # On unix use /tmp by default
            self.tmpDir = os.environ.get("TMPDIR", "/tmp")
            self.tmpDir = os.environ.get("TMP", self.tmpDir)
        else:
            # On Windows use the current directory
            self.tmpDir = os.environ.get("TMPDIR", "")
            self.tmpDir = os.environ.get("TMP", self.tmpDir)
            self.tmpDir = os.environ.get("TEMP", self.tmpDir)
        if not os.path.isdir(self.tmpDir):
            self.tmpDir = ""
        elif not os.access(self.tmpDir, os.F_OK + os.W_OK):
            self.tmpDir = ""

    def defaultPath(self):
        raise NotImplementedError

    def executableExtension(name):
        if os.name != 'nt':
            return name
        else:
            return name+".exe"
    executableExtension = staticmethod(executableExtension)

    def executable(command):
        """Checks that the solver command is executable,
        And returns the actual path to it."""

        if os.path.isabs(command):
            if os.access(command, os.X_OK):
                return command
        for path in os.environ.get("PATH", []).split(os.pathsep):
            if os.access(os.path.join(path, command), os.X_OK):
                return os.path.join(path, command)
        return False
    executable = staticmethod(executable)

class GLPK_CMD(LpSolver_CMD):
    """The GLPK LP solver"""
    def defaultPath(self):
        return self.executableExtension("glpsol")

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise Exception("PuLP: cannot execute "+self.path)
        if not self.keepFiles:
            pid = os.getpid()
            tmpLp = os.path.join(self.tmpDir, "%d-pulp.lp" % pid)
            tmpSol = os.path.join(self.tmpDir, "%d-pulp.sol" % pid)
        else:
            tmpLp = lp.name+"-pulp.lp"
            tmpSol = lp.name+"-pulp.sol"
        lp.writeLP(tmpLp, writeSOS = 0)
        # proc = ["glpsol", "--lpt", tmpLp, "-o", tmpSol]
        proc = ["glpsol", "--cpxlp", tmpLp, "-o", tmpSol]
        if not self.mip: proc.append('--nomip')
        proc.extend(self.options)
        print "GLPSOL options:", proc
        
        ######### pascal: create some new attributes to be used in ilp.py
        self.process = proc
        self.lpfile = tmpLp 
        self.solfile = tmpSol 
        ########## end of pascal
        
        self.solution_time = clock()
        if not self.msg:
            proc[0] = self.path
            f = os.popen(" ".join(proc))
            f.read()
            rc = f.close()
            if rc != None:
                raise Exception("PuLP: Error while trying to execute "+self.path)
        else:
            if os.name != 'nt':
                #rc = os.spawnv(os.P_WAIT, self.executable(self.path), proc)
                rc = os.spawnvp(os.P_WAIT, self.path, proc)
            else:
                #rc = os.spawnv(os.P_WAIT, self.executable(self.path), proc)
                rc = os.spawnv(os.P_WAIT, proc)
            if rc == 127:
                raise Exception("PuLP: Error while trying to execute "+self.path)
        self.solution_time += clock()

        if not os.path.exists(tmpSol):
            raise Exception("PuLP: Error while executing %s. No solution file" %self.path)
        lp.status, values = self.readsol(tmpSol)
        lp.assignVarsVals(values)
        if not self.keepFiles:
            try: os.remove(tmpLp)
            except: pass
            try: os.remove(tmpSol)
            except: pass
        return lp.status

    def readsol(self,filename):
        """Read a GLPK solution file"""
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

try:
    import pulpGLPK
    
    class GLPK_MEM(LpSolver):
        """The GLPK LP solver (via a module)"""
        def __init__(self, mip = 1, msg = 1, presolve = 1):
            LpSolver.__init__(self, mip, msg)
            self.presolve = presolve

        def copy(self):
            """Make a copy of self"""
        
            aCopy = LpSolver.copy()
            aCopy.presolve = self.presolve
            return aCopy

        def available(self):
            """True if the solver is available"""
            return True

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            lp.status = pulpGLPK.solve(lp.objective, lp.constraints, lp.sense, self.msg,
                self.mip, self.presolve)
            return lp.status
    
    GLPK = GLPK_MEM
except ImportError:
    class GLPK_MEM(LpSolver):
        """The GLPK LP solver (via a module)"""
        def available(self):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise RuntimeError, "GLPK_MEM: Not Available"

    GLPK = GLPK_CMD

class CPLEX_CMD(LpSolver_CMD):
    """The CPLEX LP solver"""
    def defaultPath(self):
        return self.executableExtension("cplex")

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise Exception("PuLP: cannot execute "+self.path)
        if not self.keepFiles:
            pid = os.getpid()
            tmpLp = os.path.join(self.tmpDir, "%d-pulp.lp" % pid)
            # Should probably use another CPLEX solution format
            tmpSol = os.path.join(self.tmpDir, "%d-pulp.txt" % pid)
        else:
            tmpLp = lp.name+"-pulp.lp"
            # Should probably use another CPLEX solution format
            tmpSol = lp.name+"-pulp.txt"
        lp.writeLP(tmpLp, writeSOS = 1)
        try: os.remove(tmpSol)
        except: pass
        if not self.msg:
            #TODO make this work properly in windows
            #cplex = os.popen(self.path+" > /dev/null 2> /dev/null", "w")
            cplex = os.popen(self.path, "w")
        else:
            cplex = os.popen(self.path, "w")
        cplex.write("read "+tmpLp+"\n")
        for option in self.options:
            cplex.write(option+"\n")
        if lp.isMIP():
            if self.mip:
                cplex.write("mipopt\n")
                cplex.write("change problem fixed\n")
            else:
                cplex.write("change problem lp\n")
                
        cplex.write("optimize\n")
        cplex.write("write "+tmpSol+"\n")
        cplex.write("quit\n")
        if cplex.close() != None:
            #TODO: Will need to fix the solution file reading of Cplex
            raise "PuLP: Error while trying to execute "+self.path
        if not self.keepFiles:
            try: os.remove(tmpLp)
            except: pass
        if not os.path.exists(tmpSol):
            status = LpStatusInfeasible
        else:
            status, values, reducedCosts, shadowPrices, slacks = self.readsol(tmpSol)
        if not self.keepFiles:
            try: os.remove(tmpSol)
            except: pass
            try: os.remove("cplex.log")
            except: pass
        if status != LpStatusInfeasible:
            lp.assignVarsVals(values)
            lp.assignVarsDj(reducedCosts)                          
            lp.assignConsPi(shadowPrices)
            lp.assignConsSlack(slacks)          
        lp.status = status
        return status

    def readsol(self,filename):
        """Read a CPLEX solution file"""
        f = file(filename)
        for i in range(3): f.readline()
        statusString = f.readline()[18:30]
        cplexStatus = {
            "OPTIMAL SOLN":LpStatusOptimal,
            }
        if statusString not in cplexStatus:
            raise ValueError, "Unknown status returned by CPLEX: "+statusString
        status = cplexStatus[statusString]
        
        for i in range(13): f.readline()
        shadowPrices = {}
        slacks = {}
        shadowPrices = {}
        slacks = {}
        while 1:
            l = f.readline()
            if l[:10] == " SECTION 2": break
            line = l[3:].split()
            if len(line):
                name = line[1]
                shadowPrice = line[7]
                slack = line[4]
                #TODO investigate why this is negative
                shadowPrices[name] = -float(shadowPrice)
                slacks[name] = slack
        
        for i in  range(3): f.readline()
        values = {}
        reducedCosts = {}
        while 1:
            l = f.readline()
            if l == "": break
            line = l[3:].split()
            if len(line):
                name = line[1]
                value = float(line[3])
                reducedCost = float(line[7])
                values[name] = value
                reducedCosts[name] = reducedCost

        return status, values, reducedCosts, shadowPrices, slacks

try:
    import pulpCPLEX
    
    class CPLEX_MEM(LpSolver):
        """The CPLEX LP solver (via a module)"""
        def __init__(self, mip = 1, msg = 1, timeLimit = -1):
            LpSolver.__init__(self, mip, msg)
            self.timeLimit = timeLimit

        def available(self):
            """True if the solver is available"""
            return True

        def grabLicence(self):
            """Returns True if a CPLEX licence can be obtained.
            The licence is kept until releaseLicence() is called."""
            return pulpCPLEX.grabLicence()

        def releaseLicence(self):
            """Release a previously obtained CPLEX licence"""
            pulpCPLEX.releaseLicence()

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            lp.status = pulpCPLEX.solve(lp.objective, lp.constraints, lp.sense, self.msg,
                self.mip, self.timeLimit)
            return lp.status
    
    CPLEX = CPLEX_MEM
except ImportError:
    class CPLEX_MEM(LpSolver):
        """The CPLEX LP solver (via a module)"""
        def available(self):
            """True if the solver is available"""
            return False
        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise RuntimeError, "CPLEX_MEM: Not Available"

    CPLEX = CPLEX_CMD
try:
    import ctypes
    if os.name in ['nt','dos']:
        CplexDll = ctypes.windll.LoadLibrary(cplex_dll_path)
    else:
        CplexDll = ctypes.cdll.LoadLibrary(cplex_dll_path)
        
    class CPLEX_DLL(LpSolver):
        """The CPLEX LP/MIP solver (via a Dynamic library DLL - windows or SO - Linux)"""
        EPGAP = 2009
        def __init__(self, mip = 1, msg = 1, timeLimit = -1,
                       epgap = None, logfile = None, debug = False):
            LpSolver.__init__(self, mip, msg)
            self.timeLimit = timeLimit
            self.debug = debug
            self.grabLicence()
            if epgap is not None:
                self.changeEpgap(epgap)
            if logfile is not None:
                self.setlogfile(logfile)
            else:
                self.logfile = None

        def setlogfile(self, filename):
            """
            sets the logfile for cplex output
            """
            CplexDll.CPXfopen.argtypes = [ctypes.c_char_p, 
                                          ctypes.c_char_p]
            CplexDll.CPXfopen.restype = ctypes.c_void_p
            self.logfilep = CplexDll.CPXfopen(filename, "w")
            CplexDll.CPXsetlogfile.argtypes = [ctypes.c_void_p, 
                                          ctypes.c_void_p] 
            CplexDll.CPXsetlogfile(self.env, self.logfilep)
        
        def changeEpgap(self, epgap = 10**-4):
            """
            Change cplex solver integer bound gap tolerence
            """
            CplexDll.CPXsetdblparam.argtypes = [ctypes.c_void_p, ctypes.c_int, 
                                                ctypes.c_double]
            CplexDll.CPXsetdblparam(self.env,CPLEX_DLL.EPGAP,epgap)

        def findSolutionValues(self, lp, numcols, numrows):
            byref = ctypes.byref
            solutionStatus = ctypes.c_int()
            objectiveValue = ctypes.c_double()
            x = (ctypes.c_double * numcols)()
            pi = (ctypes.c_double * numrows)()
            slack = (ctypes.c_double * numrows)()
            dj = (ctypes.c_double * numcols)()
            status= CplexDll.CPXsolwrite(self.env, self.hprob, "CplexTest.sol")
            if lp.isMIP():
                solutionStatus.value = CplexDll.CPXgetstat(self.env, self.hprob)
                status = CplexDll.CPXgetobjval(self.env, self.hprob, byref(objectiveValue))
                if status != 0 and status != 1217: #no solution exists
                    raise RuntimeError, ("Error in CPXgetobjval status=" + str(status))
                
                status = CplexDll.CPXgetx(self.env, self.hprob, byref(x), 0, numcols - 1)
                if status != 0 and status != 1217:
                    raise RuntimeError, "Error in CPXgetx status=" + str(status)
                
            else:
                status = CplexDll.CPXsolution(self.env, self.hprob, byref(solutionStatus), 
                                              byref(objectiveValue), byref(x), byref(pi), 
                                              byref(slack), byref(dj))
            CplexLpStatus = collections.defaultdict(lambda : LpStatusUndefined)
            CplexLpStatus.update({1: LpStatusOptimal, 3: LpStatusInfeasible, 
                                  2: LpStatusUnbounded, 0: LpStatusNotSolved, 
                                  101: LpStatusOptimal, 103: LpStatusInfeasible})
            #populate pulp solution values
            variablevalues = {}
            variabledjvalues = {}
            constraintpivalues = {}
            constraintslackvalues = {}
            for i in range(numcols):
                variablevalues[self.n2v[i].name] = x[i]
                variabledjvalues[self.n2v[i].name] = dj[i]
            lp.assignVarsVals(variablevalues)
            lp.assignVarsDj(variabledjvalues)            
            #put pi and slack variables against the constraints
            for i in range(numrows):
                constraintpivalues[self.n2c[i]] = pi[i]
                constraintslackvalues[self.n2c[i]] = slack[i]                
            lp.assignConsPi(constraintpivalues)
            lp.assignConsSlack(constraintslackvalues)
            
            #TODO: clear up the name of self.n2c
            
            if self.msg:
                print "Cplex status=", solutionStatus.value
            lp.resolveOK = True
            for var in lp.variables():
                var.isModified = False
            lp.status = CplexLpStatus[solutionStatus.value]
            return lp.status


        def __del__(self):
            #LpSolver.__del__(self)
            self.releaseLicence()


        def available(self):
            """True if the solver is available"""
            return True

        def grabLicence(self):
            """Returns True if a CPLEX licence can be obtained.
            The licence is kept until releaseLicence() is called."""
            status = ctypes.c_int()
            self.env = CplexDll.CPXopenCPLEX(ctypes.byref(status))
            if not(status.value == 0):
                raise RuntimeError, "CPLEX library failed on CPXopenCPLEX status=" + str(status)
            

        def releaseLicence(self):
            """Release a previously obtained CPLEX licence"""
            if getattr(self,"env",False):
                status=CplexDll.CPXcloseCPLEX(self.env)
            else:
                raise RuntimeError, "No CPLEX enviroment to close"

        def callSolver(self, isMIP):
            """Solves the problem with cplex
            """
            #output an mps file for debug
            SOLV_FILE_MPS=1
            if self.debug: 
                status = CplexDll.CPXwriteprob (self.env, self.hprob, "CplexTest.lp", None);
                if status != 0:
                    raise RuntimeError, "Error in CPXwriteprob status=" + str(status) 
            #solve the problem
            if self.debug:
                print "About to solve"
            self.cplexTime = -clock()
            if isMIP and self.mip:
                status= CplexDll.CPXmipopt(self.env, self.hprob)
                if status != 0:
                    raise RuntimeError, "Error in CPXmipopt status=" + str(status)
            else:
                status = CplexDll.CPXlpopt(self.env, self.hprob)
                if status != 0:
                    raise RuntimeError, "Error in CPXlpopt status=" + str(status)
            self.cplexTime += clock()
            if self.debug:
                print "finished solving"


        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            #TODO alter so that msg parameter is handled correctly
            status = ctypes.c_int()
            byref = ctypes.byref   #shortcut to function
            self.hprob = CplexDll.CPXcreateprob(self.env, byref(status), lp.name)
            if status.value != 0:
                raise RuntimeError, "Error in CPXcreateprob status=" + str(status) 
            if self.debug: print "Before getCplexDllArrays"
            (numcols, numrows, numels, rangeCount, 
                objSense, obj, objconst, 
                rhs, rangeValues, rowSense, matbeg, matcnt, matind, 
                matval, lb, ub, initValues, colname, 
                rowname, xctype, n2v, n2c )= self.getCplexStyleArrays(lp)
            status.value = CplexDll.CPXcopylpwnames (self.env, self.hprob, numcols, numrows,
                                 objSense, obj, rhs, rowSense, matbeg, matcnt,
                                 matind, matval, lb, ub, None, colname, rowname);
            if status.value != 0:
                raise RuntimeError, "Error in CPXcopylpwnames status=" + str(status) 
            if lp.isMIP() and self.mip:
                status.value = CplexDll.CPXcopyctype(self.env, self.hprob, xctype)
            if status.value != 0:
                raise RuntimeError, "Error in CPXcopyctype status=" + str(status) 
            #set the initial solution            
            self.callSolver(lp.isMIP())
            #get the solution information
            solutionStatus = self.findSolutionValues(lp, numcols, numrows)          
            for var in lp.variables():
                var.modified = False
            return solutionStatus

        
        def actualResolve(self,lp):
            """looks at which variables have been modified and changes them
            """
            #TODO: Add changing variables not just adding them
            #TODO: look at constraints
            modifiedVars = [var for var in lp.variables() if var.modified] 
            #assumes that all variables flagged as modified need to be added to the 
            #problem
            newVars = modifiedVars 
            #print newVars
            self.v2n.update([(var, i+self.addedVars) for i,var in enumerate(newVars)])
            self.n2v.update([(i+self.addedVars, var) for i,var in enumerate(newVars)])
            self.vname2n.update([(var.name, i+self.addedVars) for i,var in enumerate(newVars)])
            oldVars = self.addedVars
            self.addedVars += len(newVars)
            (ccnt,nzcnt,obj,cmatbeg, cmatlen, cmatind,cmatval,lb,ub, initvals,
             colname, coltype) = self.getSparseCols(newVars, lp, oldVars, defBound = 1e20)
            CPXaddcolsStatus = CplexDll.CPXaddcols(self.env, self.hprob,
                                          ccnt, nzcnt,
                                          obj,cmatbeg, 
                                          cmatind,cmatval, 
                                          lb,ub,colname)
#            columnTypestatus = CplexDll.CPXaddcols(self.env, self.hprob,
#                                          ccnt, nzcnt,
#                                          obj,cmatbeg, 
#                                          cmatind,cmatval, 
#                                          lb,ub,colname)
            #add the column types
            if lp.isMIP() and self.mip:
                indices = (ctypes.c_int * len(newVars))()
                for i,var in enumerate(newVars):
                    indices[i] = oldVars +i
                CPXchgctypeStatus = CplexDll.CPXchgctype (self.env, self.hprob,
                                                 ccnt, indices, coltype);
            #solve the problem
            self.callSolver(lp.isMIP())
            #get the solution information
            solutionStatus = self.findSolutionValues(lp, self.addedVars, self.addedRows)          
           
            for var in modifiedVars:
                var.modified = False
            return solutionStatus
                
        def getSparseCols(self, vars, lp, offset = 0, defBound = 1e20):
            """outputs the variables in var as a sparse matrix,
            suitable for cplex and Coin

            Copyright (c) Stuart Mitchell 2007
            """
            numVars = len(vars)
            obj = (ctypes.c_double * numVars)()
            cmatbeg = (ctypes.c_int * numVars)()
            mycmatind = []
            mycmatval = []
            rangeCount = 0
            #values for variables
            colNames =  (ctypes.c_char_p * numVars)()
            lowerBounds =  (ctypes.c_double * numVars)()
            upperBounds =  (ctypes.c_double * numVars)()
            initValues =  (ctypes.c_double * numVars)()
            if self.debug: print "Get Variable information"
            i=0
            for v in vars:
                colNames[i] = str(v.name)
                initValues[i] = v.init
                if v.lowBound != None:
                    lowerBounds[i] = v.lowBound
                else:
                    lowerBounds[i] = -defBound
                if v.upBound != None:
                    upperBounds[i] = v.upBound
                else:
                    upperBounds[i] = defBound
                i+= 1
                #create the new variables    
            #values for constraints
            #return the coefficient matrix as a series of vectors
            #return the coefficient matrix as a series of vectors
            myobjectCoeffs = {}
            numRows = len(lp.constraints)
            sparseMatrix = sparse.Matrix(range(numRows), range(numVars))
            for var in vars:
                for row,coeff in var.expression.iteritems():
                   if row.name == lp.objective.name:
                        myobjectCoeffs[var] = coeff
                   else:
                        sparseMatrix.add(self.c2n[row.name], self.v2n[var] - offset, coeff)
            #objective values
            objectCoeffs = (ctypes.c_double * numVars)()
            for var in vars:
                objectCoeffs[self.v2n[var]-offset] = myobjectCoeffs[var]
            (numels, mystartsBase, mylenBase, myindBase, 
             myelemBase) = sparseMatrix.col_based_arrays() 
            elemBase = ctypesArrayFill(myelemBase, ctypes.c_double)
            indBase = ctypesArrayFill(myindBase, ctypes.c_int)
            startsBase = ctypesArrayFill(mystartsBase, ctypes.c_int)
            lenBase = ctypesArrayFill(mylenBase, ctypes.c_int)
            #MIP Variables
            NumVarCharArray = ctypes.c_char * numVars
            columnType = NumVarCharArray()
            if lp.isMIP():
                CplexLpCategories = {LpContinuous: "C",
                                    LpInteger: "I"}
                for v in vars:
                    columnType[self.v2n[v] - offset] = CplexLpCategories[v.cat]
            return  numVars, numels,  objectCoeffs, \
                startsBase, lenBase, indBase, \
                elemBase, lowerBounds, upperBounds, initValues, colNames, \
                columnType
             
            
        
    CPLEX = CPLEX_DLL
except (ImportError,OSError):
    class CPLEX_DLL(LpSolver):
        """The CPLEX LP/MIP solver (via a Dynamic library DLL - windows or SO - Linux)"""
        def available(self):
            """True if the solver is available"""
            return False
        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise RuntimeError, "CPLEX_DLL: Not Available"



class XPRESS(LpSolver_CMD):
    """The XPRESS LP solver"""
    def defaultPath(self):
        return self.executableExtension("optimizer")

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise Exception("PuLP: cannot execute "+self.path)
        if not self.keepFiles:
            pid = os.getpid()
            tmpLp = os.path.join(self.tmpDir, "%d-pulp.lp" % pid)
            tmpSol = os.path.join(self.tmpDir, "%d-pulp.prt" % pid)
        else:
            tmpLp = lp.name+"-pulp.lp"
            tmpSol = lp.name+"-pulp.prt"
        lp.writeLP(tmpLp, writeSOS = 1, mip = self.mip)
        if not self.msg:
            xpress = os.popen(self.path+" "+lp.name+" > /dev/null 2> /dev/null", "w")
        else:
            xpress = os.popen(self.path+" "+lp.name, "w")
        xpress.write("READPROB "+tmpLp+"\n")
        if lp.sense == LpMaximize:
            xpress.write("MAXIM\n")
        else:
            xpress.write("MINIM\n")
        if lp.isMIP() and self.mip:
            xpress.write("GLOBAL\n")
        xpress.write("WRITEPRTSOL "+tmpSol+"\n")
        xpress.write("QUIT\n")
        if xpress.close() != None:
            raise Exception("PuLP: Error while executing "+self.path)
        status, values = self.readsol(tmpSol)
        if not self.keepFiles:
            try: os.remove(tmpLp)
            except: pass
            try: os.remove(tmpSol)
            except: pass
        lp.status = status
        lp.assignVarsVals(values)
        if abs(lp.infeasibilityGap(self.mip)) > 1e-5: # Arbitrary
            lp.status = LpStatusInfeasible
        return lp.status

    def readsol(self,filename):
        """Read an XPRESS solution file"""
        f = file(filename)
        for i in range(6): f.readline()
        l = f.readline().split()

        rows = int(l[2])
        cols = int(l[5])
        for i in range(3): f.readline()
        statusString = f.readline().split()[0]
        xpressStatus = {
            "Optimal":LpStatusOptimal,
            }
        if statusString not in xpressStatus:
            raise ValueError, "Unknow status returned by XPRESS: "+statusString
        status = xpressStatus[statusString]
        values = {}
        while 1:
            l = f.readline()
            if l == "": break
            line = l.split()
            if len(line) and line[0] == 'C':
                name = line[2]
                value = float(line[4])
                values[name] = value
        return status, values

class COIN_CMD(LpSolver_CMD):
    """The COIN CLP/CBC LP solver"""
    def defaultPath(self):
        return (self.executableExtension("clp"), self.executableExtension("cbc"))

    def __init__(self, path = None, keepFiles = 0, mip = 1,
            msg = 1, cuts = 1, presolve = 1, dual = 1, strong = 5, options = []):
        """Here, path is a tuple containing the path to clp and cbc"""
        LpSolver_CMD.__init__(self, path, keepFiles, mip, msg, options)
        self.cuts = cuts
        self.presolve = presolve
        self.dual = dual
        self.strong = strong

    def copy(self):
        """Make a copy of self"""
        
        aCopy = LpSolver_CMD.copy(self)
        aCopy.cuts = self.cuts
        aCopy.presolve = self.presolve
        aCopy.dual = self.dual
        aCopy.strong = self.strong
        return aCopy

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if lp.isMIP() and self.mip: return self.solve_CBC(lp)
        else: return self.solve_CLP(lp)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path[0]) and \
            self.executable(self.path[1])

    def solve_CBC(self, lp):
        """Solve a MIP problem using CBC"""
        if not self.executable(self.path[1]):
            raise Exception("PuLP: cannot execute "+self.path[1])
        if not self.keepFiles:
            pid = os.getpid()
            tmpLp = os.path.join(self.tmpDir, "%d-pulp.mps" % pid)
            tmpSol = os.path.join(self.tmpDir, "%d-pulp.sol" % pid)
        else:
            tmpLp = lp.name+"-pulp.mps"
            tmpSol = lp.name+"-pulp.sol"
        vs, variablesNames, constraintsNames, objectiveName = lp.writeMPS(tmpLp, rename = 1)
#        if not self.msg:
#            cbc = os.popen(self.path[1]+" - > /dev/null 2> /dev/null","w")
#        else:
#            cbc = os.popen(self.path[1]+" -","w")
        #TODO make this os agnostic 
        cbc = os.popen(self.path[1]+" -","w")
        cbc.write("import "+tmpLp+"\n")
        if self.presolve:
            cbc.write("presolve on\n")
        cbc.write("strong %d\n" % self.strong)
        if self.cuts:
            cbc.write("gomory on\n")
            cbc.write("oddhole on\n")
            cbc.write("knapsack on\n")
            cbc.write("probing on\n")
        for option in self.options:
            cbc.write(option+"\n")
        if lp.sense == LpMinimize:
            cbc.write("min\n")
        else:
            cbc.write("max\n")
        if self.mip:
            cbc.write("branch\n")
        else:
            cbc.write("initialSolve\n")
        cbc.write("solution "+tmpSol+"\n")
        cbc.write("quit\n")
        if cbc.close() != None:
            raise Exception("PuLP: Error while trying to execute "+self.path[1])
        if not os.path.exists(tmpSol):
            raise Exception("PuLP: Error while executing "+self.path[1])
        lp.status, values = self.readsol_CBC(tmpSol, lp, vs)
        lp.assignVarsVals(values)
        if not self.keepFiles:
            try: os.remove(tmpLp)
            except: pass
            try: os.remove(tmpSol)
            except: pass
        return lp.status

    def solve_CLP(self, lp):
        """Solve a problem using CLP"""
        if not self.executable(self.path[0]):
            raise Exception("PuLP: cannot execute "+self.path[0])
        if not self.keepFiles:
            pid = os.getpid()
            tmpLp = os.path.join(self.tmpDir, "%d-pulp.mps" % pid)
            tmpSol = os.path.join(self.tmpDir, "%d-pulp.sol" % pid)
        else:
            tmpLp = lp.name+"-pulp.mps"
            tmpSol = lp.name+"-pulp.sol"
        vs, variablesNames, constraintsNames, objectiveName = lp.writeMPS(tmpLp, rename = 1)
        if not self.msg:
            clp = os.popen(self.path[0]+" - > /dev/null 2> /dev/null","w")
        else:
            clp = os.popen(self.path[0]+" -","w")
        clp.write("import "+tmpLp+"\n")
        if self.presolve:
            clp.write("presolve on\n")
        for option in self.options:
            clp.write(option+"\n")
        if lp.sense == LpMinimize:
            clp.write("min\n")
        else:
            clp.write("max\n")
        if self.dual:
            clp.write("dualS\n")
        else:
            clp.write("primalS\n")
        clp.write("solution "+tmpSol+"\n")
        clp.write("quit\n")
        if clp.close() != None:
            raise Exception("PuLP: Error while trying to execute "+self.path[0])
        if not os.path.exists(tmpSol):
            raise Exception("PuLP: Error while executing "+self.path[0])
        lp.status, values = self.readsol_CLP(tmpSol, lp, vs, variablesNames, constraintsNames, objectiveName)
        lp.assignVarsVals(values)
        if not self.keepFiles:
            try: os.remove(tmpLp)
            except: pass
            try: os.remove(tmpSol)
            except: pass
        return lp.status

    def readsol_CLP(self,filename, lp, vs, variablesNames, constraintsNames, objectiveName):
        """Read a CLP solution file"""
        values = {}

        reverseVn = {}
        for k,n in variablesNames.iteritems():
            reverseVn[n] = k

        for v in vs:
            values[v.name] = 0.0

        status = LpStatusOptimal # status is very approximate
        f = file(filename)
        for l in f:
            if len(l)<=2: break
            if l[:2] == "**":
                status = LpStatusInfeasible
                l = l[2:]
            l = l.split()
            vn = l[1]
            if vn in reverseVn:
                values[reverseVn[vn]] = float(l[2])
        return status, values

    def readsol_CBC(self,filename, lp, vs):
        """Read a CBC solution file"""
        f = file(filename)
        for i in range(len(lp.constraints)): f.readline()
        values = {}
        for v in vs:
            l = f.readline().split()
            values[v.name] = float(l[1])
        status = LpStatusUndefined # No status info
        return status, values

try:
    import pulpCOIN
    
    class COIN_MEM(LpSolver):
        """The COIN LP solver (via a module)"""
        def __init__(self, mip = 1, msg = 1, cuts = 1, presolve = 1, dual = 1,
            crash = 0, scale = 1, rounding = 1, integerPresolve = 1, strong = 5):
            LpSolver.__init__(self, mip, msg)
            self.cuts = cuts
            self.presolve = presolve
            self.dual = dual
            self.crash = crash
            self.scale = scale
            self.rounding = rounding
            self.integerPresolve = integerPresolve
            self.strong = strong

        def copy(self):
            """Make a copy of self"""
        
            aCopy = LpSolver.copy()
            aCopy.cuts = self.cuts
            aCopy.presolve = self.presolve
            aCopy.dual = self.dual
            aCopy.crash = self.crash
            aCopy.scale = self.scale
            aCopy.rounding = self.rounding
            aCopy.integerPresolve = self.integerPresolve
            aCopy.strong = self.strong
            return aCopy

        def available(self):
            """True if the solver is available"""
            return True

        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            lp.status = pulpCOIN.solve(lp.objective, lp.constraints, lp.sense, 
                self.msg, self.mip, self.presolve, self.dual, self.crash, self.scale,
                self.rounding, self.integerPresolve, self.strong, self.cuts)
            return lp.status
    
    COIN = COIN_MEM
except ImportError:
    class COIN_MEM(LpSolver):
        """The COIN LP solver (via a module)"""
        def available(self):
            """True if the solver is available"""
            return False
        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise RuntimeError, "COIN_MEM: Not Available"

    COIN = COIN_CMD
try:
    import ctypes
    #CoinMP=ctypes.cdll.LoadLibrary(coinMP_path)
    #linux hack to get working
    for libpath in coinMP_path[:-1]:
        ctypes.CDLL(libpath, mode = ctypes.RTLD_GLOBAL)
    CoinMP  = ctypes.CDLL(coinMP_path[-1], mode = ctypes.RTLD_GLOBAL)
    
    class COINMP_DLL(LpSolver):
        """The COIN_MP LP/MIP solver (via a DLL windows only)"""
        COIN_INT_LOGLEVEL = 7
        COIN_REAL_MAXSECONDS = 16
        COIN_REAL_MIPMAXSEC = 19
        COIN_REAL_MIPFRACGAP = 34
        CoinMP.CoinGetInfinity.restype = ctypes.c_double
        CoinMP.CoinGetVersionStr.restype = ctypes.c_char_p
        CoinMP.CoinGetSolutionText.restype=ctypes.c_char_p
        CoinMP.CoinGetObjectValue.restype=ctypes.c_double
        def __init__(self, mip = 1, msg = 1, cuts = 1, presolve = 1, dual = 1,
            crash = 0, scale = 1, rounding = 1, integerPresolve = 1, strong = 5,
            maxSeconds = None, fracGap = None):
            LpSolver.__init__(self, mip, msg)
            self.maxSeconds = None
            if maxSeconds is not None:
                self.maxSeconds = float(maxSeconds)
            self.fracGap = None
            if fracGap is not None:
                self.fracGap = float(fracGap)
            #Todo: these options are not yet implemented
            self.cuts = cuts
            self.presolve = presolve
            self.dual = dual
            self.crash = crash
            self.scale = scale
            self.rounding = rounding
            self.integerPresolve = integerPresolve
            self.strong = strong

        def copy(self):
            """Make a copy of self"""
        
            aCopy = LpSolver.copy()
            aCopy.cuts = self.cuts
            aCopy.presolve = self.presolve
            aCopy.dual = self.dual
            aCopy.crash = self.crash
            aCopy.scale = self.scale
            aCopy.rounding = self.rounding
            aCopy.integerPresolve = self.integerPresolve
            aCopy.strong = self.strong
            return aCopy

        def available(self):
            """True if the solver is available"""
            return True

        def getSolverVersion(self):
            """
            returns a solver version string 
             
            example:
            >>> COINMP_DLL().getSolverVersion() # doctest: +ELLIPSIS
            '...'
            """
            return CoinMP.CoinGetVersionStr()
            


        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            #TODO alter so that msg parameter is handled correctly
            self.debug = 0
            #initialise solver
            CoinMP.CoinInitSolver("")
            #create problem
            hProb = CoinMP.CoinCreateProblem(lp.name);
            #set problem options
            if self.maxSeconds:
                if self.mip:
                    CoinMP.CoinSetRealOption(hProb, self.COIN_REAL_MIPMAXSEC,
                                          ctypes.c_double(self.maxSeconds))
                else:
                    CoinMP.CoinSetRealOption(hProb, self.COIN_REAL_MAXSECONDS,
                                          ctypes.c_double(self.maxSeconds))
            if self.fracGap:
               #Hopefully this is the bound gap tolerance
               CoinMP.CoinSetRealOption(hProb, self.COIN_REAL_MIPFRACGAP,
                                          ctypes.c_double(self.fracGap))
            #CoinGetInfinity is needed for varibles with no bounds
            coinDblMax = CoinMP.CoinGetInfinity()
            if self.debug: print "Before getCoinMPArrays"
            (numVars, numRows, numels, rangeCount,
                objectSense, objectCoeffs, objectConst, 
                rhsValues, rangeValues, rowType, startsBase, 
                lenBase, indBase, 
                elemBase, lowerBounds, upperBounds, initValues, colNames, 
                rowNames, columnType, n2v, n2c) = self.getCplexStyleArrays(lp)
            CoinMP.CoinLoadProblem(hProb, numVars, numRows,
                                   numels, rangeCount,
                                   objectSense, objectCoeffs,
                                   objectConst,
                                   rhsValues, rangeValues,
                                   rowType, startsBase,
                                   lenBase, indBase,
                                   elemBase, lowerBounds,
                                   upperBounds, initValues,
                                   colNames, rowNames )
            if lp.isMIP() and self.mip:
                CoinMP.CoinLoadInteger(hProb,columnType)
            if self.msg == 0:
                #close stdout to get rid of messages
                tempfile = open(mktemp(),'w')
                savestdout = os.dup(1)
                os.close(1)
                if os.dup(tempfile.fileno()) != 1:
                    raise RuntimeError, "couldn't redirect stdout - dup() error"
            self.coinTime = -clock()
            CoinMP.CoinOptimizeProblem(hProb, 0);
            self.coinTime += clock()

            if self.msg == 0:
                #reopen stdout
                os.close(1)
                os.dup(savestdout)
           
            CoinLpStatus = {0:LpStatusOptimal,
                            1:LpStatusInfeasible,
                            2:LpStatusInfeasible,
                            3:LpStatusNotSolved,
                            4:LpStatusNotSolved,
                            5:LpStatusNotSolved,
                            -1:LpStatusUndefined
                            }
            solutionStatus = CoinMP.CoinGetSolutionStatus(hProb)
            solutionText = CoinMP.CoinGetSolutionText(hProb,solutionStatus)
            objectValue =  CoinMP.CoinGetObjectValue(hProb)

            #get the solution values
            NumVarDoubleArray = ctypes.c_double * numVars
            NumRowsDoubleArray = ctypes.c_double * numRows
            cActivity = NumVarDoubleArray()
            cReducedCost = NumVarDoubleArray()
            cSlackValues = NumRowsDoubleArray()
            cShadowPrices = NumRowsDoubleArray()
            CoinMP.CoinGetSolutionValues(hProb, ctypes.byref(cActivity),
                                         ctypes.byref(cReducedCost),
                                         ctypes.byref(cSlackValues),
                                         ctypes.byref(cShadowPrices))

            variablevalues = {}
            variabledjvalues = {}
            constraintpivalues = {}
            constraintslackvalues = {}
            for i in range(numVars):
                variablevalues[self.n2v[i].name] = cActivity[i]
                variabledjvalues[self.n2v[i].name] = cReducedCost[i]
            lp.assignVarsVals(variablevalues)
            lp.assignVarsDj(variabledjvalues)            
            #put pi and slack variables against the constraints
            for i in range(numRows):
                constraintpivalues[self.n2c[i]] = cShadowPrices[i]
                constraintslackvalues[self.n2c[i]] = rhsValues[i] - cSlackValues[i]                
            lp.assignConsPi(constraintpivalues)
            lp.assignConsSlack(constraintslackvalues)
                       
            CoinMP.CoinFreeSolver()
            lp.status = CoinLpStatus[CoinMP.CoinGetSolutionStatus(hProb)]
            return lp.status

        def getCoinMPArrays(self,lp,coinDblMax):
            """returns the arrays suitable to pass to a cdll CoinMP

            Copywrite (c) Stuart Mitchell 2007
            """
            rangeCount = 0
            objectSense = lp.sense
            vars=lp.variables()
            numVars = len(vars)
            #print "In getCoinMPArrays"
            #associate each variable with a ordinal
            v2n=dict(((vars[i],i) for i in range(numVars)))
            vname2n=dict(((vars[i].name,i) for i in range(numVars)))            
            n2v=dict((i,vars[i]) for i in range(numVars))
            #print "After Dictionaries"
            #objective values
            NumVarDoubleArray = ctypes.c_double * numVars
            objectCoeffs=NumVarDoubleArray()
            print "Get objective Values"
            for v,val in lp.objective.iteritems():
                objectCoeffs[v2n[v]]=val
            #values for variables
            NumVarStrArray = ctypes.c_char_p * numVars
            colNames = NumVarStrArray()
            lowerBounds = NumVarDoubleArray()
            upperBounds = NumVarDoubleArray()
            initValues = NumVarDoubleArray()
            #print "Get Variable information"
            for v in lp.variables():
                colNames[v2n[v]] = str(v.name)
                initValues[v2n[v]] = 0.0
                if v.lowBound != None:
                    lowerBounds[v2n[v]] = v.lowBound
                else:
                    lowerBounds[v2n[v]] = -coinDblMax
                if v.upBound != None:
                    upperBounds[v2n[v]] = v.upBound
                else:
                    upperBounds[v2n[v]] = coinDblMax
            #values for constraints
            numRows =len(lp.constraints)
            NumRowDoubleArray = ctypes.c_double * numRows
            NumRowStrArray = ctypes.c_char_p * numRows
            NumRowCharArray = ctypes.c_char * numRows
            coinSense={LpConstraintEQ:"E", LpConstraintLE:"L", LpConstraintGE:"G"}
            rhsValues = NumRowDoubleArray()
            rangeValues = NumRowDoubleArray()
            rowNames = NumRowStrArray()
            rowType = NumRowCharArray()
            c2n = {}
            n2c = {}
            i = 0
            print "Get constraint information"
            for c in lp.constraints:
                rhsValues[i] = -lp.constraints[c].constant
                #range values are constraints that are >= a and <=b
                rangeValues[i] = 0.0
                rowNames[i] = str(c)
                rowType[i] = coinSense[lp.constraints[c].sense]
                c2n[c] = i
                n2c[i] = c
                i = i+1

            #return the coefficient matrix as a series of vectors
            coeffs = lp.coefficients()
            numels = len(coeffs)  #seems to be all this is good for
            NumelsDoubleArray = ctypes.c_double * numels
            NumelsIntArray = ctypes.c_int * (numels)
            elemBase = NumelsDoubleArray()
            myelemBase = []
            indBase = NumelsIntArray()
            myindBase = []
            mystartsBase = []
            mylenBase = []
            NumVarspIntArray = ctypes.c_int * (numVars + 1)
            startsBase = NumVarspIntArray()
            lenBase = NumVarspIntArray()
            if self.debug: print "get Coefficient information"
            elements = [[] for i in range(numVars)]
            for var in coeffs:
                elements[vname2n[var[0]]].append([var[1],var[2]])
            if self.debug: print "constructed array"
                
            for v in vars:
                startsBase[v2n[v]] = len (myelemBase)
                mystartsBase.append(len (myelemBase))
                myelemBase.extend((var[1] for var in elements[v2n[v]]))
                myindBase.extend((c2n[var[0]] for var in elements[v2n[v]]))
                lenBase[v2n[v]] = len(myelemBase) - startsBase[v2n[v]]
                mylenBase.append(len(myelemBase) - startsBase[v2n[v]])
            startsBase[numVars] = len(myelemBase)
            for i in range(numels):
                elemBase[i]=myelemBase[i]
                indBase[i]=myindBase[i]
            #MIP Variables
            NumVarCharArray = ctypes.c_char * numVars
            columnType = NumVarCharArray()
            if lp.isMIP():
                CoinLpCategories = {LpContinuous: "C",
                                    LpInteger: "I"}
                for v in lp.variables():
                    columnType[v2n[v]] = CoinLpCategories[v.cat]
            ##print "return all info and solve"
            return  numVars, numRows, numels, rangeCount, objectSense, objectCoeffs, \
                rhsValues, rangeValues, rowType, startsBase, lenBase, indBase, \
                elemBase, lowerBounds, upperBounds, initValues, colNames, \
                rowNames, columnType, n2v, n2c
        
    
    COIN = COINMP_DLL
except (ImportError,OSError):

    class COINMP_DLL(LpSolver):
        """The COIN_MP LP MIP solver (via a DLL Windows only)"""
        def available(self):
            """True if the solver is available"""
            return False
        def actualSolve(self, lp):
            """Solve a well formulated lp problem"""
            raise RuntimeError, "COINMP_DLL: Not Available"

    COIN = COIN_MEM

try:
    import ctypes
    def ctypesArrayFill(myList,type = ctypes.c_double):
    	"""Creates a c array with ctypes from a python list
    	type is the type of the c array
    	"""
    	ctype= type * len(myList)
    	cList = ctype()
    	for i,elem in enumerate(myList):
        	cList[i] = elem 
    	return cList
except(ImportError):
    def ctypesArrayFill(myList,type = None):
    	return None

