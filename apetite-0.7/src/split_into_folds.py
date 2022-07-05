#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
import os
import operator
import optparse


'''

Script for splitting files in input dir into N folds, where N can be
specified as an option (-n).

The script creates a new dir (default path: ./folds), containing N
subdirs called: 1/, 2/, ... N/. Each such dir has two subdirs: train/
and test/. These contain symbolic links pointing to the original file
in the input dir.

'''
    
usage = "usage: %prog [options] dir"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-n", "--number_of_folds", type=int, default=10, help="number of folds for cross-validation (default: 10)")
parser.add_option("-x", "--file_extension", type=str, default='.tml.xml', help="file extension for file selection (default: tml.xml)")
parser.add_option("-d", "--fold_dir", default='./folds', help="path to output directory for experiment (default path: ./folds)")


(options, args) = parser.parse_args()


_dir = args[0]
n = options.number_of_folds
    
# create list of files from _dir
file_ext = options.file_extension
files = [os.path.join(_dir,f) for f in os.listdir(_dir)
         if f.endswith(file_ext) and not f.startswith('.')]

file_ct = len(files)
if file_ct == 0:
    sys.exit("Error: No file with file extension %s in %s" %(file_ext,_dir))
print "%s files in %s" %(file_ct, _dir)


# divide into sections
if n == 1: n += 1 # at least 2 sections!
sections = [None]*n
for i in xrange(n):
    sections[i] = []
for i in xrange(len(files)):
    sections[i % n].append(files[i])
    

# create output dir
print "Creating output folds directory %s/" %options.fold_dir
fold_dir = options.fold_dir
if os.path.isdir(fold_dir):
    os.system("rm -rf %s" % fold_dir)
elif os.path.isfile(fold_dir):
    raise OSError("A file with the same name as the desired dir, " \
                  "'%s', already exists." % fold_dir)
os.makedirs(fold_dir)


# create foldX subdirs, each containing train/ and test/ subdirs
for run in xrange(n):
    print ">> Creating subdir fold %s..." %run
    foldx_dir = os.path.join(fold_dir,str(run))
    os.makedirs(foldx_dir)
    test_files = sections[run]
    train_files = sections[0:run] + sections[run+1:n]
    # flatten into a single list of training files
    train_files = reduce(operator.add, train_files)
    # create train/ and test/ dirs and fill'em up
    train_dir = os.path.join(foldx_dir,'train')
    os.makedirs(train_dir)
    for (i,f) in enumerate(train_files):
        #f_path = os.path.abspath(f)
        f_path = os.path.join("../../..",f)  # FIXME: totally ad hoc for svn commit
        os.system('ln -s %s %s' %(f_path,train_dir))
    print i+1, "files for training"
    test_dir = os.path.join(foldx_dir,'test')
    os.makedirs(test_dir)
    for (i,f) in enumerate(test_files):
        #f_path = os.path.abspath(f)
        f_path = os.path.join("../../..",f) # FIXME: totally ad hoc for svn commit
        os.system('ln -s %s %s' %(f_path,test_dir))
    print i+1, "files for test"
