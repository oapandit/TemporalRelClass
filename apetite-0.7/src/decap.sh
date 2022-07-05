#!/bin/sh
##
## decap.sh
## 
## Made by Pascal Denis
## Login   <pascal@galveston>
## 
## Started on  Wed Feb 10 10:47:01 2010 Pascal Denis
## Last update Wed Feb 10 10:49:25 2010 Pascal Denis
##

for d in `ls $1`; do cat $1/$d | tr '[A-Z]' '[a-z]'> tmp; mv tmp $1/$d; done
