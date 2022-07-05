class Source:
    pass


class SimpleClassifierSource(Source):
    """ Data source class for simple format files: white-space
    separated feature-value strings, first item is class label"""

    def __init__(self,infile):
        """Create a data source from a simple source file"""
        self.stream = open(infile)
        return

    def __iter__(self):
        return self 

    def next(self):
        # If we're at the end, rewind stream and stop
        line = self.stream.readline().rstrip()
        if (line == ''):
            self.stream.seek(0)
            raise StopIteration
        # get features
        items = line.split()
        # get category
        cl = intern(items[0]) # intern() ensures faster dict lookup
        # collect feature/value pairs
        items = map( intern, items[1:] )
        return (cl,items)


class C4_5Source(Source):
    """Data source class for C4.5 files: comma-separated feature
    values, last item is class label. Feature names (normally provided
    in *.names file) are ignored and replaced by indices."""

    def __init__(self,infile):
        """Create a data source from a C4.5 file"""
        self.stream = open(infile)
        return

    def __iter__(self):
        return self 

    def next(self):
        # If we're at the end, rewind stream and stop
        line = self.stream.readline().rstrip()
        if (line == ''):
            self.stream.seek(0)
            raise StopIteration
        # get features
        if (line[-1] == '.'):
            line = line[:-1]
        items = line.split(',')
        # get class label
        cl = intern(items.pop())
        # collect feature/value pairs
        items = [ (i,intern(items[i])) for i in xrange(0,len(items)) ]
        # return cl,features pair
        return (cl,items)
        # return (items,cl,line) # output needed for official TADM interface



