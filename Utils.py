import numpy
def save(pathname,data):
    numpy.save(pathname,data)
def load(pathname):
    return numpy.load(pathname,allow_pickle=True)


