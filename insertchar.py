import json #used to decode the JSON files and fetch the data
import urllib.request as urllib #used for Google Knowledge Graph
import ast
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.random import normal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score
import spacy
from sklearn.datasets import make_classification
from demjson import decode
import nltk
from sklearn.preprocessing import PolynomialFeatures

def KnuthMorrisPratt(text, pattern):

    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos

for s in KnuthMorrisPratt([4,'a',3,9,5,6], [5,6]): print(s)