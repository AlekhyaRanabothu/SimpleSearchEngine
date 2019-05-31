# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:18:51 2019

@author: alekh
"""

from nltk.tokenize import word_tokenize
import cran
import util


def tokenise(documents):
    tokens = word_tokenize(doc)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    return alpha_tokens


if __name__ == "__main__":
    cf = cran.CranFile("cran.all")
    documents = []
    tokens=[]
    for doc in cf.docs:
        documents.append(doc.body)
        
    #print(documents)
    for doc in documents:
        tokens.append(util.tokenize(doc))
    print(tokens)
    
