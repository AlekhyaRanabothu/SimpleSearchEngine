"""
query processing

"""
import math
import sys
from operator import itemgetter

import numpy as np
from nltk.tokenize import word_tokenize

import cranqry
import index
import norvig_spell
import util
from index import InvertedIndex, IndexItem, Posting


class QueryProcessor:

    def __init__(self, query, index, collection):
        """ index is the inverted index; collection is the document collection"""
        self.raw_query = query
        self.index = index
        self.filename = collection
        self.tokens = []

    def preprocessing(self):

        """ apply the same preprocessing steps used by indexing,
            also use the provided spelling corrector. Note that
            spelling corrector should be applied before stopword
            removal and stemming (why?)"""
        print(self.raw_query)
        tokens = word_tokenize(self.raw_query)
        alpha_tokens = [norvig_spell.correction(token) for token in tokens if
                        token.isalpha()]  # tokenizing the query,norvig_spell check and removing punctuations
        self.tokens = [

            util.stemming(
                token.lower()

            ) for token in alpha_tokens if
            not util.isStopWord(token)]  # remove stopwords
        return self.tokens
        # ToDo: return a list of terms

    def unitVector(self, vect):
        # converting a vector to unit vector (used for computing dot product for cosine similarity)
        mag = 0.0
        for i in range(0, len(vect)):
            mag = mag + vect[i] ** 2
        mag = math.sqrt(mag)
        unitVect = [x / mag for x in vect]
        return unitVect

    def booleanQuery(self):
        """ boolean query processing; note that a query like "A B C" is transformed to "A AND B AND C"
        for retrieving posting lists and merge them"""
        ivObj = InvertedIndex()
        ivObj.load(self.filename)
        index_item = ivObj.items[self.tokens[0]]
        # Get the doc ids from the sorted postings in the same order.
        docs = index_item.get_sorted_doc_ids()
        for token in self.tokens:
            index_item = ivObj.items[token]
            # Find intersection between the current docs and the index_item for the current token.
            docs = index_item.intersection(docs)
        return docs

    # ToDo: return a list of docIDs

    def vectorQuery(self, k):
        """ vector query processing, using the cosine similarity. """
        # ToDo: return top k pairs of (docID, similarity), ranked by their cosine similarity with the query in the descending order
        # You can use term frequency or TFIDF to construct the vectors
        result = {}
        ivObj = InvertedIndex()
        ivObj.load(self.filename)  # loading the InvertedIndex
        doc_set = set()
        term_idf_list = []
        for term in self.tokens:  # for every term in the query finding the document IDs where the term is present
            if term in self.index:
                doc_set = doc_set.union(set(self.index[term].posting.keys()))
            term_idf_list.append(ivObj.idf(term) * 1.0 / len(self.tokens))  # calculating tf-idf weights for query
        doc_list = list(doc_set)
        for docID in doc_list:  # Calculating tf-idf weights for the above documents
            for term in self.tokens:
                if term in self.index:
                    if docID in result.keys():
                        result[docID].append(ivObj.tfidf(term, docID))
                    else:
                        result[docID] = [ivObj.tfidf(term, docID)]
                else:
                    if docID in result.keys():
                        result[docID].append(0.0)
                    else:
                        result[docID] = [0.0]

        score_dict = {}
        term_idf_list_np = np.array(self.unitVector(term_idf_list))  # calculating unit vector for each document
        for docID in doc_list:
            unit_result = self.unitVector(result[docID])
            unit_np = np.array(unit_result)
            score_dict[docID] = np.dot(term_idf_list_np, unit_np)  # dot product for query and each document
        score_list = score_dict.items()
        final = sorted(score_list, key=itemgetter(1), reverse=True)
        similarity = []
        for i in range(0, k):
            similarity.append(final[i])
        return similarity  # list of (docID,cosine similarity) in order of ranking


def query(index_file, processing_algorithm, query_file, query_id):
    """ the main query processing program, using QueryProcessor"""
    cranqryobj = cranqry.loadCranQry(query_file)
    dict_query = {}
    for q in cranqryobj:
        dict_query[q] = cranqryobj[q].text
    query_txt = dict_query[query_id]
    indexObject = index.InvertedIndex()
    items = indexObject.load(index_file)
    QPobj = QueryProcessor(query_txt, items, index_file)
    QPobj.preprocessing()
    doc_ids = []
    if processing_algorithm == "0":  # boolean Query
        doc_ids = QPobj.booleanQuery()
    elif processing_algorithm == "1":  # vector Query
        doc_ids = QPobj.vectorQuery(3)  # first 3 documents based on ranking
    else:
        print("Invalid Processing algorithm")
    print(doc_ids)
    return doc_ids
    # ToDo: the commandline usage: "echo query_string | python query.py index_file processing_algorithm"
    # processing_algorithm: 0 for booleanQuery and 1 for vectorQuery
    # for booleanQuery, the program will print the total number of documents and the list of docuement IDs
    # for vectorQuery, the program will output the top 3 most similar documents


if __name__ == "__main__":
    # test(
    index_file = str(sys.argv[1])
    process_algorithm = sys.argv[2]
    query_text = str(sys.argv[3])
    queryID = str(sys.argv[4])
    # print(index_file, process_algorithm, query_text, queryID)
    query(index_file, process_algorithm, query_text, queryID)
    # query("index_file",1,"query.text","112")
