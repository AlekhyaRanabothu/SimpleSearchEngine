import math
import operator
import sys

"""

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

"""
import util
import cran
import pickle


class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []

    def __repr__(self):
        return str(self.positions)

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        """ sort positions"""
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        """ return the term frequency in the document"""
        return len(self.positions)


class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {}  # postings are stored in a python dict for easier index building
        self.sorted_postings = []  # may sort them by docID for easier query processing

    def __repr__(self):
        return str(self.posting)

    def add(self, docid, pos):
        """ add a posting"""
        if docid not in self.posting:
            self.posting[docid] = Posting(docid)
        self.posting[docid].append(pos)

    def sort(self):
        """ sort by document ID for more efficient merging. For each document also sort the positions"""
        for key in self.posting:  # sorting documentIDs and positions
            self.posting[key].sort()
        self.sorted_postings = sorted(self.posting.items(), key=operator.itemgetter(0))

    def intersection(self, doc_ids):
        """
        To perform intersection between given set of docIDs and docIDs present in the given Index item.
        :param doc_ids: Set of document ids to perform intersection with.
        :return: result of intersection.
        """
        i, j = 0, 0
        n = len(doc_ids)
        m = len(self.sorted_postings)
        # Getting docIDs from the sorted postings
        result = []
        # Generate the intersection using Merge algorithm.
        while i < n and j < m:
            x = doc_ids[i]
            y = self.sorted_postings[j]
            if x == y[0]:
                result.append(x)
                i += 1
                j += 1
            elif x < y[0]:
                i += 1
            else:
                j += 1
        return result

    def get_sorted_doc_ids(self):
        # return all doc ids from the sorted postings in the same order.
        return [x[0] for x in self.sorted_postings]


class InvertedIndex:

    def __init__(self):
        self.items = {}  # list of IndexItems
        self.nDocs = 0  # the number of indexed documents

    def indexDoc(self, doc):  # indexing a Document object
        """ indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead"""
        self.nDocs += 1
        tokens = util.tokenize(doc.title + "\n" + doc.body)
        for i, token in enumerate(tokens):
            if token not in self.items:
                self.items[token] = IndexItem(token)
            self.items[token].add(doc.docID, i)

    # ToDo: indexing only title and body; use some functions defined in util.py
    # (1) convert to lower cases,
    # (2) remove stopwords,
    # (3) stemming

    def sort(self):
        """ sort all posting lists by docID"""
        for key in self.items:
            self.items[key].sort()

    def find(self, term):
        return self.items[term]

    def save(self, filename):
        """ save to disk"""
        # saving InvertedIndex object into a pickle file with name "index_file" to the disk in the current directory

        with open(filename, "wb") as output:
            pickle.dump(self.items, output)
            output.close()

    def load(self, filename):
        """ load from disk"""
        # To load the InvertedIndex stored in "index_file" from the disk
        with open(filename, "rb") as input:
            self.items = pickle.load(input)
            input.close()
        set1 = set()
        for it in self.items:
            set1 = set1.union(set(self.items[it].posting.keys()))
        self.nDocs = len(set1)
        return self.items

    def idf(self, term):
        """ compute the inverted document frequency for a given term"""
        # ToDo: return the IDF of the term
        if term in self.items.keys():
            return 1.0 + math.log(float(self.nDocs) / len(self.items[term].posting.keys()))
        else:
            return 0.0

    # more methods if needed
    def tf(self, term, docId):
        if docId not in self.items[term].posting:
            return 0
        else:
            return self.items[term].posting[docId].term_freq()

    def tfidf(self, term, docId):
        if self.tf(term, docId) == 0:
            return 0.0
        return self.tf(term, docId) * self.idf(term)


def indexingCranfield(cran_all_file, index_file):
    # ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python index.py cran.all index_file"
    # the index is saved to index_file
    cf = cran.CranFile(cran_all_file)
    indexobj = InvertedIndex()
    for doc in cf.docs:
        indexobj.indexDoc(doc)
    print(indexobj.items)
    print("Done")
    indexobj.sort()
    indexobj.save(index_file)


if __name__ == "__main__":
    # test()
    crandata_file = str(sys.argv[1])
    index_file = str(sys.argv[2])
    indexingCranfield(crandata_file, index_file)
