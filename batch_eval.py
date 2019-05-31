"""
a program for evaluating the quality of search algorithms using the vector model

it runs over all queries in query.text and get the top 10 results,
and then qrels.text is used to compute the NDCG metric

usage:
    python batch_eval.py index_file query.text qrels.text n

    output is the average NDCG over all the queries for boolean model and vector model respectively.
	also compute the p-value of the two ranking results.
"""
import metrics
import random
import cranqry
from cranqry import loadCranQry
from query import query
import index
import numpy as np
from metrics import dcg_score
from metrics import ndcg_score
from index import IndexItem
from index import Posting
from index import InvertedIndex
from query import QueryProcessor
from scipy import stats
import sys


# compute ndcg scores for booleanQuery and vector query models for randomly generated n query_ids and p value using ttest and wilcoxon test
def eval(index_file, query_text, qrels, n):
    qrys = cranqry.loadCranQry(query_text)
    queries = {}
    for q in qrys:
        queries[q] = qrys[q].text
    query_ids = list(queries.keys())
    query_ids.sort()
    query_ids_ints = []
    for k in range(0, len(query_ids)):  # generating n random queries
        query_ids_ints.append(int(query_ids[k]))
    set1 = set()
    while len(set1) != n:
        set1.add(random.choice(query_ids_ints))
    selected_queries = list(set1)
    docs = set()
    qrels = {}

    f = open("qrels.text", "r")  # parsing relevant queries(qrels.text)
    l = f.readline()
    while l:
        j = l.split(" ")
        if query_ids_ints[int(j[0]) - 1] in qrels.keys():
            qrels[query_ids_ints[int(j[0]) - 1]].append(int(j[1]))
        else:
            qrels[query_ids_ints[int(j[0]) - 1]] = [int(j[1])]
        l = f.readline()
    cranqryobj = cranqry.loadCranQry(query_text)
    dict_query = {}
    for q in cranqryobj:
        dict_query[int(q)] = cranqryobj[q].text  # matching queries in query.text and qrels.text
    indexObject = index.InvertedIndex()
    items = indexObject.load(index_file)
    vector_ndcg_score = {}
    vector_score_dict = {}
    for q in selected_queries:
        print(q)
        query_raw = dict_query[q]
        QPobj = QueryProcessor(query_raw, items, index_file)
        QPobj.preprocessing()
        result_list = QPobj.vectorQuery(10)  # fetching first 10 documents for a query using vector model
        boolean_result_list = QPobj.booleanQuery()
        print("Boolean query result : ", boolean_result_list) # fetching documents for a query using booleanQuery
        ndcg_boolean = 0
        truth_list = qrels[q]
        boolean_output_list = []
        rank_doc_list = list(map(lambda x: int(x[0]), result_list))
        print("Relavant documents for this query : ", truth_list)  # relavant documents for the query
        print("Vector model result : ", rank_doc_list)  # documents result list for vector model
        vector_score_list = []
        for id in boolean_result_list:  # calculating the predicted scores for boolean model
            if int(id) in truth_list:
                boolean_output_list.append(1)
            else:
                boolean_output_list.append(0)
        boolean_score_list = []
        if len(boolean_score_list) < 10:
            boolean_score_list = boolean_output_list
            while len(boolean_score_list) != 10:
                boolean_score_list.append(0)
        elif len(boolean_score_list) > 10:
            for i in range(0, 10):
                boolean_score_list[i] = boolean_output_list[i]
        for id in rank_doc_list:  # calculating the predicted scores for vector model

            if id in truth_list:
                vector_score_list.append(1)
            else:
                vector_score_list.append(0)
        vector_score_dict[q] = vector_score_list
        truth_score_list = []
        for i in range(0, len(vector_score_list)):  # calculating the ground_truth scores for vector model
            truth_score_list.append(vector_score_list[i])
        truth_score_list.sort(reverse=True)

        boolean_truth_score_list = []
        for i in range(0, len(boolean_score_list)):  # calculating the ground_truth scores for boolean model
            boolean_truth_score_list.append(boolean_score_list[i])
        boolean_truth_score_list.sort(reverse=True)
        print("Vector model ground_truth list is:\n", truth_score_list)
        print("Vector ranking score list is:\n", vector_score_list)
        print("Boolean model ground_truth list is:\n", boolean_truth_score_list)
        print("Boolean model score list is:\n", boolean_score_list)
        vector_ndcg_score[q] = [ndcg_score(np.array(boolean_truth_score_list), np.array(boolean_score_list)),
                                ndcg_score(np.array(truth_score_list), np.array(vector_score_list))]
    vector_list = []  # compute ndcg score for boolean and vector models for all the randomly generated queries
    boolean_list = []
    for qu in vector_ndcg_score:
        vector_list.append(vector_ndcg_score[qu][1])
        boolean_list.append(vector_ndcg_score[qu][0])

    print("ndcg score of boolean and vector models for all the queries:\n", vector_ndcg_score)
    print("ndcg scores list for boolean model for all the queries:\n", boolean_list)
    print("ndcg scores list for vector model for all the queries:\n", vector_list)
    p_value_wilcoxon = stats.wilcoxon(np.array(boolean_list), np.array(
        vector_list))  # calculating p value using wilcoxon test and ttest  for boolean and vector models  p_value_ttest=stats.ttest_ind(np.array(boolean_list),np.array(vector_list), equal_var = False)
    print("wilcoxon test p value is:", p_value_wilcoxon[1])
    print("ttest p value is :", p_value_ttest[1])


if __name__ == "__main__":
    index_file = str(sys.argv[1])
    query_text = str(sys.argv[2])
    qrels_text = str(sys.argv[3])
    n_sample = str(sys.argv[4])
    eval(index_file, query_text, qrels_text, int(n_sample))
