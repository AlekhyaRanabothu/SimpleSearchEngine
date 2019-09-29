# SimpleSearchEngine

-- Built a search engine for Cranfield dataset

-- tokenize the corpus, remove stopwords, PorterStemmer for stemming

-- norvig spell checker

-- built an inverted index of the documents in the corpus using tf-idf weights and process the queries with Boolean Model and Vector Model

-- BooleanModel-used AND(Intersection) for Query processing

-- VectorModel-cosine similarity for ranking

-- evaluation of boolean and vector model using ndcg score,t-test and wilcoxon test

************************************************************************************************************
# Project flow and Additional details

•	The goal of this project is to build a basic search engine which gives out the documents from the corpus, that are related to the search queries.

•	Basically it takes queryID as input and gives the relevant document IDs in the corpus as output.

•	The dataset used for this project is Cranfield dataset. An inverted index of the corpus is built. The inverted index contains the dictionary of terms in the corpus and for each term the document ids of the documents in which the term is present and the positions of the term in each document are stored.

•	In this project we developed Boolean and vector models for query processing. Query IDs and corresponding query terms are stored in a file. When a query ID is given for processing, the corresponding query is fetched from the query file. 

•	For Boolean model, we calculate the intersection(AND) of the terms of the query and return the document IDs in the inverted index for the all terms present in the query. 

•	For vector model, we construct unit vectors of tf-idf weights for query and all the documents where terms in the query are present in the Inverted Index. 

•	Then we find cosine similarity between query vector and all the other document vectors. Now we sort these based on the similarity and return the top k vectors. We compute the ground_truth and predicted scores and ndcg scores for Boolean and vector models. Then we compare the Boolean and vector models by computing the p value by Wilcoxon test and t-test.
