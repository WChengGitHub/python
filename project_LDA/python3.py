from gensim.corpora import Dictionary
from gensim.models import ldamodel
import numpy

texts = [['bank','river','shore','water'],
        ['river','water','flow','fast','tree'],
        ['bank','water','fall','flow'],
        ['bank','bank','water','rain','river'],
        ['river','water','mud','tree'],
        ['money','transaction','bank','finance'],
        ['bank','borrow','money'],
        ['bank','finance'],
        ['finance','money','sell','bank'],
        ['borrow','sell'],
        ['bank','loan','sell']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

numpy.random.seed(1) # setting random seed to get the same results each time.
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)


bow_water = ['bank','water','bank']
bow_finance = ['money','transaction','bank','finance']

bow = model.id2word.doc2bow(bow_finance) # convert to bag of words format first
doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)


#print(word_topics)
print(doc_topics)
#print(phi_values)

bow = model.id2word.doc2bow(bow_water) # convert to bag of words format first
doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)


#print(word_topics)
print(doc_topics)
#print(phi_values)

# bow = model.id2word.doc2bow(bow_finance) # convert to bag of words format first
# doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)
#
# word_topics