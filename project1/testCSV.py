# This Python file uses the following encoding: utf-8  m
import os
import csv
import jieba
import lda
import gc
import mysql.connector
import  numpy as np
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

stopword_file = open("./stopword.txt", mode="r", encoding="utf-8")
stopword_content_string = stopword_file.read()
stopword_list = stopword_content_string.split('\n')

data_path="../data"
segment_path="segment"
def loadTrain():
    allfiles=[]
    files=os.listdir("%s/%s"%(data_path,segment_path))
    for file in files:
        f=open("%s/%s/%s"%(data_path,segment_path,file),mode="r")
        text=f.read();
        allfiles.append(text)
        #print(allfiles)
    return allfiles


def readCSV(path):
    with open(path, mode="r", encoding="GB18030") as csvfile:
        list=[]
        rows = csv.reader(csvfile)
        for i,row in enumerate(rows):
            list.append(row)

    return list



def readCSVRow(path,n):
    with open(path,mode="r",encoding="GB18030") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[n] for row in reader]
    return column

#print(readCSVRow("demo.csv",1))


def writeCSVRow(path1,path2,data):
    with open(path1,mode="r",encoding="GB18030") as csvfile:
        rows = csv.reader(csvfile)
        with open(path2,'w',newline='',encoding="GB18030") as f:
            writer = csv.writer(f)
            for i,row in enumerate(rows):
                row.append(data[i])
                writer.writerow(row)


def seg_data(data):
    list=[]
    for i in range(len(data)):
        seg_d=jieba.cut(data[i])
        list.append(" ".join(seg_d))

    return list


#writeCSVRow("data.csv","data1.csv",seg_data(readCSVRow("data.csv",2)))
def get_stop_words(path):
    with open(path,"r",encoding="utf-8") as file:
        return set([line.strip() for line in file])


def seg_stop_data(data):
    list = []
    stop_words = get_stop_words("stopword.txt")
    stop_words.add(' ')
    stop_words.add(' ')
    stop_words.add('\u3000')
    stop_words.add('\n')
    stop_words.add('2340')
    for i in range(len(data)):
        seg_d = jieba.cut(data[i])
        seg_da =[item for item in seg_d if str(item) not in stop_words]
        #print(seg_da)
        #print(seg_da)
        list.append(" ".join(seg_da))

    return list

def content_model(data):
    dic = corpora.Dictionary(data)
    dic.save("dic.m")
    corpus = [dic.doc2bow(text) for text in data]
    corpora.MmCorpus.serialize("corpus.m",corpus)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.LdaModel(corpus_tfidf,id2word=dic,num_topics=20,alpha=1)
    lda.save("ldaModel.model")


def useLdAModel():
    dic = corpora.Dictionary.load("dic.m")
    corpus = corpora.MmCorpus("corpus.m")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]


    lda = models.ldamodel.LdaModel.load ("ldaModel.model")
    print(lda.print_topics(20))
    n = 10
    topics_words = lda.get_topics()
    for i, topic_words in enumerate(topics_words):
        topic_word_ptr = np.array(dic)[np.argsort(topic_words)][:-(n + 1):-1]
        topic_word_ptrs = []
        for j in range(n):
            topic_word_ptrs.append(dic[topic_word_ptr[j]])
        print("topic:{}\n -{}".format(i, topic_word_ptrs))
        del topic_word_ptr
        gc.collect()

    corpus_lda = lda[corpus_tfidf]
    # print(corpus_lda[10])
    docments_topics = corpus_lda
    list = []
    for i, docment_topics in enumerate(docments_topics):
        li = []
        for j in range(len(docment_topics)):
            li.append(docment_topics[j][1])
            # print(li)
        # print(np.argsort(li)[0])
        list.append(np.argsort(li)[0])

    return list

def content_vectorizer_model(data):
    # print(type(data))

    # print(stopword_list)

    vectorizer = CountVectorizer(stop_words=stopword_list)
    x = vectorizer.fit_transform(data)
    words = vectorizer.get_feature_names()

    n=3
    model = lda.LDA(n_topics=n, n_iter=2000, random_state=1)
    model.fit_transform(x)


    topics_words = model.topic_word_
    for i, topic_words in enumerate(topics_words):
        topic_word_ptr = np.array(words)[np.argsort(topic_words)][:-(n + 1):-1]
        print("topic:{}\n -{}".format(i, topic_word_ptr))
        del topic_word_ptr
        gc.collect()

    dopics_words = model.doc_topic_

    list=[]
    for i,dopic_words in enumerate(dopics_words):
        list.append(dopic_words.argmax())


    return list

def dealData(data):
    list = []
    for i,da in enumerate(data):
        da.append(i)
        dat = tuple(da)
        list.append(dat)
    return list

def dealData1(data):
    list=[]
    for i in range(len(data)):
        da = data[i]
        dad = data[i].split(" ")
        list.append(dad)

    return list

def insertIntoDatabase(data):
   conn = mysql.connector.connect(user="root",password="123456",database="news_classify")
   cursor = conn.cursor();

   sql = "insert into tb_article(article_title,author,article_context,seg_context,topic_id,article_id) values (%s,%s,%s,%s,%s,%s)"

   cursor.executemany(sql,data)
   conn.commit();

   cursor.close()
   conn.close();



def predict_data(data):
    dic = corpora.Dictionary.load("dic.m")
    corpus = [dic.doc2bow(text) for text in data]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("ldaModel.model")
    corpus_lda = lda[corpus_tfidf]
    # print(corpus_lda[10])
    docments_topics = corpus_lda
    list = []
    for i, docment_topics in enumerate(docments_topics):
        li = []
        for j in range(len(docment_topics)):
            li.append(docment_topics[j][1])
            # print(li)
        # print(np.argsort(li)[0])
        list.append(np.argsort(li)[0])

    return list
'''
data = readCSVRow("data1.csv",3)
data_i = content_vectorizer_model(data)
writeCSVRow("data1.csv","data2.csv",data_i)
'''
'''
data1 = readCSV("data2.csv")
data2 = dealData(data1)
insertIntoDatabase(data2)

#print(get_stop_words("stopword.txt"))
'''
'''
data = readCSVRow("news-data.csv",0) #把文章内容提取出来
data1 = seg_stop_data(data) #把内容中不需要的部分去掉
writeCSVRow("news-data.csv","news-data1.csv",data1)#把内容分词,重新存储到另一张表 #
'''
'''
data = readCSVRow("news-data1.csv",2)
data = dealData1(data)
data_i=predict_data(data)#训练模型
'''
#writeCSVRow("news-data1.csv","news-data2.csv",data_i)#写入主题号

'''
data1 = readCSV("data5.csv")
data2 = dealData(data1)
insertIntoDatabase(data2) #写入数据库
'''

'''
data=loadTrain()
data=seg_stop_data(data)
data=dealData1(data)
print(data)
content_model(data)
'''

useLdAModel()

'''
data = readCSVRow("news-data.csv",0) #把文章内容提取出来
data1 = seg_stop_data(data) #把内容中不需要的部分去掉
writeCSVRow("news-data.csv","news-data1.csv",data1)#把内容分词,重新存储到另一张表 #

data = readCSVRow("news-data1.csv",2)
data = dealData1(data)
data_i=predict_data(data)#训练模型
print(data_i)
writeCSVRow("news-data1.csv","news-data2.csv",data_i)
'''