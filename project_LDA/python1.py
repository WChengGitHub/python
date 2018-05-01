# This Python file uses the following encoding: utf-8  m
import os
import csv
import jieba
import  numpy as np
import gc

from gensim import corpora, models, similarities

data_path="D:/Code/data"

def get_data(path):
    print("get_data_start")
    data = []
    dirs = os.listdir(path)
    for dir in dirs:
        no_dir=["csv","segment","计算机","model"]
        if dir in no_dir:
            continue

        #print(dir)
        dir_path = "%s/%s"%(path,dir)
        files = os.listdir(dir_path)
        for file in files:
            data_row = []
            file_path = "%s/%s/%s"%(path,dir,file)
            f = open(file_path,mode='r',encoding="GB18030")
            try:
                content = f.read()
                seg_stop_content = seg_stop_data(content)
                #print(seg_stop_content)
                write_data(file,seg_stop_content)
            except Exception:
                continue
            finally:
                f.close()

            data_row.append(file)
            data_row.append(dir)
            #print(dir)
            data.append(data_row)
    print("get_data_end")
    return data

def get_stop_words(path):
    with open(path,"r",encoding="utf-8") as file:
        return set([line.strip() for line in file])

def seg_stop_data(data):
    list = []
    #stop_data=["i","Internet","x","Web","t","Computer", "k" ,"Proceedings","V" ,"y","W" ,"Systems","j","v", "K"  ,"In"  ,"k" , "int","Microsoft","time" , "τ","S" , "k" , "≤","systems", "r","∈"  ,"Agent" , "Windows","network" ,"′" , "based" ,"z", "I","x" ,  "β",  "k","Visual" , "f" , "″",  "≠","WWW"  ,"i" , "t"  ,"TCP" ,"w" ,"Information","q" , "Σ","f","IP"  ,"∈", "′" ,"i" ,"CA" , "Φ" ]
    stop_words = get_stop_words("stopword.txt")
    stop_words.add(' ')
    stop_words.add(' ')
    stop_words.add('\u3000')
    stop_words.add('\n')
    stop_words.add('2340')
    #for d in stop_data:
        #stop_words.add(d)
    seg_d = jieba.cut(data)
    seg_da =[item for item in seg_d if str(item) not in stop_words]
    return seg_da

def write_data(file,data):
    if not os.path.exists("%s/%s" % (data_path, "segment")):
        os.mkdir("%s/%s" % (data_path, "segment"))
    #print("write")
    output = open("%s/%s/%s"%(data_path,"segment",file),mode="w",encoding="GB18030")
    output.write(" ".join(data))
    output.close()

def put_data_to_csv(data,path):
    if not os.path.exists("%s/%s"%(path,"csv")):
        os.mkdir("%s/%s"%(path,"csv"))

    csv_path="%s/%s/%s"%(path,"csv","news.csv")
    with open(csv_path,mode="w",newline='',encoding="GB18030") as f:
        writer = csv.writer(f)
        for i in range(len(data)):
            writer.writerow(data[i])

def get_train_data():
    print("get_train_data start")
    path="%s/%s/%s"%(data_path,"csv","news.csv")
    data = readCSVRow(path,0)
    list=[]
    for file in data:
        file_path="%s/%s/%s"%(data_path,"segment",file)
        f = open(file_path,mode="r",encoding="GB18030")
        #print(file_path)
        content = f.read()
        #print(file)
        #print(content.split(" "))
        list.append(content.split(" "))
    print("get_train_data end")
    return list

def readCSVRow(path, n):
        with open(path, mode="r", encoding="GB18030") as csvfile:
            reader = csv.reader(csvfile)
            column = [row[n] for row in reader]
        return column


def readCSV(path):
    print("readCSV start")
    list = []
    with open(path, mode="r", encoding="GB18030") as csvfile:
        rows = csv.reader(csvfile)
        for i,row in enumerate(rows):
            list.append(row)

    print("readCSV end")
    return list



def content_model(data):
    print("content_model start")
    dic = corpora.Dictionary(data)
    dic.save("dic.m")
    corpus = [dic.doc2bow(text) for text in data]
    corpora.MmCorpus.serialize("corpus.m",corpus)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.LdaModel(corpus_tfidf,id2word=dic,num_topics=20,alpha=1)
    lda.save("ldaModel.model")
    print("content_model end")

def predict_data(data):
    print("predict_start")
    dic = corpora.Dictionary.load("dic.m")
    corpus = [dic.doc2bow(text) for text in data]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("ldaModel.model")
    list=[]
    for i in range(len(data)):
        test_doc = data[i]  # 查看训练集中第三个样本的主题分布
        doc_bow = dic.doc2bow(test_doc)  # 文档转换成bow
        doc_lda = lda[doc_bow]  # 得到新文档的主题分布
        # 输出新文档的主题分布
        #print(doc_lda)
        li=[]
        for j in range(len(doc_lda)):
            li.append(doc_lda[j][1])

        topic = np.asarray(li).argmax()
        #print(topic)
        #print(sorted(li,reverse=True))
        list.append(topic)
    print("predict_end")
    return list

def writeCSVRow(path1,path2,data):
    print("writeCSVRow start")
    with open(path1,mode="r",encoding="GB18030") as csvfile:
        rows = csv.reader(csvfile)
        with open(path2,'w',newline='',encoding="GB18030") as f:
            writer = csv.writer(f)
            for i,row in enumerate(rows):
                #print(data[i])
                row.append(data[i])
                writer.writerow(row)
    print("writeCSVRow end")

def show_topic_words():
    dic = corpora.Dictionary.load("dic.m")
    corpus = corpora.MmCorpus("corpus.m")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("ldaModel.model")
    data = lda.print_topics(20)
    for i in range(len(data)):
        list= []
        adata = data[i][1].split("+")
        for j in range(len(adata)):
            aadata = adata[j].split("*")
            list.append(aadata[1])
        print("topic %s"%(i))
        print(" ".join(list))


def get_topic_words(base_path,n):
    datas=[]
    path="%s/%s"%(base_path,n)
    dic = corpora.Dictionary.load("%s/%s"%(path,"dic.m"))
    corpus = corpora.MmCorpus("%s/%s"%(path,"corpus.m"))
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("%s/%s"%(path,"ldaModel.model"))
    data = lda.print_topics(n)
    for i in range(len(data)):
        list= []
        adata = data[i][1].split("+")
        for j in range(len(adata)):
            aadata = adata[j].split("*")
            list.append(aadata[1])
        datas.append("topic %s"%(i))
        datas.append(" ".join(list))
    return datas

def statics_all():
    base_path="D:/Code/data/model"
    for i in range(51)[10:]:
        path2 = "%s/%s/%s/%s" % (data_path, "csv", i, "news1.csv")
        data = readCSV(path2)
        datas=statics(data,base_path,i)
        write(datas[0],i,"topic-word.txt")
        write(datas[1], i, "statics.txt")

def statics(data,base_path,n):
    print("statics start")
    datas=get_topic_words(base_path,n)
    datas1=[]
    dic = {}
    topics=[]
    for da in data:
        key = "%s"%(da[1])
        if not key in dic:
            dic[key] = 1
            topics.append(da[1])
        else:
            dic[key]=dic[key]+1
    #for (k,v) in dic.items():
        #print("dic[%s]=%s"%(k,v))

    for topic in topics:
         dicI={}
         list=[]
         for da in data:
            if da[1] == topic:
                if not da[2] in dicI:
                    dicI[da[2]] =1
                else:
                    dicI[da[2]] = dicI[da[2]] + 1
         keys = dicI.keys();
         for k in sort(keys):
             v=dicI.get(k)
             s="topic %s : %.2f%%    "%(k,v/dic[topic]*100)
             list.append(s)

         #print("%s %s"%(topic,dic[topic]))
         datas1.append("%s %s"%(topic,dic[topic]))
         datas1.append(list)
         #print(list)
    print("statics start")
    return datas,datas1


def write(datas,n,title):
    base_path="D:/Code/data/csv"
    path="%s/%s"%(base_path,n)

    if not os.path.exists(path):
        os.mkdir(path)

    with open("%s/%s"%(path,title),mode="w",encoding="utf-8") as f:
        for da in datas:
            #print(da)
            f.write(str(da)+"\n\n")



    '''
    print(data[0][1])
    adata = data[0][1].split("+")
    print(adata[0])
    aadata = adata[0].split("*")
    print(aadata[1])
    '''
def sort(keys):
    list=[]
    for k in keys:
        list.append(int(k))
    list = sorted(list)
    list1=[]
    for i in range(len(list)):
        list1.append(str(list[i]))
    return list1


def content_model1(data,base_path,n):
    print("content_model1 start")

    path="%s/%s"%(base_path,n)
    if not os.path.exists(path):
        os.mkdir(path)

    dic = corpora.Dictionary(data)
    dic.save("%s/%s"%(path,"dic.m"))
    corpus = [dic.doc2bow(text) for text in data]
    corpora.MmCorpus.serialize("%s/%s"%(path,"corpus.m"),corpus)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    del data
    del corpus
    del tfidf
    gc.collect()

    lda = models.LdaModel(corpus_tfidf,id2word=dic,num_topics=n,alpha=1)
    lda.save("%s/%s"%(path,"ldaModel.model"))
    gc.collect()
    print("content_model1 end")

def trian_models(data):
    print("train_models begin")
    bath_path="D:/Code/data/model"
    if not os.path.exists(bath_path):
        os.mkdir(bath_path)
    for i in (range(51))[45:]:
        content_model1(data,bath_path,i)
    print("train_models end")

def predict_data1s(data,base_path):

    for i in (range(45))[10:]:
        data_i=[]
        data_i=predict_data1(data, base_path, i)
        write_to_csv_n(data_i,i)



def predict_data1(data,base_path,n):
    path = "%s/%s" % (base_path, n)
    print("predict_start")
    dic = corpora.Dictionary.load("%s/%s"%(path,"dic.m"))
    corpus = [dic.doc2bow(text) for text in data]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("%s/%s"%(path,"ldaModel.model"))
    list=[]
    for i in range(len(data)):
        test_doc = data[i]  # 查看训练集中第三个样本的主题分布
        doc_bow = dic.doc2bow(test_doc)  # 文档转换成bow
        doc_lda = lda[doc_bow]  # 得到新文档的主题分布
        # 输出新文档的主题分布
        #print(doc_lda)
        li=[]
        for j in range(len(doc_lda)):
            li.append(doc_lda[j][1])

        topic = np.asarray(li).argmax()
        #print(topic)
        #print(sorted(li,reverse=True))
        list.append(topic)
    print("predict_end")
    return list

def write_to_csv_n(data,n):
    path1 = "%s/%s/%s" % (data_path, "csv", "news.csv")
    base_path = "%s/%s/%s" % (data_path, "csv",n)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path2 = "%s/%s" % (base_path, "news1.csv")
    writeCSVRow(path1, path2, data)


'''
data = get_data(data_path)
#print(data[0])
put_data_to_csv(data,data_path)
data = get_train_data()
content_model(data)
'''
'''
data = get_train_data()
data_i = predict_data(data)
print(data_i)

path1 = "%s/%s/%s"%(data_path,"csv","news.csv")
path2 = "%s/%s/%s"%(data_path,"csv","news1.csv")
writeCSVRow(path1,path2,data_i)
#show_topic_words()

path2 = "%s/%s/%s"%(data_path,"csv","news1.csv")
data = readCSV(path2)
statics(data)
'''

#data = get_train_data()
#trian_models(data)
#base_path="D:/Code/data/model"
#n=10
data_i=predict_data1(data,base_path,n)
#write_to_csv_n(data_i,10)


'''
path2 = "%s/%s/%s/%s"%(data_path,"csv","10","news1.csv")
data = readCSV(path2)
data = statics(data,"D:/Code/data/model",10)
write(data,10)
'''


data = get_train_data()
base_path="D:/Code/data/model"
predict_data1s(data,base_path)

statics_all()