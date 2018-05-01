# This Python file uses the following encoding: utf-8  m
import os
import csv
import jieba
import  numpy as np
import gc

from gensim import corpora, models, similarities

data_path="D:/Code/data"
def get_data(path,types,num):

    print("get_data_start")
    data = []
    dirs = os.listdir(path)
    types_n=0

    for dir in dirs:
        no_dir=["csv","segment","model"]
        if dir in no_dir:
            continue
        if types_n == types:
            break;
        if not dir in ["政治","经济"]:
            continue
        print(dir)
        types_n = types_n +1
        #print(dir)
        dir_path = "%s/%s"%(path,dir)
        files = os.listdir(dir_path)
        num_n = 0
        for file in files:
            if num_n == num:
                break
            num_n = num_n + 1
            data_row = []
            file_path = "%s/%s/%s"%(path,dir,file)
            f = open(file_path,mode='r',encoding="GB18030")
            try:
                content = f.read()
                seg_stop_content = seg_stop_data(content)
                #print(seg_stop_content)
                write_data(file,seg_stop_content,types,num)
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
    print("get_stop_words start")
    with open(path,"r",encoding="utf-8") as file:
        return set([line.strip() for line in file])
    print("get_stop_words end")

def seg_stop_data(data):
    list = []
    #stop_data=["i","Internet","x","Web","t","Computer", "k" ,"Proceedings","V" ,"y","W" ,"Systems","j","v", "K"  ,"In"  ,"k" , "int","Microsoft","time" , "τ","S" , "k" , "≤","systems", "r","∈"  ,"Agent" , "Windows","network" ,"′" , "based" ,"z", "I","x" ,  "β",  "k","Visual" , "f" , "″",  "≠","WWW"  ,"i" , "t"  ,"TCP" ,"w" ,"Information","q" , "Σ","f","IP"  ,"∈", "′" ,"i" ,"CA" , "Φ" ]
    stop_words = get_stop_words("stopword.txt")
    stop_words.add(' ')
    stop_words.add(' ')
    stop_words.add("◆")
    stop_words.add('\u3000')
    stop_words.add('\n')
    stop_words.add('2340')
    #for d in stop_data:
        #stop_words.add(d)
    seg_d = jieba.cut(data)
    seg_da =[item for item in seg_d if str(item) not in stop_words]
    return seg_da

def write_data(file,data,types,num):
    if not os.path.exists("%s/%s" % (data_path, "segment")):
        os.mkdir("%s/%s" % (data_path, "segment"))
    if not os.path.exists("%s/%s/%s-%s" % (data_path, "segment",types,num)):
        os.mkdir("%s/%s/%s-%s" % (data_path, "segment",types,num))
    #print("write")
    output = open("%s/%s/%s"%(data_path,"%s/%s-%s"%("segment",types,num),file),mode="w",encoding="GB18030")
    output.write(" ".join(data))
    output.close()

def put_data_to_csv(data, path,types,num):
    print("put_data_to_csv start")
    if not os.path.exists("%s/%s" % (path, "csv")):
        os.mkdir("%s/%s" % (path, "csv"))

    if not os.path.exists("%s/%s/%s-%s" % (path, "csv",types,num)):
        os.mkdir("%s/%s/%s-%s" % (path, "csv",types,num))

    csv_path = "%s/%s/%s-%s/%s" % (path, "csv", types,num,"%s%s-%s%s"%("news",types,num,".csv"))
    with open(csv_path, mode="w", newline='', encoding="GB18030") as f:
        writer = csv.writer(f)
        for i in range(len(data)):
            writer.writerow(data[i])

    print("put_data_to_csv end")

def get_train_data(path,types,num):
    print("get_train_data start")
    csv_path = "%s/%s/%s-%s/%s" % (path, "csv", types, num, "%s%s-%s%s" % ("news", types, num, ".csv"))
    data = readCSVRow(csv_path,0)
    list=[]
    for file in data:
        file_path="%s/%s/%s"%(data_path,"%s/%s-%s"%("segment",types,num),file)
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

def content_model1(data,path,types,num):
    print("content_model1 start")

    path1="%s/%s"%(path,"model")
    if not os.path.exists(path1):
        os.mkdir(path1)

    path2 = "%s/%s/%s-%s" % (path, "model",types,num)
    if not os.path.exists(path2):
        os.mkdir(path2)

    dic = corpora.Dictionary(data)
    dic.save("%s/%s"%(path2,"dic.m"))
    corpus = [dic.doc2bow(text) for text in data]
    corpora.MmCorpus.serialize("%s/%s"%(path2,"corpus.m"),corpus)
    #tfidf = models.TfidfModel(corpus)
    #corpus_tfidf = tfidf[corpus]

    np.random.seed(1)
    lda = models.LdaModel(corpus,id2word=dic,num_topics=types,iterations=500)
    lda.save("%s/%s"%(path2,"ldaModel.model"))

    print("content_model1 end")

def predict_data1(data,path,types,num):
    path = path2 = "%s/%s/%s-%s" % (path, "model",types,num)
    print("predict_start")
    dic = corpora.Dictionary.load("%s/%s"%(path,"dic.m"))
    corpus = [dic.doc2bow(text) for text in data]
    #tfidf = models.TfidfModel(corpus)
    #corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("%s/%s"%(path,"ldaModel.model"))
    list=[]
    for i in range(len(data)):
        test_doc = data[i]  # 查看训练集中第三个样本的主题分布
        doc_bow = dic.doc2bow(test_doc)  # 文档转换成bow

        doc_topics, word_topics, phi_values = lda.get_document_topics(doc_bow, per_word_topics=True)
        print(doc_topics)
        # 输出新文档的主题分布
        li=[]
        for j in range(len(doc_topics)):
            li.append(doc_topics[j][1])
        if(len(li) != 1):
            topic = np.asarray(li).argmax()
        else:
            topic = doc_topics[0][0]
        print(topic)
        #print(topic)
        #print(sorted(li,reverse=True))
        list.append(topic)
    print("predict_end")
    return list

def write_to_csv_n(path,data,types,num):
    csv_path = "%s/%s/%s-%s/%s" % (path, "csv", types, num, "%s%s-%s%s" % ("news", types, num, ".csv"))
    csv_path_train="%s/%s/%s-%s/%s" % (path, "csv", types, num, "%s-%s%s-%s%s" % ("trians","news", types, num, ".csv"))
    writeCSVRow(csv_path , csv_path_train, data)

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

def get_topic_words(path,types,num):
    datas=[]
    path1="%s/%s/%s-%s" % (path, "model",types,num)
    dic = corpora.Dictionary.load("%s/%s"%(path1,"dic.m"))
    corpus = corpora.MmCorpus("%s/%s"%(path1,"corpus.m"))
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel.load("%s/%s"%(path1,"ldaModel.model"))
    data = lda.show_topics()
    print(data)
    for i in range(len(data)):
        list= []
        adata = data[i][1].split("+")
        for j in range(len(adata)):
            aadata = adata[j].split("*")
            list.append(aadata[1])
        datas.append("topic %s"%(i))
        datas.append(" ".join(list))
    return datas

def statics(data,path,types,num):
    print("statics start")
    datas=get_topic_words(path,types,num)
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

def sort(keys):
    list=[]
    for k in keys:
        list.append(int(k))
    list = sorted(list)
    list1=[]
    for i in range(len(list)):
        list1.append(str(list[i]))
    return list1


def write(datas,path,types,num,title):
    path1 = "%s/%s/%s-%s" % (path, "csv", types, num)
    if not os.path.exists(path1):
        os.mkdir(path1)

    with open("%s/%s"%(path1,title),mode="w",encoding="utf-8") as f:
        for da in datas:
            #print(da)
            f.write(str(da)+"\n\n")

def readCSV(path,types,num):
    csv_path_train = "%s/%s/%s-%s/%s" % (path, "csv", types, num, "%s-%s%s-%s%s" % ("trians", "news", types, num, ".csv"))
    print("readCSV start")
    list = []
    with open(csv_path_train, mode="r", encoding="GB18030") as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows):
            list.append(row)

    print("readCSV end")
    return list

types=2
num = 2000
data = get_data(data_path,types,num)
put_data_to_csv(data,data_path,types,num)


data = get_train_data(data_path,types,num)
content_model1(data,data_path,types,num)



data = get_train_data(data_path, types, num)
data_i = predict_data1(data,data_path,types,num)
write_to_csv_n(data_path,data_i,types,num)



data = readCSV(data_path,types,num)
data = statics(data,data_path,types,num)
write(data[0],data_path,types,num,"topic-word.txt")
write(data[1],data_path,types,num,"statics.txt")
