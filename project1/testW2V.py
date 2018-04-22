# This Python file uses the following encoding: utf-8  m
import gc
import jieba
from gensim.models import Word2Vec
import logging,gensim,os
import csv
import multiprocessing
from gensim.models.word2vec import PathLineSentences


data_path="../data"
segment_path="segment_stop"
def segment_stop_content():
    stop_words = get_stop_words("stopword.txt")
    stop_words.add(' ')
    stop_words.add(' ')
    stop_words.add('\u3000')
    stop_words.add('\n')
    stop_words.add('2340')
    files=os.listdir(data_path)
    for file in files:
        file_path="%s/%s"%(data_path,file)
        print(file_path)
        try:
            f=open(file_path,mode='r')
            content=f.read();
        except:
            continue

        if not os.path.exists("%s/%s"%(data_path,segment_path)):
            os.mkdir("%s/%s"%(data_path,segment_path))

        try:
            output = open("%s/%s/%s"%(data_path,segment_path,file),"w",encoding="utf-8")
            content_seg = jieba.cut(content);
            seg_da = [item for item in content_seg if str(item) not in stop_words]
            print(seg_da)
            output.write(" ".join(seg_da ))
        except:
            continue
        finally:
            output.close();
def loadTrain():
    allfiles=[]
    files=os.listdir("%s/%s"%(data_path,segment_path))
    for file in files:
        f=open("%s/%s/%s"%(data_path,segment_path,file),mode="r")
        text=f.read();
        allfiles.append(text)
        #print(allfiles)
    return allfiles

def readCSVRow(path,n):
    with open(path,mode="r",encoding="GB18030") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[n] for row in reader]
    return column

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


#data是一个存储的多个字符串的list
#把data中的每一个字符串变成字符串数组（以空格分开）
def dealData(data): #
    list=[]
    for i in range(len(data)):
        list.append(data[i].split(" "))
    return list

def train(input_dir):
    gc.collect()
    #data = PathLineSentences(input_dir)
    model = gensim.models.Word2Vec(PathLineSentences(input_dir),workers=multiprocessing.cpu_count() * 2, sg=1)
    '''
    for i in range(len(data))[::100]:
        if i==0:
            continue
        tte = model.corpus_count + len(data[i:i+100])
        model.train(data[i:i+100],total_examples=tte,epochs=model.epochs)
    '''

    model.save("word2Vector.model")
    print('ok')



def useModel():
    model = Word2Vec.load("word2Vector.model")
    result = model.most_similar(positive=['中小学', '政治', '学生', '国家教委', '教育', '俱乐部', '史实', '公元前', '干部', '祖国'])
    for item in result:
        print(" "+item[0]+" 相似度："+str(item[1]))

'''
data = readCSVRow("news-data1.csv",2)
data = dealData(data)
train(data)
'''

'''
data = loadTrain()
data = seg_stop_data(data)
data = dealData(data)
print(len(data))
train(data)
'''
input_path="%s/%s"%(data_path,segment_path)
#segment_stop_content()
#train(input_path)
useModel()
