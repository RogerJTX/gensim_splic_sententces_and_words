import jieba
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from bs4 import BeautifulSoup
import logging
import pymongo
import base64
from goose3 import Goose
from goose3.text import StopWordsChinese
import urllib
import time, requests
import datetime, random
import chardet
import pandas as pd
import matplotlib
import jieba as jb
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import chardet
import urllib.request
# from boilerpipe.extract import Extractor
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.cluster import KMeans

import operator
from functools import reduce
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn import datasets
from sklearn.cluster import DBSCAN
import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图




# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub(' ', line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 加载停用词
stopwords = stopwordslist("D:\Python\python_code\Liangzhi\TianPengTrans-tmp\etl\pytorch\百度停用词表.txt")

def ceshi(str1):
    model_dm = Doc2Vec.load("model_doc2vec_result")
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    test_text = [w for w in list(jieba.cut(remove_punctuation(str1))) if w not in stopwords]

    inferred_vector_dm = model_dm.infer_vector(test_text)  ##得到文本的向量
    # print(inferred_vector_dm)

    return inferred_vector_dm

def myPredict(sec):
    format_sec = " ".join([w for w in list(jieba.cut(remove_punctuation(sec))) if w not in stopwords])
    ceshi(format_sec)


doc_2_vec = ceshi('2018年03月23日晚上大概十一点多钟我和张三骑着摩托车从住处出门想看看有什么能吃的东西.')
print(doc_2_vec)
print(type(doc_2_vec))


pca = PCA(n_components=2)
output = pca.fit_transform((doc_2_vec).reshape(-1,2))
print(output)


X = output
db = skc.DBSCAN(eps=0.03, min_samples=3).fit(X) #DBSCAN聚类方法 还有参数，matric = ""距离计算方法
labels = db.labels_  #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

print('每个样本的簇标号:')
print(labels)

raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('噪声比:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels)) #轮廓系数评价聚类的好坏

for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = X[labels == i]
    print(one_cluster)
    plt.plot(one_cluster[:,0],one_cluster[:,1],'o')

plt.show()

