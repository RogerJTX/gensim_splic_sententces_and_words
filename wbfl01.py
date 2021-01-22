# coding=utf-8

import torch#基本的torch函数
import torch.autograd as autograd#自动求导
import torch.nn as nn#神经网络类都在这个里面
import torch.nn.functional as F#几乎所有的激励函数
import torch.optim as optim
import numpy
from gensim.models.word2vec import Word2Vec
import gensim



# x = torch.randn((3,4,5))
# print(x)

# x1 = torch.FloatTensor([2])


# 读取数据，用gensim中的word2vec训练词向量
file = open('wiki.txt', encoding='utf-8')
sss=[]
while True:
    ss=file.readline().replace('\n', '').rstrip()
    if ss=='':
        break
    s1=ss.split(" ")
    sss.append(s1)
file.close()
model = Word2Vec(size=200, workers=5,sg=1)  # 生成词向量为200维，考虑上下5个单词共10个单词，采用sg=1的方法也就是skip-gram
model.build_vocab(sss)
model.train(sss,total_examples = model.corpus_count,epochs = model.iter)
model.save('./data/gensim_w2v_sg0_model')            # 保存模型
new_model = gensim.models.Word2Vec.load('w2v_model') # 调用模型
sim_words = new_model.most_similar(positive=['女人'])
for word,similarity in sim_words:
    print(word,similarity)                           # 输出’女人‘相近的词语和概率
print(model['女孩'])                                 # 输出’女孩‘的词向量
