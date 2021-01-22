import multiprocessing
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
import numpy as np

inp='123456.txt'
sents = TaggedLineDocument(inp) #直接对文本标号 （大数据适用，但是这样后续不能查看对应的内容，需要其他方法查询）
for k,v in sents:
    print(k, v)
model = Doc2Vec(sents, size = 200, window = 8, alpha = 0.015)
outp1='docmodel'
model.save(outp1)

model=Doc2Vec.load(outp1)
sims = model.docvecs.most_similar(45)#0代表第一个句子或段落
print(sims)

#如果有两个句子
doc_words1=['验证','失败','验证码','未','收到']
doc_words2=['今天','奖励','有','哪些','呢']
doc_words3=['姓名','张三']
#转换为向量：
invec1 = model.infer_vector(doc_words1, alpha=0.1, min_alpha=0.0001, steps=5)
# print(invec1)
invec3 = model.infer_vector(doc_words3, alpha=0.1, min_alpha=0.0001, steps=5)

sims1 = model.docvecs.most_similar([invec3])#计算训练模型中与句子1相似的内容
print(sims1)
for word, similarity in sims1:
    print(word, similarity)

# print(model.docvecs.similarity(0,1086620))#计算句子的相似度（0和1086620为句子的标号）
# 打印结果相似度位： 0.9385169567251749








'''
模型参数说明：
1.dm=1 PV-DM  dm=0 PV-DBOW。
2.size 所得向量的维度。 
3.window 上下文词语离当前词语的最大距离。
4.alpha 初始学习率，在训练中会下降到min_alpha。
5.min_count 词频小于min_count的词会被忽略。
6.max_vocab_size 最大词汇表size，每一百万词会需要1GB的内存，默认没有限制。
7.sample 下采样比例。
8.iter 在整个语料上的迭代次数(epochs)，推荐10到20。
9.hs=1 hierarchical softmax ，hs=0(default) negative sampling。
10.dm_mean=0(default) 上下文向量取综合，dm_mean=1 上下文向量取均值。
11.dbow_words:1训练词向量，0只训练doc向量。

'''