# coding:utf-8
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
import jieba
from gensim.corpora import WikiCorpus
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec



model_file_name = 'wiki.model'
new_model = gensim.models.Word2Vec.load(model_file_name)  # 调用模型
sim_words = new_model.wv.most_similar(positive=['女孩'])
for word, similarity in sim_words:
    print(word, similarity)  # 输出’女人‘相近的词语和概率
print(new_model.wv['女孩'])  # 输出’女孩‘的词向量




