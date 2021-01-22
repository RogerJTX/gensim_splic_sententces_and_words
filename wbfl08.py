from gensim import corpora
import jieba
documents = ['工业互联网平台的核心技术是什么',
            '工业现场生产过程优化场景有哪些']
def word_cut(doc):
    seg = [jieba.lcut(w) for w in doc]
    return seg

texts= word_cut(documents)
print(texts)

##为语料库中出现的所有单词分配了一个唯一的整数id
dictionary = corpora.Dictionary(texts)


print(dictionary.token2id)


##该函数doc2bow()只计算每个不同单词的出现次数，将单词转换为整数单词id，并将结果作为稀疏向量返
bow_corpus = [dictionary.doc2bow(text) for text in texts]
print(bow_corpus)