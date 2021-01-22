import jieba
import gensim
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
import re


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
    test_text = [w for w in list(jieba.cut(str1))]

    inferred_vector_dm = model_dm.infer_vector(test_text)  ##得到文本的向量
    # print(inferred_vector_dm)

    return inferred_vector_dm

def myPredict(sec):
    format_sec = " ".join([w for w in list(jieba.cut(remove_punctuation(sec))) if w not in stopwords])
    ceshi(format_sec)


doc_2_vec = ceshi('2018年03月23日晚上大概十一点多钟我和张三骑着摩托车从住处出门想看看有什么能吃的东西.')
print(doc_2_vec)
