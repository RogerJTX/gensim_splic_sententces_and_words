import jieba
import jieba.posseg as pseg
from gensim import corpora, models, similarities


def StopWordsList(filepath):
    wlst = [w.strip() for w in open(filepath, 'r', encoding='utf8').readlines()]
    return wlst


def seg_sentence(sentence, stop_words):
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']#过滤数字m
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    sentence_seged = pseg.cut(sentence)
    # sentence_seged = set(sentence_seged)
    outstr = []
    for word, flag in sentence_seged:
        # if word not in stop_words:
        if word not in stop_words and flag not in stop_flag:
            outstr.append(word)
    return outstr


if __name__ == '__main__':
    spPath = 'stopwords.txt'
    tpath = 'test.txt'

    stop_words = StopWordsList(spPath)
    keyword = '吃鸡'
