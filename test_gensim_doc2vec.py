import jieba
import gensim
from gensim.models.doc2vec import Doc2Vec
import multiprocessing

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_datasest():
    docs_list = []
    x_train = []
    c = 0
    d = 0
    with open("test_yuliao_news.txt", 'r', encoding='gbk') as cf:
        for line in cf:
            if line == '\n' or line == '\r':
                pass
            else:
                each_doc = line.replace('\n', '')
                # print(line.replace('\n', ''))
                docs_list.append(each_doc)
                c += 1
                d += 1

                print(c)
                if c % 100000 == 0:

                    for i, text in enumerate(docs_list):
                        word_list = text.split(' ')
                        l = len(word_list)
                        word_list[l - 1] = word_list[l - 1].strip()
                        document = TaggededDocument(word_list, tags=[i])
                        print(document)
                        x_train.append(document)
                        print(d)


                    docs_list = []
                    c = 0
    return x_train


def train(x_train, size=100, epoch_num=1):  ##size 是你最终训练出的句子向量的维度，自己尝试着修改一下

    model_dm = Doc2Vec(x_train, min_count=3, window=5, size=size, sample=1e-3, negative=5, workers=multiprocessing.cpu_count())
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=10)
    model_dm.save('model_doc2vec_11.22')  ##模型保存的位置

    return model_dm


def ceshi(str1):
    model_dm = Doc2Vec.load("model_doc2vec_11.22")
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    test_text = [w for w in list(jieba.cut(str1))]

    inferred_vector_dm = model_dm.infer_vector(test_text)  ##得到文本的向量
    print(inferred_vector_dm)

    return inferred_vector_dm


if __name__ == '__main__':
    x_train = get_datasest()
    model_dm = train(x_train)

    doc_2_vec = ceshi('2018年03月23日晚上大概十一点多钟我和张三骑着摩托车从住处出门想看看有什么能吃的东西.')
    print(type(doc_2_vec))
    print(doc_2_vec.shape)