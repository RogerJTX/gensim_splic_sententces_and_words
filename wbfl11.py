# coding=utf-8
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


df = pd.read_csv('sucai01.csv')
df=df[['cat','review']]
print("数据总量: %d ." % len(df))
df.sample(5)

print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
# df[df.isnull().values==True]
df = df[pd.notnull(df['review'])]

d = {'cat': df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)

df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(8, 5))
plt.title("类目数量分布")
plt.ylabel('数量', fontsize=18)
plt.xlabel('类目', fontsize=18)
# plt.show()

df['cat_id'] = df['cat'].factorize()[0]
cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)
df.sample(5)
print(id_to_cat)


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 加载停用词
stopwords = stopwordslist("百度停用词表.txt")

df['cut_review'] = df['review'].apply(lambda x: ",,,".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df.head()
print(df.head())



def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white',
                          font_path='simhei.ttf',
                          max_words=50, max_font_size=40,
                          random_state=42
                          ).generate(str(tup))
    return wordcloud


cat_desc = dict()
for cat in cat_id_df.cat.values:
    text = df.loc[df['cat'] == cat, 'cut_review']
    text = (' '.join(map(str, text))).split(' ')
    cat_desc[cat] = text

fig, axes = plt.subplots(5, 2, figsize=(30, 38))     # fig, ax = plt.subplots(1,3),其中参数1和3分别代表子图的行数和列数，一共有 1x3 个子图像。函数返回一个figure图像和子图ax的array列表。fig, ax = plt.subplots(1,3,1),最后一个参数1代表第一个子图。
# print(axes)
k = 0
for i in range(5):
    for j in range(1):
        cat = id_to_cat[k]
        most100 = Counter(cat_desc[cat]).most_common(100)
        ax = axes[i, j]
        ax.imshow(generate_wordcloud(most100), interpolation="bilinear")
        ax.axis('off')
        ax.set_title("{} Top 100".format(cat), fontsize=30)
        k += 1
        # print(k)
# plt.show()

tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(df.cut_review)
labels = df.cat_id
print(features.shape)
print('-----------------------------')
print(features)


print('-----------------------------------------------------------------------------------')
N = 2
for cat, cat_id in sorted(cat_to_id.items()):
    features_chi2 = chi2(features, labels == cat_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(cat))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))


print('-----------------------------------------------------------------------------------')
X_train, X_test, y_train, y_test = train_test_split(df['cut_review'], df['cat_id'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



clf = MultinomialNB().fit(X_train_tfidf, y_train)

y_predict = clf.predict(X_train_tfidf)
print(f1_score(y_predict, y_train, average='micro'))

def myPredict(sec):
    format_sec=" ".join([w for w in list(jb.cut(sec)) if w not in stopwords])
    pred_cat_id=clf.predict(count_vect.transform([format_sec]))

    print(id_to_cat[pred_cat_id[0]])

print(myPredict('1. Infrared non-planar plasmonic perfect absorber for enhanced sensitive refractive index sensing. OPTICAL MATERIALS, 2016, 53: 195-200'))
print(myPredict('1981.09. – 1986.07. 北京师范大学 化学系，本科；'))
print(myPredict('1994.09． – 1998.06. 中国地质大学 地球化学，研究生（博）；'))
print(myPredict('超材料技术、超晶格与红外波调控、微波控制材料、磁光电器件技术'))
print(myPredict('2005年至2009年 华中科技大学电子科学与技术专业本科生；'))
print('-----------------------------------------------------------------------------------')
print(myPredict('13.一种适用于复杂曲面加工的夹具及其使用方法，发明人：吴志刚、吴康江佳俊，申请号：2017101580440'))
print(myPredict('微波介质材料及其相关通信器件；'))

print('-----------------------------------------------------------------------------------')
print(myPredict('1994-1996：德国杜伊斯堡大学访问教授'))
print(myPredict('1. Luo, H., Chen, F., Wang, X.*, Dai, W., Xiong, Y., Yang, J., & Gong, R. (2019). A novel two-layer honeycomb sandwich structure absorber with high-performance microwave absorption. Composites Part A: Applied Science and Manufacturing, 119, 1-7.'))
