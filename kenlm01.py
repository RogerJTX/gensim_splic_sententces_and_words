import kenlm
import math


print(dir(kenlm))

model = kenlm.Model('test_result_o5_big.arpa')
a = model.score('this is a sentence .', bos = True, eos = True)
print('1=',model.score('this is a sentence .', bos = True, eos = True))
print(type(a))
print(int(a))
print('2=',model.score('this do i you apple', bos = True, eos = True))
b = model.score('一起 吃饭', bos = True, eos = True)
print('3=',model.score('一起 吃饭', bos = True, eos = True))
print('4=',model.score('hello', bos = True, eos = True))
print('5=',model.score('获 中国电 子学会 优 秀博士学位论文提名奖', bos = True, eos = True))

print(model.score('1991 - 1998 北京邮电大学信 息工程系 本科/硕士', bos = True, eos = True))
c = model.score('1991 - 198 本科/硕士 北9学信 京邮 电大 息工程系 ', bos = True, eos = True)
print(model.score('1991 - 198 本科/硕士 北9学信 京邮 电大 息工程系 ', bos = True, eos = True))

print(model.score('爱说大话 是 可能 我喜欢 1998 吃里爬外 阿克苏', bos = True, eos = True))
print(model.score('记者获悉,首日客流量就突破 了120000人次,创新历史新高, 苏鲜生波士顿龙虾单日售出超过9000斤', bos = True, eos = True))
print(model.score('购物车里都装了些什么? 由于小店地处B1层, 正逢周五地铁 高峰日,客流十分大。当天最高峰, 小店在店人数达到50人,还有不少市民特地从 金山、 奉贤跑来抢购, 抢购价 格十分优惠的米面粮油、水果蔬菜、饮料酒水,多款爆品一度售空!当天,为了满足供应需求, 五角场苏宁易购广 场也不得不开启疯狂 补货模式。', bos = True, eos = True))
print(model.score('i love eating apple', bos = True, eos = True))
d1 = model.score('i love eating apple', bos = True, eos = True)
print(model.score('我们 去 北京 天安门', bos = True, eos = True))

print('-------------------------------------------------------')
print(model.perplexity('i love eating apple'))
e1 = model.perplexity('i love eating apple')
# print('d1*e1='+str(int(d1)*int(e1)))
print(model.perplexity('i you eating jake ball'))
print(model.perplexity('We are finally getting better at predicting organized conflict'))
print(model.perplexity('爱说大话 是 可能 我喜欢 1998 吃里爬外 阿克苏'))
print(model.perplexity('获 中国电 子学会优 秀博士学位论文提名奖'))
print(model.perplexity('1991 - 1998 北京邮电大学信 息工程系 本科/硕士'))
print(model.perplexity('1991 - 198 本科/硕士 北9学信 京邮 电大 息工程系 '))


print('-------------------------------------------------------')
print(math.pow(10, a))
print(math.pow(10, b))
print(math.pow(10, c))

print('-------------------------------------------------------')

print('1',model.score('2003.10 — 2007.3 任校团委副书记'))
print('1',model.score('1998.9 — 2001.5 历任校团委文体部部长、宣传部部长、实践部部长'))
print('1',model.score('陈超于2010年9月前往 法国 巴黎索邦大学 （原巴黎第六大学，世界排名35）攻读博士学位。博士期间主要从事大数据挖掘以及在智慧城市相关的应用研究。2014年9月毕业后以副教授加盟重庆大学计算机学院，这是近几年来学院首次以副教授职称引进应届毕业生。'))
print('1',model.score('1991 - 1998 北京邮电大学信 息工程系 本科/硕士'))
print('1',model.score(' Chao Chen , Yan Ding, et al. , Yan Ding, et al. IEEE Systems Journal , 2019. '))
print('1',model.score('1、  实时软件工程'))
print('1',model.score('重庆市基础与前沿研究计划项目合同（ 基于大数据挖掘的结构健康状态分析与评估研究， ，主持'))
print('1',model.score('重庆市重点自然科学基金项目， CSTC2011BA6026 ，大型在役桥梁基于生命感知的安全评估基础理论研究， ，参加'))
print('1',model.score('日本文部科学省科研项目， 21650050 ，ファジィ量子計算に基づく画像の量子表現とその超高速画像圧縮への応用に関する研究 （基于模糊量子计算的图像表示及其超高速图像压缩应用研究）， 2009/01 月 -2010/12 ，参加'))
print('2',model.score('爱说大话 是 可能 我喜欢 1998 吃里爬外 阿克苏'))
print('2',model.score('1991 - 198 本科/硕士 北9学信 京邮 电大 息工程系 '))
print('2',model.score('沿研究计划项 重庆市 基于大数据挖掘的结构'))
print('2',model.score('历任校团委文体部部长 2003.10 实践部部长'))
print('2',model.score('基于大数据挖掘的结构健康状态分析与评估研究， ，主持 中央高校基本科研业务费科研专项项目（'))
print('2',model.score('年度招收硕士3名，招收数学，计算机等相关专业。 2 个人简介： 重庆大学计算机学院，副教授，硕士导师。2000年重庆大学计算机学院本科毕业留校任教至今。2000年10月在日本电气通信大学短期交换留学一年；2003年在重庆大学计算机学院获得硕士学位；2007年公派至日本东京工业大学留学攻读博士学位，师从计算智能领域著名学者广田薰（Kaoru Hirota）教授'))
print('2',model.score('年度招收硕士3名，招收数学，计算机等相关专业。 重庆大学计算机学院，副教授，硕士导师。2000年重庆大学计算机学院本科毕业留校任教至今。2000年10月在日本电气通信大学短期交换留学一年；2003年在重庆大学计算机学院获得硕士学位；2007年公派至日本东京工业大学留学攻读博士学位，师从计算智能领域著名学者广田薰（Kaoru Hirota）教授'))


p1 = model.score('重 庆 市 基 础 与 前 沿 计 划 项 目 合 同')
p2 = model.score('重 庆 市 基 础 与')
p3 = model.score('前 沿 计 划 项 目 合 同')

p = p1 - p2 - p3
print("prob1: {} \t prob2: {} \t prob3: {} \t prob: {}".format(p1, p2, p3, p))

p1 = model.score('“ Dream ”')
p2 = model.score('据 说 是')
p3 = model.score('可 能 有')

print("prob1: {} \t prob2: {} \t prob3: {} \t prob: {}".format(p1, p2, p3, p))

p1 = model.score('肥 宅 快 乐 水')
p2 = model.score('肥 宅 快 乐 鸡 翅')
p3 = model.score('肥 宅 快 乐 埃 里 克 京 东 方 哈 迪 斯')

print("prob1: {} \t prob2: {} \t prob3: {} \t prob: {}".format(p1, p2, p3, p))

p1 = model.score('智 能 信 息 处 理 科 学 与 技 术')
p2 = model.score('科 学 与 技 术')
p3 = model.score('智 能 信 息 处 理 ')

print("prob1: {} \t prob2: {} \t prob3: {} \t prob: {}".format(p1, p2, p3, p))

p0 = model.score('年度招收硕士3名，招收数学，计算机等相关专业。 重庆大学计算机学院，副教授，硕士导师。2000年重庆 大学计算机学院本科毕业留校任教至今。2000年10月在日本电气通信大学短期交换留学一年；2003年在重庆大学计算机学院获得硕士学位；2007年公派至日本东京工业大学留学攻读博士学位，师从计算智能领域著名学者广田薰（Kaoru Hirota）教授')
p1 = model.score('业 。重')
p2 = model.score('招 收 硕')
p3 = model.score('重 庆 大')

print("prob1: {} \t prob2: {} \t prob3: {} \t prob: {}".format(p1, p2, p3, p))

p0 = model.score('Fossil HR具有单色显示屏，能够显示一些常用的信息，包括运动跟踪、天气信息和应用程序通知，同时非常节能，足以提供两周的电池续航时间。')
p1 = model.score('具 有 单')
p2 = model.score('用 的 信')
p3 = model.score('踪 、 天')

print("prob1: {} \t prob2: {} \t prob3: {} \t prob: {}".format(p1, p2, p3, p))
