import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from text_preprocessing import text_data, seg_lst
from matplotlib import pyplot as plt
import pyLDAvis.gensim_models



seg_lst = [lst for lst in seg_lst if lst]
dictionary= gensim.corpora.Dictionary(seg_lst)     # label for word
corpus = [dictionary.doc2bow(i) for i in seg_lst]  #Bag-of-words model



def coherence(num_topics):
    ldamodel = LdaModel(corpus, num_topics = num_topics, id2word = dictionary, passes = 30, random_state = 42)
    print(ldamodel.print_topics(num_topics = num_topics, num_words = 15))
    ldacm = CoherenceModel(model = ldamodel, texts = seg_lst, dictionary = dictionary, coherence="c_v", processes=1)
    print(ldacm.get_coherence())
    return ldacm.get_coherence()



def coherence_count(num_topics):   #計算news 21個topic 中最高的coherence value
    x = range(1,num_topics+1)
    y = [coherence(i) for i in x]
    global X, Y
    X, Y = x, y
    return x, y

def count_topic(num):  #y_value max最大 對應的x_topic是
    x, y = coherence_count(num)
    global max_x
    max_x = []
    for i in range(len(y)):
        if y[i] == max(y):
            max_x.append(x[i])
    return max_x


if __name__ == '__main__':
    count_topic(21) #可以先決定你的要計算的 coherence score共要計算幾類的topic


if 'X' in globals() and 'Y' in globals():
    plt.plot(X, Y)
    plt.xlabel("主題數目")
    plt.ylabel("coherence大小")
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.title("主題-coherence變化情形")
    plt.show()


num_topics = max_x[0]
lda = LdaModel(corpus, num_topics = num_topics, id2word = dictionary, passes = 30, random_state = 42)
topics_lst = lda.print_topics()
print(topics_lst)

output_file = 'C:/Users/Administrator/Desktop/NLP/result'  #store your train result
data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, mds='mmds')
pyLDAvis.save_html(data, f"{output_file}/{num_topics}_topic_model.html")
print(output_file)

