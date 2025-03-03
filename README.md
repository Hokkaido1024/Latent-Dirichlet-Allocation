# Latent-Dirichlet-Allocation

LDA (Latent Dirichlet Allocation) is a generative model used to discover hidden topics within a text dataset. It assumes that each document is composed of multiple topics, and each word is associated with different topics with varying probabilities. By applying LDA, we can infer the most probable topic distribution for each document and identify the high-frequency words for each topic.Using Anaconda for environment setup, with Visual Studio Code as the IDE.

conda install python ==3.7.16

Preprocessing (Text Preprocessing)：
If you want to uncover latent topics in a text corpus, you can use LDA for Topic Modeling. Since this project works with Chinese text, we utilize ckip-transformers, which provides Traditional Chinese Transformers models and NLP tools, including CkipWordSegmenter, CkipPosTagger, and CkipNerChunker. For more details, please refer to:
https://github.com/ckiplab/ckip-transformers?tab=readme-ov-filepip install ckip-transformers

Additionally, stopword filtering is required for effective data cleaning. If the predefined stopword list does not meet your needs, you can modify it accordingly. The stopword list is sourced from: https://github.com/goto456/stopwords/blob/master/cn_stopwords.txt

Data Transformation：
After text preprocessing, the data is converted into the Bag of Words (BoW) format using Dictionary and doc2bow(), making it suitable for LDA modeling. Before feeding the data into LDA, determining an optimal number of topics is essential. We first train the LDA model and evaluate its topic quality using CoherenceModel, which calculates the coherence score. A higher coherence score indicates better topic quality.

Based on the coherence score results, we select the optimal number of topics and retrain the LDA model to obtain the final topic distribution for the dataset.pip install gensim

Visualization：
Finally, we use Multidimensional Scaling (MDS) to visualize the topic distribution across documents and assess the effectiveness of the LDA model.

conda install matplotlib
pip install pyLDAvis gensim_models

This code is implemented using the text crawled from web-crawler-for-chinatimes to perform LDA modeling. If you want to use your own dataset, please remember to modify the `file_path` or adjust the data loading method. If you encounter any issues, feel free to reach out, and I'll make changes when possible.

