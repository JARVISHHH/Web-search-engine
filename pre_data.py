import os, os.path
import pickle as pkl
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.fields import *
from jieba.analyse import ChineseAnalyzer
from whoosh.qparser import QueryParser
from whoosh.analysis import NgramTokenizer
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import numpy as np


class pre_data:
    
    def indexing(self):
        """
        功能：根据网页类型进行索引的构建。文本类型有自己的索引，文档类型也有自己的索引，两个索引是分开的。
        """
        if not os.path.exists("indexdir_text"):
            os.mkdir("indexdir_text")
        if not os.path.exists("indexdir_doc"):
            os.mkdir("indexdir_doc")
        # 导入中文分词工具 
        analyser = ChineseAnalyzer()
        schema = Schema(ID = STORED, url = TEXT(stored = True), title = TEXT(stored = True, analyzer = analyser), content = TEXT(analyzer = analyser))

        # 创建一个索引
        ix = create_in("indexdir_text", schema)
        writer = ix.writer()
        # 将文本网页中的相关信息都加入到这个索引中
        for id in self.text_url.keys():
            url = self.id_to_str[id]
            title = self.text_url[id]["title"]
            content = self.text_url[id]["text"]
            writer.add_document(ID = id, url = url, title = title, content = content)
        writer.commit()
        
        # 重复上述类似过程
        ix = create_in("indexdir_doc", schema)
        writer = ix.writer()
        for id in self.doc_url.keys():
            url = self.id_to_str[id]
            title = ""
            content = self.doc_url[id]["text"]
            writer.add_document(ID = id, url = url, title = title, content = content)
        writer.commit()
    
    def compute_page_rank(self):
        """
        功能：计算网页的PageRank值，直接调用nextworkx中的函数
        """

        # 创建一张图
        G = nx.DiGraph()

        # 将网页的有向边都加到这个图上去
        for head in self.text_url.keys():
            for tail in self.text_url[head]["related url"].keys():
                G.add_edge(head,tail)

        # 调用pagerank函数进行计算
        pr = nx.pagerank(G,alpha=0.85)

        # 将计算好后的pagerank值加入到对应字典的相关信息中
        for node, value in pr.items():
            if node in self.text_url.keys():
                self.text_url[node]["pagerank"] = value
            if node in self.doc_url.keys():
                self.doc_url[node]["pagerank"] = value
    
    def text_vectorizer(self):
        """
        功能：构建文本网页的空间向量模型
        """
        train = []  # 训练集
        dimension_to_id = {}  # 构建维度到id的字典，从这个字典中可以知道向量某一维度对应的时哪个url。
        for id in self.text_url.keys():
            url = self.id_to_str[id]
            # 对网页内容进行中文分词后，加入训练集中
            sentences = self.text_url[id]["text"].split()
            sent_words = [list(jieba.cut(sent)) for sent in sentences]
            each_sent = [" ".join(sent) for sent in sent_words]
            document = " ".join(each_sent)
            dimension_to_id[len(train)] = id
            train.append(document)
        
        self.text_dimension = dimension_to_id
        self.text_tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        text_tv_fit = self.text_tv.fit_transform(train)  # 训练并获得结果矩阵
        vector_array = text_tv_fit.toarray()
        self.text_matrix = np.array(vector_array)
        # 将每个url对应的向量加入到其字典的相关信息中
        for dimension in range(len(vector_array)):
            id = dimension_to_id[dimension]
            x = np.array(vector_array[dimension])
            self.text_url[id]["vector"] = [np.linalg.norm(x), x]
    
    def doc_vectorizer(self):
        """
        功能：构建文档网页的空间向量模型，与上个函数类似
        """
        train = []
        dimension_to_id = {}
        for id in self.doc_url.keys():
            url = self.id_to_str[id]
            sentences = self.doc_url[id]["text"].split()
            sent_words = [list(jieba.cut(sent)) for sent in sentences]
            each_sent = [" ".join(sent) for sent in sent_words]
            document = " ".join(each_sent)
            dimension_to_id[len(train)] = id
            train.append(document)
        
        self.doc_dimension = dimension_to_id
        self.doc_tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        tv_fit = self.doc_tv.fit_transform(train)
        vector_array = tv_fit.toarray()
        self.doc_matrix = np.array(vector_array)
        for dimension in range(len(vector_array)):
            id = dimension_to_id[dimension]
            x = np.array(vector_array[dimension])
            # 有的向量长度可能非常小或直接就是0，这时候就把向量长度设为1
            if np.linalg.norm(x) == 0:
                self.doc_url[id]["vector"] = [1, x]
                continue
            self.doc_url[id]["vector"] = [np.linalg.norm(x), x]
    
    
    def get_data(self):
        """
        功能：从url.txt文件中获得待处理的数据
        """
        with open(os.path.join(os.path.dirname(__file__), "url.txt"), "rb") as file:
            self.num, self.str_to_id, self.id_to_str, self.skip_urls, self.text_url, self.doc_url, queue_urls = pkl.load(file)
    
    def save_data(self):
        """
        功能：保存处理好之后的数据
        """
        with open(os.path.join(os.path.dirname(__file__), "data.txt"), "wb") as file:
            pkl.dump([self.num, self.str_to_id, self.id_to_str, self.text_url, self.doc_url, self.text_tv, self.text_matrix, self.text_dimension, self.doc_tv, self.doc_matrix, self.doc_dimension], file)

if __name__ == "__main__":
    nankai_data = pre_data()
    print("获取数据")
    nankai_data.get_data()
    print("开始进行索引构建")
    nankai_data.indexing()
    print("开始进行向量化")
    nankai_data.text_vectorizer()
    nankai_data.doc_vectorizer()
    print("开始计算pagerank")
    nankai_data.compute_page_rank()
    print("保存数据")
    nankai_data.save_data()