import os, os.path
import pickle as pkl
from whoosh.index import open_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from whoosh.qparser import QueryParser
import numpy as np
import jieba
import sys


class search_query:
    def __init__(self):
        self.text_ix = open_dir("indexdir_text")
        self.doc_ix = open_dir("indexdir_doc")
        with open(os.path.join(os.path.dirname(__file__), "data.txt"), "rb") as file:
            self.num, self.str_to_id, self.id_to_str, self.text_url, self.doc_url, self.text_tv, self.text_matrix, self.text_dimension, self.doc_tv, self.doc_matrix, self.doc_dimension = pkl.load(file)
        with open(os.path.join(os.path.dirname(__file__), "users.txt"), "rb") as file:
            self.users = pkl.load(file)

    def index_search(self, url_dict, ix, tv, matrix, dimension, query_content,  web = "", field = "content"):
        print("开始搜索")
        coe = 0.5
        scores = {}
        sum = 0
        with ix.searcher() as searcher:
            query = QueryParser(field, ix.schema).parse(query_content)
            results = searcher.search(query, scored = False)
            # results = searcher.search(query)
            sentences = query_content.split()
            sent_words = [list(jieba.cut(sent)) for sent in sentences]
            each_sent = [" ".join(sent) for sent in sent_words]
            document = " ".join(each_sent)
            query_vector = np.array(tv.transform([document]).toarray()[0])
            dot_result = np.dot(query_vector, matrix.T)
            for i in range(len(dot_result)):
                sum += dot_result[i] / url_dict[dimension[i]]["vector"][0]

            i = 0
            for result in results:
                if web != "" and web not in result["url"]:
                    continue
                id = result["ID"]
                scores[id] = url_dict[id]["pagerank"] * coe + np.dot(url_dict[id]["vector"][1], query_vector) / url_dict[id]["vector"][0] / sum * (1 - coe)
            for id, score in sorted(scores.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
                if i > 20:
                    return
                i += 1
                print(url_dict[id]["pagerank"] * coe, "   ", np.dot(url_dict[id]["vector"][1], query_vector) / url_dict[id]["vector"][0] / sum * (1 - coe))
                if "title" not in url_dict[id].keys():
                    print("score: ", score,  " url: " + self.id_to_str[id])
                else:
                    print("score: ", score,  "title: " + str(url_dict[id]["title"]) + " url: " + self.id_to_str[id])
            print("查询结果输出完毕")
    
    def save_user_data(self):
        with open(os.path.join(os.path.dirname(__file__), "users.txt"), "wb") as file:
            pkl.dump(self.users, file)

if __name__ == "__main__":
    # users = {}
    # with open(os.path.join(os.path.dirname(__file__), "users.txt"), "wb") as file:
    #     pkl.dump(users, file)
    searcher = search_query()
    choose = input("登录还是注册（登录/注册）？")
    if choose == "登录":
        name = input("请输入用户名：")
        if name not in searcher.users.keys():
            print("用户不存在")
            sys.exit(0)
        password = input("请输入密码：")
        if password != searcher.users[name]["password"]:
            print("密码错误")
            sys.exit(0)
    elif choose == "注册":
        name = input("请输入用户名：")
        if name in searcher.users.keys():
            print("重名")
            sys.exit(0)
        password = input("请输入密码：")
        searcher.users[name] = {"password": password, "log": []}
    else:
        print("请正确输入")
        sys.exit(0)
    while(1):
        choose = input("是否要进行站内搜索？(y/n)")
        web = ''
        if choose == 'y':
            web = input("请输入网站：")
        search_type = input("请输入您想进行的查询（文档/文本/日志）（输入任意其他内容则退出）：")
        if search_type == "文档":
            search_field = input("请输入您想查询的项目(content)：")
            if search_field not in ["url", "content"]:
                print("请输入正确的内容")
                continue
            search_content = input("请输入您想查询的内容：")
            log = "文档：" + search_content
            if web != '':
                log += "(site: " + web + ")"
            searcher.users[name]["log"].append("文档：" + search_content)
            searcher.index_search(searcher.doc_url, searcher.doc_ix, searcher.doc_tv, searcher.doc_matrix, searcher.doc_dimension, search_content, web, search_field)
        elif search_type == "文本":
            search_field = input("请输入您想查询的项目(title/content)：")
            if search_field not in ["url", "title", "content"]:
                print("请输入正确的内容")
                continue
            search_content = input("请输入您想查询的内容：")
            log = "文本：" + search_content
            if web != '':
                log += "(site: " + web + ")"
            searcher.users[name]["log"].append(log)
            searcher.index_search(searcher.text_url, searcher.text_ix, searcher.text_tv, searcher.text_matrix, searcher.text_dimension, search_content, web, search_field)
        elif search_type == "日志":
            print(searcher.users[name]["log"])
        else:
            searcher.save_user_data()
            break