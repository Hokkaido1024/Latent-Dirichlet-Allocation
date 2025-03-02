import os 
import re
import pandas as pd
import urllib.request
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import warnings 
warnings.filterwarnings('ignore')

folder_path = r'c:/Users/Administrator/INTERN/Chinatimes_news'    #換成你的text_file_path(data_set)
excel_files = [file for file in os.listdir(folder_path) if file.endswith(('.csv', '.xls'))]

if excel_files:
    print(f"\n總共找到 {len(excel_files)} 個 Excel 檔案。\n")

def input_file(excel_files):    
    for file in excel_files:
        print(f"正在讀取檔案：{file}")
        file_path = os.path.join(folder_path, file)
        
        df = pd.read_csv(file_path)
        col_data = df['Text'].dropna().tolist()
        text_data = []

        for cel in col_data:
                text_data.append(cel[0:])       
                print(text_data)
    return text_data
    
text_data = input_file(excel_files)


#gpu:device = 0, cpu:device = -1
ws_driver = CkipWordSegmenter(model= 'bert-base', device = 0)
pos_driver = CkipPosTagger(model= 'bert-base' , device = 0)
ner_driver = CkipNerChunker(model= 'bert-base', device = 0)


stop_words = 'https://github.com/goto456/stopwords/blob/master/cn_stopwords.txt'  #中文專用的stopwords 
# 匯入停用詞
def read_stopword():
    # Use urllib.request to open the URL
    with urllib.request.urlopen(stop_words) as response:
        data = response.read().decode('utf-8')
        stopword = [word.strip("\n") for word in data.splitlines()]
    return stopword
stopwords = read_stopword()

# 對文章進行斷詞
def do_CKIP_WS(article):
    ws_results = ws_driver([str(article)])
    return ws_results

# 對詞組進行詞性標示
def do_CKIP_POS(ws_result):
    pos = pos_driver(ws_result[0])
    all_list = []
    for sent in pos:

        all_list.append(sent)
    return all_list

# 保留名詞與動詞

def pos_filter(pos):
    for i in list(set(pos)):
        if i.startswith("N") or i.startswith("V"):
            return "Yes"
        else:
            continue

# 去除數字與網址詞組
def remove_number_url(ws):
    number_pattern = "^\d+\.?\d*"
    url_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
    space_pattern = "^ .*"
    num_regex = re.match(number_pattern, ws)
    url_regex = re.match(url_pattern, ws)
    space_regex = re.match(space_pattern, ws)
    if num_regex ==  None and url_regex == None and space_regex == None:
        return True
    else:
        return False

# 執行資料清洗
def cleaner(ws_results, pos_results, stopwords):
    word_lst = []
    for ws, pos in zip(ws_results[0], pos_results):
        in_stopwords_or_not = ws not in stopwords  #詞組是否存為停用詞
        if_len_greater_than_1 = len(ws) > 1        #詞組長度必須大於1
        is_V_or_N = pos_filter(pos)                #詞組是否為名詞、動詞
        is_num_or_url = remove_number_url(ws)      #詞組是否為數字、網址、空白開頭
        if in_stopwords_or_not and if_len_greater_than_1 and is_V_or_N == "Yes" and is_num_or_url:
            word_lst.append(str(ws))
        else:
            pass
    return word_lst

def seg_word(text_data):
  seg_lst = []
  for i in range(len(text_data)):
    ws_results = do_CKIP_WS(text_data[i])
    pos_results = do_CKIP_POS(ws_results)
    seg_lst.append(cleaner(ws_results, pos_results, stopwords))
  return seg_lst

seg_lst = seg_word(text_data)  #finish preprocessing

