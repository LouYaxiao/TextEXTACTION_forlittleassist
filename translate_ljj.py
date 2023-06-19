import pdfplumber
import re
import jieba
from googletrans import Translator
# pdf_path=r'E:\GIS大赛-臭氧植被\'To Put an An...liberal Abyss_Leyan Hou.pdf'
# stop_path=r'E:\pythonProject\StopWords.txt'
# txt_path=r'E:\pythonProject\out.txt'
# trans_path=r'E:\pythonProject\trans.txt'
def pdf2txt(pdf_path):

    # path=r'E:\GIS大赛-臭氧植被\'To Put an An...liberal Abyss_Leyan Hou.pdf'
    txt_out = open(txt_path, mode='w', encoding='utf-8')
    with pdfplumber.open(pdf_path) as pdf_file:
        page = len(pdf_file.pages)
        for index in range(page):
            page_content = pdf_file.pages[index]
            text = page_content.extract_text()
            txt_out.write(text)

def word_frequency_statistics(txt_path,stop_path,language):

    stop = open(stop_path, 'r', encoding='utf-8').read()
    f = open(txt_path, 'r', encoding='utf-8')
    txt = f.read()
    txt = txt.lower()
    f.close()
    # 英文词频统计
    if language=='EN':
        array=re.split('[ \s]',txt)
        d={}
        for word in array:
            word=word.lower()
            if word not in stop:
                if word in d:
                    d[word]+=1
                else:
                    d[word]=1
#del [d['the'],d['and'],d['to'],d['in'],d['on'],d['of'],d['we'],d['a'],d['with'],d['are'],d['is'],d['were'],d['was']]
        list1 = sorted(d.items(), key=lambda x: x[1])
        print(list1[-1:-(10 + 1):-1])

#中文词频统计
    if language == "CN":
        words = jieba.cut_for_search(txt)
        d = {}
        for word in words:
            if word not in stop:
                if(len(word)==1):
                    continue
                elif word in d:
                    d[word]+=1
                else:
                    d[word]=1
        list1 = sorted(d.items(), key=lambda x: x[1])
        print(list1[-1:-(10 + 1):-1])

def translate_text(txt_path,trans_path):
    f = open(txt_path, 'r', encoding='utf-8')
    txt = f.read()
    txt = txt.lower()
    f.close()
    long = len(txt)
    lim = 5000
    if long > lim:
        txt_segments = []
        while len(txt) > lim:
            txt_segments.append(txt[:lim])
            txt = txt[lim:]
        txt_segments.append(txt)
    else:
        txt_segments = txt
    translator = Translator()
    file = open(trans_path, 'w', encoding='utf-8')
    for i in range(len(txt_segments)):
        result = translator.translate(txt_segments[i], dest='zh-CN')
        file.write(result.text)
    file.close()



if __name__ == "__main__":
    pdf_path = r'D:\python\Nonmonotonic Reasoning, Preferential Models(综述1）.pdf'
    stop_path = r'D:\python\StopWords.txt'
    txt_path = r'D:\python\out.txt'
    trans_path = r'D:\python\trans.txt'
    pdf2txt(pdf_path)
    word_frequency_statistics(txt_path,stop_path,'EN')
    translate_text(txt_path,trans_path)