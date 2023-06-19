import pdfplumber
import re
import jieba
import transformers
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor
import os

stop_path = r'D:\python\StopWords.txt'  # 频繁词消去文件，作为全局变量


def process(pdf_path, pdf_name):
    # 将pdf转换为txt，可以考虑替换一下，因为另一个也有不错的
    # total_pdf是pdf的完整路径，包含了自己的pdf文件名
    total_pdf = os.path.join(pdf_path,pdf_name)
    # 指定新文件夹的名称
    folder_name = pdf_name + "_translate_result"
    # 翻译结果存储的路径,加上了文件夹的名字
    save_path = os.path.join(pdf_path,folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("文件夹已经创建")
    else:
        print("文件夹已经存在")

    tran_name = pdf_name + "_tran.txt"
    # 将pdf转换为txt文件，txt文件保存的路径
    tran_path = os.path.join(save_path, tran_name)
    try:
        with open(tran_path, 'w') as file:
            file.write()
            print("新建文件完成")
    except:
        print("文件已经存在")
    trans_name = "trans.txt"
    trans_path = os.path.join(save_path,trans_name)
    print("trans_path is ", trans_path)
    try:
        with open(trans_path, 'w') as file:
            file.write()
            print("新建文件完成")
    except:
        print("文件已经存在")

    # print("准备翻译")
    pdf2txt(total_pdf,tran_path)
    # 统计单词
    word_frequency_statistics(tran_path, stop_path, 'EN')
    translate_text(tran_path, trans_path)



def pdf2txt(total_pdf,tran_path):
    # print("正在翻译:")
    # print(total_pdf)
    txt_out = open(tran_path,mode='w',encoding='utf-8')
    with pdfplumber.open(total_pdf) as pdf_file:
        page = len(pdf_file.pages)
        for index in range(page):
            page_content = pdf_file.pages[index]
            text = page_content.extract_text()
            txt_out.write(text)


def word_frequency_statistics(txt_path, stop_path, language):
    try:
        stop = open(stop_path, 'r', encoding='utf-8').read()
    except:
        print("未能打开文件")
    f = open(txt_path, 'r', encoding='utf-8')
    txt = f.read()
    txt = txt.lower()
    # print(txt)
    f.close()
    # 英文词频统计
    if language == 'EN':
        array = re.split('[ \s]', txt)
        d = {}
        for word in array:
            word = word.lower()
            if word not in stop:
                if word in d:
                    d[word] += 1
                else:
                    d[word] = 1
#del [d['the'],d['and'],d['to'],d['in'],d['on'],d['of'],d['we'],d['a'],d['with'],d['are'],d['is'],d['were'],d['was']]
        list1 = sorted(d.items(), key=lambda x: x[1])
        print(list1[-1:-(10 + 1):-1])
#中文词频统计
    if language == "CN":
        words = jieba.cut_for_search(txt)
        d = {}
        for word in words:
            if word not in stop:
                if(len(word) == 1):
                    continue
                elif word in d:
                    d[word] += 1
                else:
                    d[word] = 1
        list1 = sorted(d.items(), key=lambda x: x[1])
        print(list1[-1:-(10 + 1):-1])


def translate_text(txt_path, trans_path):
    # print("txt_path is:", txt_path)
    # print("trans_path is:", trans_path)
    try:
        f = open(txt_path, 'r', encoding='utf-8')
    except:
        print("txt_path 无法打开")
    txt = f.read()
    txt = txt.lower()
    f.close()
    long = len(txt)
    lim = 5000
    txt_segments = []
    if long > lim:

        while len(txt) > lim:
            txt_segments.append(txt[:lim])
            txt = txt[lim:]
        txt_segments.append(txt)
    else:
        txt_segments = txt
    translator = Translator()
    # print("txt_segments is:",txt_segments)
    file = open(trans_path, 'w', encoding='utf-8')
    # print("translator is:",translator)
    for i in range(len(txt_segments)):
        result = translator.translate(txt_segments[i], dest='zh-CN')
        # print(result.text)
        file.write(result.text)
    file.close()



if __name__ == "__main__":
    # pdf_path = r'D:\python\Nonmonotonic Reasoning, Preferential Models(综述1）.pdf'    # 要进行读取的pdf文件

    # txt_path = r'D:\python\out2.txt'         # pdf文件读取出来后的输出文件，
    # trans_path = r'D:\python\trans2.txt'     # 翻译出的文件
    dir_path = r'D:\Python'

    num_threads = 4
    executor = ThreadPoolExecutor(max_workers=num_threads)
    # 寻找pdf文件
    for file_name in os.listdir(dir_path):
        print(file_name)
        if file_name.endswith(".pdf"):
            executor.submit(process, dir_path, file_name)

executor.shutdown()




