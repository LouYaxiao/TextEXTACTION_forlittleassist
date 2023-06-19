import openai
import pdfplumber
import re
import jieba
from googletrans import Translator
import os
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification
import PyPDF4

stop_path = r'D:\python\StopWords.txt'  # 频繁词消去文件，作为全局变量
# Set OpenAI API access key
openai.api_key = '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'

# 阿靖
def aj_extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF4.PdfFileReader(file)
        text = ''
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            text += page.extractText()
        return text

def generate_mind_map(text):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Save mind map as PDF
def save_as_png(mind_map, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(mind_map)
            # print(mind_map, "创建完成")
    except:
        print(mind_map, "已经存在")

# 翻译的

def process_tran(pdf_path, pdf_name):
    # 将pdf转换为txt，可以考虑替换一下，因为另一个也有不错的
    # total_pdf是pdf的完整路径，包含了自己的pdf文件名
    total_pdf = os.path.join(pdf_path, pdf_name)
    print("total_pdf is :", total_pdf)
    text = aj_extract_text_from_pdf(total_pdf)

    # Generate mind map
    try:
        mind_map = generate_mind_map(text)
    except:
        print("未能生成思维导图")
    print(pdf_name,"has generated")
    # Save as PDF file
    mind_map_file = pdf_name + '_mind_map.png'
    total_mind_map_file = os.path.join(pdf_path, mind_map_file)
    try:
        save_as_png(mind_map, total_mind_map_file)
    except:
        print("未能生成pdf思维导图")
    print(pdf_name, " is Mind map generated and saved as mind_map.pdf")


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

def extract_text_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def chunked_summarization(summarizer, text, max_length=200, min_length=40, do_sample=False):
    tokens = summarizer.tokenizer(text, truncation=False)["input_ids"]
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    chunks_text = [summarizer.tokenizer.decode(chunk) for chunk in chunks]
    summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['summary_text'] for chunk in chunks_text]
    summary = ' '.join(summaries)
    return summary

def chunked_pipeline(nlp_qa, context, question):
    max_length = 500
    chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
    qa_results = [nlp_qa(question=question, context=chunk) for chunk in chunks]
    qa_results = sorted(qa_results, key=lambda x: x['score'], reverse=True)
    return qa_results[0]

# 潇宝

def process_summary(pdf_name,pdf_path,dir_path):
    p = os.path.join(pdf_path,pdf_name)
    # print("p is :", p)
    text = extract_text_from_pdf(p)

    # Generate mind map
    mind_map = generate_mind_map(text)

    # Save as PDF file
    mind_map_file = pdf_name + 'mind_map.pdf'
    mind_map_file = mind_map_file
    save_as_pdf(mind_map, mind_map_file)
    print(pdf_name, " is Mind map generated and saved as mind_map.pdf")
    papers_text = []
    text = extract_text_from_pdf(file_path)
    papers_text.append(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    papers_embeddings = model.encode(papers_text)

    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

    papers_ner_results = []
    for text in papers_text:
        ner_results = ner_pipeline(text[
                                   :100000])  # Limiting to first 100000 characters. You might need to adjust this value based on the length of your text.
        papers_ner_results.append(ner_results)

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

    summary_texts = [chunked_summarization(summarizer, text) for text in papers_text]
    save_txt = summary_texts[0]
    folder_name = pdf_name + "_summary"
    # print("summary texts is:",summary_texts)
    # print("save_txt is:",save_txt)
    save_path = os.path.join(dir_path,folder_name)
    # print("save_path is",save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("文件夹已经创建")
    else:
        print("文件夹已经存在")

    summary_name = pdf_name + "_summary.txt"
    summary_path = os.path.join(save_path,summary_name)
    # print(summary_path)
    try:
        with open(summary_path, 'w') as file:
            file.write()
            print("新建总结文件完成")
    except:
        print("文件已经存在")

    summary = open(summary_path,'w',encoding='utf-8')
    summary.write(save_txt)
    # print(type(summary_texts))
    # print(len(summary_texts))


if __name__ == "__main__":
    # 这里在执行翻译过程
    dir_path = "D:\Python"  # the path to your folder with PDF
    num_threads = 4
    executor = ThreadPoolExecutor(max_workers=num_threads)

    # 寻找pdf文件
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".pdf"):
            executor.submit(process_tran, dir_path, file_name)

    # 潇宝
    num_threads = 4
    new_executor = ThreadPoolExecutor(max_workers=num_threads)

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".pdf"):
            pdf_name = file_name
            file_path = os.path.join(dir_path, file_name)
            pdf_path = file_path
            print(type(pdf_path))
            new_executor.submit(process_summary, pdf_name, pdf_path, dir_path)

    executor.shutdown()
    new_executor.shutdown()







