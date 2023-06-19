import threading

from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification
from concurrent.futures import ThreadPoolExecutor

def process(pdf_info):
    thread_id = threading.current_thread().ident
    model = SentenceTransformer('all-MiniLM-L6-v2')
    papers_text = pdf_info[2]
    papers_embeddings = model.encode(papers_text)
    print("线程:",thread_id)
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

    papers_ner_results = []

    # 这里在切片
    for text in papers_text:
        ner_results = ner_pipeline(text[:100000])
        # Limiting to first 100000 characters. You might need to adjust this value based on the length of your text.
        papers_ner_results.append(ner_results)

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    print("papers_ner_results is:",papers_ner_results)
    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    print("prepare summary texts")
    summary_texts = [chunked_summarization(summarizer, text) for text in papers_ner_results]
    print("finish summary texts")
    # 指定目标文件夹路径
    output_dir = r'D:\Python'  # 设置新建文件夹的位置，例如在 D:/Python/Output 下创建新文件夹
    print(output_dir)
    print(summary_texts)
    # 指定新文件夹名称
    folder_name = pdf_info[0] + "_document"

    # 使用 os.path.join() 创建完整路径
    # D:\Python\文件名_document这个文件夹里
    folder_path = os.path.join(output_dir, folder_name)
    print(folder_path)
    # 创建新文件夹
    try:
        os.mkdir(folder_path)
    except:
        print("路径已经存在")

    #创建文件名
    save_name = pdf_info[0] + "_summary.txt"
    print("save_name is:", save_name)
    # save_path = output_dir + file_name
    save_w = os.path.join(folder_path, save_name)

    print("save_path is:", save_w)
    #在指定位置创建文件
    try:
        with open(save_w, 'w') as file:
            file.write(summary_texts)
            print(save_name, "创建成功，写入完成")
    except:
        print("文件已经存在")
    print("Folder created at:", folder_path)

# 提取pdf中的内容
def extract_text_from_pdf(file_path,file_name,pdf_info):
    thread_id = threading.current_thread().ident
    print("Thread ID:", thread_id)
    pdf = PdfReader(file_path)
    pdf_inf = []
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    pdf_inf.append(file_name)
    pdf_inf.append(file_path)
    pdf_inf.append(text)
    pdf_info.append(pdf_inf)


# 将文本进行分块摘要，合并出一个总摘要
def chunked_summarization(summarizer, text, pdf_name, max_length=200, min_length=40, do_sample=False):
    tokens = summarizer.tokenizer(text, truncation=False)["input_ids"]
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    chunks_text = [summarizer.tokenizer.decode(chunk) for chunk in chunks]
    summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['summary_text'] for chunk in chunks_text]
    summary = ' '.join(summaries)
    print("finish")
    return summary


# 使用了question-answering模型
# 将问题应用于文本的分块
# 返回得分最高的答案结果
def chunked_pipeline(nlp_qa, context, question):
    max_length = 500
    chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
    qa_results = [nlp_qa(question=question, context=chunk) for chunk in chunks]
    qa_results = sorted(qa_results, key=lambda x: x['score'], reverse=True)
    return qa_results[0]


num_threads = 4

executor = ThreadPoolExecutor(max_workers=num_threads)
# 这里定义了文件的路径
dir_path = "D:\Python"  # the path to your folder with PDFs
papers_text = []
pdf_info = []   # 这里的结构为[[pdf文件名,pdf文件路径,把每页都提取出来的结果]]

for file_name in os.listdir(dir_path):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(dir_path, file_name)
        executor.submit(extract_text_from_pdf,file_path,file_name,pdf_info)
        # text = extract_text_from_pdf(file_path)
        # papers_text.append(text)
# 线程池关闭了以后不能继续工作了，只能新建一个内存池
executor.shutdown(wait=True)

# for i in range(len(pdf_info)):
#     print(pdf_info[i][0])
#     print(pdf_info[i][1])

new_executor = ThreadPoolExecutor(max_workers=num_threads)

print(len(pdf_info))
for i in range(len(pdf_info)):
    new_executor.submit(process, pdf_info[i])

new_executor.shutdown(wait=True)

# # 这里是原来的代码
#
# for file_name in os.listdir(dir_path):
#     if file_name.endswith(".pdf"):
#         file_path = os.path.join(dir_path, file_name)
#         text = extract_text_from_pdf(file_path)
#         papers_text.append(text)
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
# papers_embeddings = model.encode(papers_text)
#
# ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
#
# papers_ner_results = []
# for text in papers_text:
#     ner_results = ner_pipeline(text[:100000])  # Limiting to first 100000 characters. You might need to adjust this value based on the length of your text.
#     papers_ner_results.append(ner_results)
#
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
# nlp = pipeline("ner", model=model, tokenizer=tokenizer)
#
# summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
#
# summary_texts = [chunked_summarization(summarizer, text) for text in papers_text]
#
# print(summary_texts)

