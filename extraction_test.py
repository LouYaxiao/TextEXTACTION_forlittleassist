from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification
from concurrent.futures import ThreadPoolExecutor


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



def process(pdf_name,pdf_path,dir_path):
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
    print("summary texts is:",summary_texts)
    print("save_txt is:",save_txt)
    save_path = os.path.join(dir_path,folder_name)
    print("save_path is",save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("文件夹已经创建")
    else:
        print("文件夹已经存在")

    summary_name = pdf_name + "_summary.txt"
    summary_path = os.path.join(save_path,summary_name)
    print(summary_path)
    try:
        with open(summary_path, 'w') as file:
            file.write()
            print("新建总结文件完成")
    except:
        print("文件已经存在")

    summary = open(summary_path,'w',encoding='utf-8')
    summary.write(save_txt)
    # for i in range(len(save_txt)):

    print(type(summary_texts))
    print(len(summary_texts))



# 主函数在这里
dir_path = "D:\Python"  # the path to your folder with PDFs

pdf_name = ""
pdf_path = ""
num_threads = 4
executor = ThreadPoolExecutor(max_workers=num_threads)

for file_name in os.listdir(dir_path):
    if file_name.endswith(".pdf"):
        pdf_name = file_name
        file_path = os.path.join(dir_path, file_name)
        pdf_path = file_path
        print(type(pdf_path))
        executor.submit(process, pdf_name, pdf_path, dir_path)


executor.shutdown()



#
# nlp_qa = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
#
# context = summary_texts
# question = input("Please enter your question: ")
#
# best_answer = chunked_pipeline(nlp_qa, context, question)
# print("The answer to your question is: " + best_answer['answer'])