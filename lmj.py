import PyPDF4
import openai

# Set OpenAI API access key
openai.api_key = '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'

# Extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF4.PdfFileReader(file)
        text = ''
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            text += page.extractText()
        return text

# Generate mind map
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
def save_as_pdf(mind_map, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(mind_map)
            # print(file,"创建完成")
    except:
        print(mind_map,"已经存在")

# Main function
def main():
    # Read PDF file and extract text
    pdf_file = r"D:\python\A Logical Framework for Default Reasoning.pdf"
    text = extract_text_from_pdf(pdf_file)

    # Generate mind map
    mind_map = generate_mind_map(text)

    # Save as PDF file
    mind_map_file = 'mind_map.pdf'
    save_as_pdf(mind_map, mind_map_file)

    print("Mind map generated and saved as mind_map.pdf")

if __name__ == '__main__':
    main()