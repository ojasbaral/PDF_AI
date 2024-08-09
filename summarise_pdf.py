import argparse
import PyPDF2
import ollama

def parse_arguments():
    parser = argparse.ArgumentParser(description="Summarize a PDF using a local Llama model.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    return parser.parse_args()

def extract_text(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

def summarize_pdf(pdf_text: str) -> str:
    stream = ollama.chat(
        model='llama2',
        messages=[{'role': 'user', 'content': ("Summarize the following text from a pdf: " + pdf_text)}],
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def main():
    args = parse_arguments()
    pdf_text = extract_text(args.pdf_path)
    summarize_pdf(pdf_text)

if __name__ == '__main__':
    main()