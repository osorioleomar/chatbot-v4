import PyPDF2
import os

def pdf_to_text(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)  # Use PdfReader instead of PdfFileReader
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

input_dir = "C:\\Users\\umai\\Desktop\\projects\\chatbot\\version4\\pdf"
output_file = "training_data.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = pdf_to_text(pdf_path)
            outfile.write(text)
            outfile.write("\n")
