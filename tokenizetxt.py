from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
data_file = "preprocessed_training_data.txt"

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def tokenize(text):
    paragraphs = text.split("\n\n")
    tokenized_paragraphs = []

    for paragraph in paragraphs:
        tokens = tokenizer.encode(paragraph, truncation=True, max_length=1024)
        tokenized_paragraphs.extend(tokens)

    return tokenized_paragraphs

def save_tokens(tokens, file_path):
    with open(file_path, "w") as file:
        for token in tokens:
            file.write(f"{token}\n")

text = read_file(data_file)
tokens = tokenize(text)
save_tokens(tokens, "training_data_tokens.txt")
