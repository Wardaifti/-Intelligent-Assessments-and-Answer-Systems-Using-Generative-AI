import fitz
import openai
import os
import time

def extract_text_free_from_pdf_in_chunks(pdf_path, chunk_size=5000):
    doc = fitz.open(pdf_path)
    text_chunks = []
    text = " "
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        if len(text.split()) >= chunk_size:
            text_chunks.append(text)
            text = " "
    if text:
        text_chunks.append(text)
    return text_chunks

def count_tokens(text, api_key):
    openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": text}],
        max_tokens=1
    )
    return response.usage['total_tokens']

def generate_mcqs(text_chunk, api_key, num_questions):
    openai.api_key = api_key
    
    prompt = (
        f"Create {num_questions} multiple-choice questions based on the following text:\n\n"
        f"{text_chunk}\n\n"
        "Each question should have one correct answer and three incorrect answers."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

def main(pdf_path, api_key, num_questions, chunk_size=5000):
    text_chunks = extract_text_free_from_pdf_in_chunks(pdf_path, chunk_size)
    print(f"Extracted {len(text_chunks)} chunks of text from PDF.")
    
    for i, text_chunk in enumerate(text_chunks):
        try:
            print(f"Processing chunk {i+1} / {len(text_chunks)}")
            tokens = count_tokens(text_chunk, api_key)
            print(f"Chunk {i+1} contains {tokens} tokens.")
            
            mcqs = generate_mcqs(text_chunk, api_key, num_questions)
            print(f"Generated MCQs for chunk {i+1}:\n{mcqs}\n")
            
            time.sleep(1)
        except Exception as e:
            print(f"An error occurred while processing chunk {i+1}: {e}")

if __name__ == "__main__":
    pdf_path = "aws.pdf"
    api_key = "w9lgy5KL5PCGNrnBtM9Fm-Dis4IvoTLnWqRW7t3B1oNjT3BlbkFJl96d-kumBlG258NxC7J9XHSxBEeqtnYKPJ0vvYkfXaw3fCL0dOaJFqUH4A"
    num_questions = int(input("Enter the number of questions to be generated: "))
    chunk_size = 500
    
    main(pdf_path, api_key, num_questions, chunk_size)

    
    