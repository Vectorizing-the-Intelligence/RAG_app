import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from multiprocessing import Pool
import streamlit as st
import docx
import fitz  # PyMuPDF
import pyarrow as pa
import pyarrow.parquet as pq
import google.generativeai as genai  # Import Gemini

# --- Model Functions ---
def load_stella_model():
    """
    Load the Stella model (dunzhang/stella_en_1.5B_v5).
    """
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def encode_stella(text, tokenizer, model):
    """
    Encode a text object (e.g., paragraph or content) with Stella.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Averaging the embeddings
    return embeddings.squeeze().numpy()

# --- T5 Model Functions ---
def load_t5_model():
    """
    Load the T5 model for text generation.
    """
    model_name = "t5-small"  # You can use "t5-base" or "t5-large" for better performance
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_response_with_t5(query, answers, tokenizer, model):
    """
    Generate a refined response using the T5 model.
    """
    # Combine the query and answers into a single input text
    input_text = f"Query: {query}\nAnswers: {' '.join(answers)}"
    
    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the output
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    
    # Decode the output to get the final response
    refined_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined_response

# --- Gemini Model Functions ---
def generate_response_with_gemini(query, answers):
    """
    Generate a refined response using the Gemini model.
    """
    # Configure Gemini
    genai.configure(api_key="AIzaSyC-XVE29P1gAzZGAVkaRU-etnWtcqqChf8")  # Replace with your Gemini API key
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Combine the query and answers into a single input text
    input_text = f"Query: {query}\nAnswers: {' '.join(answers)}"
    
    # Add instructions for Gemini to generate three paragraphs
    prompt = f"Generate three paragraphs based on the following query and answers:\n{input_text}"
    
    # Generate the response
    response = model.generate_content(prompt)
    return response.text

# --- Search Functions ---
def retrieve_response(query, parquet_folder_path, tokenizer, model, metric='cosine', limit=5):
    """
    Retrieve the most relevant documents based on a query.
    """
    if not query:
        raise ValueError("Query cannot be None or empty")
    
    # Step 1: Encode the query using the Stella model
    query_embedding = encode_stella(query, tokenizer, model)
    
    # Step 2: Load the parquet files and their embeddings
    embeddings = []
    contents = []
    paragraph_numbers = []
    file_names = []
    
    # Iterate over all parquet files in the folder and extract embeddings and content
    for filename in os.listdir(parquet_folder_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(parquet_folder_path, filename)
            parquet_file = pd.read_parquet(file_path, engine='pyarrow')

            # Extract embeddings and content from the parquet file
            for i, (content, embedding) in enumerate(zip(parquet_file.paragraph, parquet_file.embedding)):
                contents.append(content)
                embeddings.append(embedding)
                paragraph_numbers.append(i + 1)  # Paragraph numbers start from 1
                file_names.append(filename)
    
    # Step 3: Compute similarity between query embedding and stored embeddings
    embeddings_array = np.array(embeddings)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], embeddings_array)[0]
    
    # Step 4: Get the top `limit` most similar documents
    top_indices = np.argsort(similarities)[-limit:][::-1]
    best_answers = [contents[i] for i in top_indices]
    best_paragraph_numbers = [paragraph_numbers[i] for i in top_indices]
    best_file_names = [file_names[i] for i in top_indices]
    
    return best_answers, best_paragraph_numbers, best_file_names

# --- Utility Functions ---
def process_file(file_path, tokenizer, model):
    """
    Process a single file to generate embeddings.
    """
    try:
        parquet_file = pd.read_parquet(file_path, engine='pyarrow')
        parquet_file['embedding'] = pd.Series()

        # Generate embeddings for each paragraph or content
        for i in range(len(parquet_file)):
            embedding = encode_stella(parquet_file.paragraph[i], tokenizer, model)
            parquet_file.at[i, 'embedding'] = embedding
        
        return parquet_file
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def add_embeddings(parquet_folder_path, output_folder_path, tokenizer, model):
    """
    Add embeddings to parquet files using multiprocessing.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    file_paths = [os.path.join(parquet_folder_path, filename) for filename in os.listdir(parquet_folder_path) if filename.endswith('.parquet')]
    
    with Pool() as pool:
        results = pool.starmap(process_file, [(file_path, tokenizer, model) for file_path in file_paths])
    
    for result, file_path in zip(results, file_paths):
        if result is not None:
            output_file_path = os.path.join(output_folder_path, f"embeddings_{os.path.basename(file_path)}")
            result.to_parquet(output_file_path, engine='pyarrow')
            print(f"Embeddings added to {os.path.basename(file_path)} and saved to {output_file_path}")

# --- File Processing Functions ---
def segment_docx_paragraphs(docx_path):
    """
    Extracts paragraphs from a DOCX file and returns a dictionary, 
    excluding paragraphs with fewer than 30 words.
    """
    doc = docx.Document(docx_path)
    paragraphs = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text and len(text.split()) >= 30:  # Avoid empty paragraphs and those with less than 30 words
            # Remove extra spaces and newlines
            text = ' '.join(text.split())
            paragraphs.append(text)

    return {'paragraph_num': [i for i in range(len(paragraphs))], 'paragraph': paragraphs}

def segment_pdf_paragraphs(pdf_path, min_words=30):
    """
    Extracts paragraphs from a PDF file using text blocks, combines small lines into larger paragraphs,
    ensures each paragraph has at least `min_words` words, and removes all '\n' characters and extra spaces.
    """
    doc = fitz.open(pdf_path)
    paragraphs = []
    current_paragraph = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")  # Extract text blocks

        for block in blocks:
            text = block[4].strip()  # The text is in the 5th element of the block tuple
            if not text:
                continue  # Skip empty blocks

            # If the block starts with a capital letter, it's likely a new paragraph
            if current_paragraph and text[0].isupper():
                # If the current paragraph is long enough, save it
                if len(current_paragraph.split()) >= min_words:
                    # Remove extra spaces and newlines
                    current_paragraph = ' '.join(current_paragraph.split())
                    paragraphs.append(current_paragraph)
                current_paragraph = text  # Start a new paragraph
            else:
                # Add the block to the current paragraph
                current_paragraph += " " + text if current_paragraph else text

        # After processing each page, check if the current paragraph is long enough
        if len(current_paragraph.split()) >= min_words:
            # Remove extra spaces and newlines
            current_paragraph = ' '.join(current_paragraph.split())
            paragraphs.append(current_paragraph)
            current_paragraph = ""  # Reset for the next page

    # Add the last paragraph if it exists and is long enough
    if current_paragraph and len(current_paragraph.split()) >= min_words:
        # Remove extra spaces and newlines
        current_paragraph = ' '.join(current_paragraph.split())
        paragraphs.append(current_paragraph)

    return {'paragraph_num': [i for i in range(len(paragraphs))], 'paragraph': paragraphs}

def save_to_parquet(data, output_file):
    """
    Saves the extracted paragraph data to a Parquet file.
    """
    table = pa.Table.from_pydict(data)
    pq.write_table(table, output_file)

def process_folder_for_files(folder_path, output_folder):
    """
    Processes all DOCX and PDF files in the specified folder, extracting their paragraphs 
    and saving them to Parquet files.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            if file_name.endswith('.docx'):
                print(f"Processing DOCX file: {file_name}")
                # Extract paragraphs from the DOCX file
                paragraphs = segment_docx_paragraphs(file_path)
                # Define the Parquet output file path
                parquet_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_docx.parquet")                
            elif file_name.endswith('.pdf'):
                print(f"Processing PDF file: {file_name}")
                # Extract paragraphs from the PDF file
                paragraphs = segment_pdf_paragraphs(file_path)
                # Define the Parquet output file path
                parquet_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_pdf.parquet")                                
            else:
                print(f"Skipping unsupported file: {file_name}")
                continue
            
            # Save to Parquet
            save_to_parquet(paragraphs, parquet_file)
            print(f"Saved {file_name} paragraphs to {parquet_file}.")

# --- Streamlit App ---
def run_streamlit_app():
    """
    Run the Streamlit app for document search.
    """
    # Load the Stella model
    tokenizer_stella, model_stella = load_stella_model()

    # Load the T5 model
    tokenizer_t5, model_t5 = load_t5_model()

    # Streamlit app
    st.title("Document Search Engine")

    # Step 1: Allow the user to upload PDF or Word files
    uploaded_files = st.file_uploader("Upload PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True)

    # Create a temporary folder to store uploaded files
    temp_folder = "temp_uploads"
    os.makedirs(temp_folder, exist_ok=True)

    if uploaded_files:
        # Save uploaded files to the temporary folder
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_folder, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Process the uploaded files
        st.info("Processing uploaded files. Please wait...")
        process_folder_for_files(temp_folder, "parquet_folder")
        st.success("Files processed successfully!")

        # Add embeddings to the parquet files
        st.info("Generating embeddings. Please wait...")
        add_embeddings("parquet_folder", "parquet_plus_embedding", tokenizer_stella, model_stella)
        st.success("Embeddings generated successfully!")

    # Step 2: Display the list of files in the parquet_plus_embedding folder
    st.subheader("Generated parquet files:")
    if os.path.exists("parquet_plus_embedding"):
        files = os.listdir("parquet_plus_embedding")
        if files:
            for file in files:
                st.write(file)
        else:
            st.write("No files found.")
    else:
        st.write("The folder parquet_plus_embedding does not exist.")

    # Step 3: Allow the user to ask a query
    query = st.text_input("Enter your query:")

    # Step 4: Allow the user to choose between T5 and Gemini
    model_choice = st.selectbox("Choose a model for generation:", ["T5", "Gemini"])

    if query:
        # Retrieve the most relevant documents
        best_answers, best_paragraph_numbers, best_file_names = retrieve_response(query, "parquet_plus_embedding", tokenizer_stella, model_stella, metric='cosine', limit=5)
        
        # Display the results in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top relevant answers:")
            for i, (answer, paragraph_num, file_name) in enumerate(zip(best_answers, best_paragraph_numbers, best_file_names)):
                st.write(f"**Paragraph {paragraph_num} from {file_name}:**")
                st.write(answer)

        with col2:
            st.subheader(f"Refined response from {model_choice} model:")
            if model_choice == "T5":
                refined_response = generate_response_with_t5(query, best_answers, tokenizer_t5, model_t5)
            else:
                refined_response = generate_response_with_gemini(query, best_answers)
            st.write(refined_response)

# --- Main Function ---
def main():
    # Run the Streamlit app
    run_streamlit_app()

if __name__ == '__main__':
    main()