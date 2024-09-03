import streamlit as st  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain.document_loaders import PyPDFLoader  # type: ignore
from transformers import T5Tokenizer, T5ForConditionalGeneration  # type: ignore
import torch  # type: ignore
import base64
import tempfile
import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')

# Summarization MODEL AND TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map='auto', torch_dtype=torch.float32, offload_folder='offload_folder'
)

# File loader and preprocessing
def file_preprocessing(file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    full_text = ""
    for text in texts:
        full_text += text.page_content  # Adjusted to access the content correctly
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    
    return full_text

# Custom extractive summarization function
def custom_extractive_summary(text, top_n=10):  # Increase top_n to include more sentences
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence scores based on TF-IDF values
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Rank and select the top_n sentences
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-top_n:]]
    
    # Combine selected sentences into a preliminary extractive summary
    extractive_summary = ' '.join(ranked_sentences)
    
    return extractive_summary

# Abstractive summarization with key sentences included
def abstractive_with_key_sentences(full_text, key_sentences):
    # Combine full text and key sentences
    combined_text = full_text + "\n\n" + key_sentences
    
    # Prepare input for the T5 model
    inputs = tokenizer.encode("summarize: " + combined_text, return_tensors="pt")
    
    # Generate refined summary using T5 model
    outputs = base_model.generate(
        inputs, max_length=300, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    
    # Decode the generated summary
    final_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_summary

# Combined summarization pipeline
def llm_pipeline(file):
    input_text = file_preprocessing(file)
    
    # Extract key sentences
    extractive_summary = custom_extractive_summary(input_text, top_n=10)  # Increase top_n
    
    # Generate abstractive summary including key sentences
    refined_summary = abstractive_with_key_sentences(input_text, extractive_summary)
    
    return refined_summary

@st.cache_data
def display_PDF(file):
    # Read the file content
    file_content = file.read()
    
    # Encode file content to base64
    base64_pdf = base64.b64encode(file_content).decode('utf-8')

    # Embed PDF in HTML for display
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code part
st.set_page_config(layout='wide', page_title="Summarization App")

def main():
    st.title('Document Summarization App using LangChain')

    uploaded_file = st.file_uploader("Upload Your PDF File", type=['pdf'])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Uploaded PDF File")
            display_PDF(uploaded_file)
        
        with col2:
            st.info("Summary")
            if st.button("Summarize"):
                with st.spinner('Summarizing...'):
                    summary = llm_pipeline(uploaded_file)
                    st.success("Summarization Complete!")
                    st.subheader("Summary")
                    st.write(summary)
    
if __name__ == '__main__':
    main()
