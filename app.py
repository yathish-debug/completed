from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import *  # Import HTML templates
from openai import OpenAI
import openai
from langchain.llms import OpenAI

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_pdf_textt(pdf_docs):
    pdf_texts = {}
    for pdf in pdf_docs:
        pdf_name = pdf.name
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_texts[pdf_name] = text
    return pdf_texts

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    if text_chunks:
        embeddings = OpenAIEmbeddings()
        try:
            vectorstore = FAISS.from_texts(
                texts=text_chunks,
                embedding=embeddings
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vectorstore: {e}")
            return None
    else:
        return None
    
# Function to initialize a conversation chain
def initialize_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display responses
def handle_user_input(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def generate_themes_codes(pdf_docs):
    prompt1='give very very long summary of all the papers '
    prompt2='Identify up to 13 most important themes in the text, provide a meaningful name for each theme in 3 words'
    ###,4 lines meaningful and dense description of the theme?"
    prompt3='Group the provided codes under related themes and print them. First, print the theme name, then list their related codes one below the other.?'
    llm = OpenAI(temperature=0.7)
    dic={}
    for key, value in get_pdf_textt(pdf_docs).items():
        filename = key
        response1 = initialize_conversation_chain(get_vectorstore(value))({'question': prompt1})
        dic[filename]=response1
    return dic


# Example usage:
# dic = generate_themes_codes(pdf_docs)





def main():
    st.set_page_config(layout="wide", page_title="Research Bot", page_icon="🤖") 
    st.write(css, unsafe_allow_html=True)
    load_dotenv()

    # Initialize session state variables
    if "themes" not in st.session_state:
        st.session_state.themes = None

    # Display columns 1 and 2
    col1, col2 = st.columns((3, 1))
    with col1:
        st.title("Research Assistant📚")
        #st.write("This is column 1")
        #st.write("Concatenated Text:")

    with col2:
        st.title("Themes and Codes📚")        
        #st.write("This is column 2")
        
    # Display file uploader in the sidebar
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs is not None:  # and vectorstore is not None:
                    st.sidebar.success("Files successfully uploaded!")
                    text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Documents processed successfully!")
                    
                    # Initialize conversation chain
                    st.session_state.conversation = initialize_conversation_chain(vectorstore)
                else:
                    st.warning("Please upload PDFs before processing.")
                      # Assuming you have a function to get pdf_docs
                

                    
                    
        
                #for key, value in get_pdf_textt(pdf_docs).items():
                    #    st.write(key,value)
                
 
                ###for key, value in generate_themes_codes(pdf_docs).items():
                ###    st.write(f"Key : {key} - Value : {value}")

    # Display vectorstore in column 1
    with col1:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if pdf_docs:
            user_question = st.text_input("Ask a question:")
            if st.button("Submit"):
                handle_user_input(user_question, st.session_state.conversation)

    # Display text chunks in column 2
    with col2:
        if st.button("Generate Themes and Codes"):
            with st.spinner("Generating"):
                if pdf_docs:
                    prompt1='give very very long summary of the paper'
                    prompt2='Identify up to 13 most important themes in the text, provide a meaningful name for each theme in 3 words'
                    ###,4 lines meaningful and dense description of the theme?"
                    prompt3='Group the provided codes under related themes and print them. First, print the theme name, then list their related codes one below the other.?'
                    llm = OpenAI(temperature=0.7)
            
                    response1 = initialize_conversation_chain(get_vectorstore(get_text_chunks(get_pdf_text(pdf_docs))))({'question': prompt1})
                    summary = response1['answer']
                    chatgpt_prompt = f"Research paper: {summary}\nQuery: {prompt2}"
                    codes = llm(chatgpt_prompt)
                    chatgpt_prompt = f"Research paper: {codes}\nQuery: {prompt3}"
                    themes = llm(chatgpt_prompt)
                    st.session_state.themes = themes
                    if st.session_state.themes is not None:
                        st.write(st.session_state.themes)
  
if __name__ == "__main__":
    main()


            



        
        
        

