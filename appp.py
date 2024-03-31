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
header = {
    "authorization" : st.secrets["OPENAI_API_KEY"],
    "content-type" : "application/json"
}
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

def generate_themes_codes(vectorstore):
    prompt1='give very very long summary of the paper '
    prompt2='Identify up to 13 most important themes in the text, provide a meaningful name for each theme in 3 words'
    ###,4 lines meaningful and dense description of the theme?"
    prompt3='Group the provided codes under related themes and print them. First, print the theme name, then list their related codes one below the other.?'
    llm = OpenAI(temperature=0.7)
    if st.session_state.conversation0 is None:
        st.session_state.conversation0 = initialize_conversation_chain(vectorstore)
    response = st.session_state.conversation0({'question': prompt1})
    chatgpt_prompt = f"Research paper: {response['answer']}\nQuery: {prompt2}"
    chatgpt_responsee = llm(chatgpt_prompt)
    
    theme = f"Research paper: {chatgpt_responsee}\nQuery: {prompt3}"

    # Use the OpenAI language model to generate a response
    chatgpt_responseee = llm(theme)
    

    
    return chatgpt_responseee#['answer']




# Example usage:
# dic = generate_themes_codes(pdf_docs)
def print_themes_codes(pdf_docs):
    if "themes" not in st.session_state or st.session_state.themes is None:
        st.session_state.themes = {}
        
    pdf_texts = get_pdf_textt(pdf_docs)
    for item, value in pdf_texts.items():
        filename = item
        text_chunks = get_text_chunks(value)
        vectorstore = get_vectorstore(text_chunks)
        if filename not in st.session_state.themes:
            st.session_state.themes[filename] = generate_themes_codes(vectorstore)
        
    # Display themes and codes
    for item, value in st.session_state.themes.items():
        st.subheader(f"File : {item}", divider='rainbow')
        st.write(value)



def submit():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""
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
        st.title("Themes and Codes⚙️")        
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
                    st.session_state.conversation = initialize_conversation_chain(vectorstore)
                    
                   
                
                
             

    # Display vectorstore in column 1
    with col1:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if "conversation0" not in st.session_state:
            st.session_state.conversation0 = None
        if "user_question" not in st.session_state:
            st.session_state.user_question = ""
    
        if pdf_docs:
            user_question = st.text_input("Ask a question:", key="widget", on_change=submit)
            user_question = st.session_state.user_question
            if user_question:
                handle_user_input(user_question,st.session_state.conversation)#, st.session_state.conversation)

    # Display text chunks in column 2
    with col2:
        if st.button("Generate Themes and Codes"):
            with st.spinner("Generating"):
                print_themes_codes(pdf_docs)
  
if __name__ == "__main__":
    main()
