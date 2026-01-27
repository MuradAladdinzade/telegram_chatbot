from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings


from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS




## number of pages from the beginning, can be adjusted as needed
## Please be aware of pricing when increasing this number
pages_RAG = 10 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")




# Load PDF
all_text = PyMuPDFLoader("text.pdf").load()

## Take only first 10 pages for testing
## adjust as needed
till_pages = all_text[:pages_RAG]


# chunking
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(till_pages)

vectordb = FAISS.from_documents(docs, embeddings_model)
print(vectordb.index.ntotal)

vectordb.save_local("./faiss_index")


