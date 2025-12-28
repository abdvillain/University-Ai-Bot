import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load the AI Brain (Vector Store)
# We use the same model to understand the user's question
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# We load the saved database
# allow_dangerous_deserialization=True is needed to trust our own local file
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_answer(user_query):
    # Search for the top 1 most similar question/answer pair
    docs = vector_db.similarity_search(user_query, k=1)
    
    if docs:
        # Return the content found in the CSV
        return docs[0].page_content
    else:
        return "Sorry, I couldn't find an answer in the data."

# 2. Build the Website Interface
interface = gr.Interface(
    fn=get_answer,                 # The function to run
    inputs="text",                 # User inputs text
    outputs="text",                # AI outputs text
    title="University AI Assistant",
    description="Ask me about library hours, admissions, or exams! (Built with RAG)"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
