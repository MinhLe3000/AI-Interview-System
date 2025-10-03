import keyboard
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from GetApikey import loadapi

API_KEY=loadapi()

# Embeddings Google
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# Load FAISS database đã lưu
db = FAISS.load_local("vector_db2chunk_nltk", embeddings, allow_dangerous_deserialization=True)

# Tạo retriever từ FAISS
retriever = db.as_retriever(search_kwargs={"k": 5})

# LLM Google Gemini (text-only)
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.5
)

# Prompt cho RAG
prompt_template = """
Bạn là một trợ lý AI. Hãy sử dụng thông tin trong context để trả lời câu hỏi của người dùng.
Nếu không tìm thấy thông tin hoặc cảm giác thông tin không được liên quan, hãy nói công cụ truy vấn đươc thông tin không liên quan và giải thích tại sao "

Context: {context}

Câu hỏi: {question}

Trả lời:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Memory lưu lịch sử hội thoại
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# Tạo RetrievalQA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Vòng lặp chat
print("💬 Chat với Java RAG Bot (gõ 'exit' để thoát)\n")
while True:
    query = input("❓Bạn: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Kết thúc chat.")
        break
    # exit khi nhấn 'Esc'
    if keyboard.is_pressed('esc'):
        print("👋 Kết thúc chat.")
        break

    result = qa_chain.invoke({"question": query})
    print("🤖 Bot:", result["answer"])


