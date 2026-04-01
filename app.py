import streamlit as st
import os
import glob
import re
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from google import genai

# --- CẤU HÌNH ---
st.set_page_config(page_title="Hệ thống Trợ lý Luật 2025", layout="centered", page_icon="⚖️")
st.title("⚖️ Chatbot tra cứu Luật An ninh mạng")
st.caption("Sinh viên thực hiện: Phạm Trung Kiên - MSSV: 2251120094")

API_KEY = "Api cua ban"
client = genai.Client(api_key=API_KEY)

# --- 1. XỬ LÝ DỮ LIỆU & CHROMA DB ---
@st.cache_resource
def setup_vector_db():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "Data")
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    
    if not pdf_files:
        return None, []

    try:
        loader = PyPDFLoader(pdf_files[0])
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        full_text = re.sub(r'Trang \d+|\d+/\d+', '', full_text) 

        # SỬA LỖI REGEX: Bỏ bắt buộc dấu chấm để không bỏ sót Điều nào
        raw_chunks = re.split(r'(?=Điều\s+\d+)', full_text)
        
        valid_chunks = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if re.match(r'^Điều\s+\d+', chunk, re.IGNORECASE):
                clean_chunk = re.sub(r'\s+', ' ', chunk)
                valid_chunks.append(clean_chunk)

        chroma_client = chromadb.Client()
        try:
            chroma_client.delete_collection(name="law_collection")
        except:
            pass
            
        collection = chroma_client.create_collection(name="law_collection")
        collection.add(
            documents=valid_chunks,
            ids=[f"dieu_{i}" for i in range(len(valid_chunks))]
        )
        # Trả về cả collection và danh sách chunks để làm Hybrid Search
        return collection, valid_chunks
    except Exception as e:
        return None, []

collection, valid_chunks = setup_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. LOGIC TRUY XUẤT & TRẢ LỜI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi về các quy định pháp luật..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not collection:
            st.error("Dữ liệu chưa sẵn sàng hoặc không tìm thấy file PDF.")
        else:
            with st.spinner("Đang phân tích chuyên sâu..."):
                
                relevant_chunks_list = []
                
                # --- HYBRID SEARCH: TÌM KIẾM CHÍNH XÁC (Trị bệnh mù số của AI) ---
                # Bắt từ khóa "điều X" trong câu hỏi người dùng
                match = re.search(r'điều\s+(\d+)', prompt, re.IGNORECASE)
                if match:
                    so_dieu = match.group(1)
                    for chunk in valid_chunks:
                        # Nếu tìm thấy đúng Điều đó, nhét ngay lên đầu danh sách
                        if re.match(rf'^Điều\s+{so_dieu}\b', chunk, re.IGNORECASE):
                            relevant_chunks_list.append(chunk)
                            break
                            
                # --- TÌM KIẾM NGỮ NGHĨA BẰNG CHROMADB ---
                results = collection.query(query_texts=[prompt], n_results=6)
                for doc in results['documents'][0]:
                    if doc not in relevant_chunks_list:
                        relevant_chunks_list.append(doc)

                # Nối tất cả các tài liệu tìm được lại
                relevant_chunks = "\n\n".join(relevant_chunks_list)

                instruction = (
                    "Bạn là một chuyên gia pháp lý cấp cao về An ninh mạng. "
                    "Nhiệm vụ của bạn là phân tích và trả lời câu hỏi một cách chi tiết, mạch lạc. "
                    "Luôn trích dẫn rõ 'Căn cứ theo Điều...'. "
                    "Tuyệt đối không bịa đặt. Nếu thông tin không có, hãy báo không có."
                )
                
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[f"NGỮ CẢNH LUẬT:\n{relevant_chunks}\n\nCÂU HỎI:\n{prompt}"],
                        config={'system_instruction': instruction}
                    )
                    
                    answer = response.text
                    st.markdown(answer)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Lỗi AI: {str(e)}")
