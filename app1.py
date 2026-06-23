import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Tắt telemetry ChromaDB (tránh lỗi posthog)
import re
import shutil
import time
import yaml
from yaml.loader import SafeLoader
from threading import Thread
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import streamlit_authenticator as stauth
# pyrefly: ignore [missing-import]
import pdfplumber

# pyrefly: ignore [missing-import]
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_huggingface.llms import HuggingFacePipeline
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain.text_splitter import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain.chains import RetrievalQA

# ---------------------------------------------------------
# Streamlit Config & Premium CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="HSU Admission Chatbot 2025",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Google Fonts, and Custom Chat Bubbles)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Layout Styling */
    .stApp {
        background-color: #0E131F;
        color: #E2E8F0;
    }
    
    /* Header section */
    .header-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 50%, #EF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #F8FAFC;
        font-size: 1.25rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
        margin-top: 16px;
    }
    
    /* Pulsing Status Dot */
    .status-container {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.5s infinite;
    }
    
    .status-text {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    /* Card design for UI indicators */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .metric-title {
        color: #64748B;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #F1F5F9;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Chat Expander Styling */
    .stExpander {
        background-color: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------
_auth_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth_config.yaml")
with open(_auth_config_path, encoding="utf-8") as _f:
    _auth_config = yaml.load(_f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=_auth_config["credentials"],
    cookie_name=_auth_config["cookie"]["name"],
    cookie_key=_auth_config["cookie"]["key"],
    cookie_expiry_days=_auth_config["cookie"]["expiry_days"],
)

# Hiển thị form đăng nhập
authenticator.login(location="main")

_auth_status = st.session_state.get("authentication_status")
_username    = st.session_state.get("username", "")
_name        = st.session_state.get("name", "")

if _auth_status is False:
    st.error("❌ Tên đăng nhập hoặc mật khẩu không đúng!")
    st.stop()
elif _auth_status is None:
    st.info("ℹ️ Vui lòng đăng nhập để sử dụng Trợ lý Tuyển sinh HSU.")
    st.stop()

# Lấy role từ config
_user_role = _auth_config["credentials"]["usernames"].get(_username, {}).get("role", "guest")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
FILE_PATH = "D:/chatbotdemo2/thongtin8.pdf"
PERSIST_DIR = "D:/chatbotdemo2/chroma_db4"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------
# CACHED RESOURCE LOADERS
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải Embedding Model (bkai-foundation-models/vietnamese-bi-encoder)...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )

@st.cache_resource(show_spinner="Đang tải mô hình LLM (Qwen/Qwen2.5-3B-Instruct ở chế độ 4-bit)...")
def load_llm_model():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

@st.cache_resource(show_spinner="Đang khởi tạo pipeline sinh văn bản...")
def load_pipeline(_model, _tokenizer, max_new_tokens, do_sample, temperature, repetition_penalty):
    """Cache pipeline theo các tham số — chỉ rebuild khi slider thay đổi."""
    return pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=_tokenizer.eos_token_id,
        device_map="auto"
    )

@st.cache_resource(show_spinner="Đang phân tích tài liệu & khởi tạo cơ sở dữ liệu Vector (Chroma DB)...")
def initialize_vector_db(pdf_path, persist_dir, _embeddings):
    # Extract PDF
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page": page_num,
                                "source": pdf_path
                            }
                        )
                    )
    except Exception as e:
        st.error(f"Lỗi khi đọc file PDF {pdf_path}: {e}")
        return None, 0

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " "]
    )
    docs = splitter.split_documents(documents)
    num_chunks = len(docs)

    # Setup Chroma
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=_embeddings
        )
        existing_count = vector_db._collection.count()
        if existing_count == num_chunks:
            return vector_db, num_chunks
        else:
            # Recreate if counts mismatch
            shutil.rmtree(persist_dir)
            
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=_embeddings,
        persist_directory=persist_dir
    )
    return vector_db, num_chunks

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_answer(answer: str) -> str:
    # 1. Strip prompt repetitions
    for marker in ["[Trả lời]", "Answer:", "[Câu hỏi]", "[Tài liệu]", "assistant"]:
        if marker in answer:
            answer = answer.split(marker)[-1]
    
    # Strip ChatML markup tags if leaked
    answer = re.sub(r"<\|im_start\|>assistant|<\|im_end\|>|<\|im_start\|>system|<\|im_start\|>user", "", answer)
    answer = answer.strip()

    # 2. Safely trim unfinished sentence at the end
    lines = answer.split("\n")
    if lines:
        last_line = lines[-1].strip()
        if last_line and not any(last_line.endswith(c) for c in [".", "!", "?", ":", '"', "'"]):
            is_list_item = re.match(r"^(\d+\.|\w\)|-|•|\*)", last_line)
            if not is_list_item and len(last_line) < 50:
                lines.pop()
                answer = "\n".join(lines).strip()
            
    return answer.strip()


def generate_followup_questions(model, tokenizer, user_query: str, answer: str, max_new_tokens: int = 80) -> list[str]:
    """Dùng LLM sinh 2 câu hỏi gợi mở liên quan đến câu trả lời vừa đưa ra."""
    followup_prompt = f"""<|im_start|>system
Bạn là trợ lý tư vấn tuyển sinh HSU. Dựa vào câu hỏi và câu trả lời dưới đây, hãy đề xuất đúng 2 câu hỏi tiếp theo ngắn gọn mà người dùng có thể muốn hỏi thêm.
Yêu cầu:
- Mỗi câu hỏi trên một dòng riêng, bắt đầu bằng dấu "-"
- Ngắn gọn, rõ ràng, liên quan trực tiếp đến chủ đề vừa hỏi
- Chỉ trả về đúng 2 câu hỏi, không giải thích thêm<|im_end|>
<|im_start|>user
Câu hỏi của người dùng: {user_query}
Câu trả lời của chatbot: {answer[:300]}

Đề xuất 2 câu hỏi tiếp theo:<|im_end|>
<|im_start|>assistant
"""
    try:
        inputs = tokenizer(followup_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Chỉ lấy phần mới sinh ra
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        # Parse các dòng bắt đầu bằng "-"
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                q = line.lstrip("-").strip()
                if q:
                    questions.append(q)
        return questions[:2]  # tối đa 2 câu
    except Exception:
        return []


# ---------------------------------------------------------
# SIDEBAR CONTROLS & STATUS
# ---------------------------------------------------------
# --- Role badge ---
_role_labels = {"admin": ("👑 Quản trị viên", "#F59E0B"), "user": ("👨‍🏫 Giáo viên", "#3B82F6"), "guest": ("🎓 Sinh viên", "#10B981")}
_rl, _rc = _role_labels.get(_user_role, ("👤 Khách", "#94A3B8"))
st.sidebar.markdown(
    f"<div style='background:rgba(255,255,255,0.05);border:1px solid {_rc}40;border-radius:8px;"
    f"padding:10px 14px;margin-bottom:12px;'>"
    f"<span style='color:{_rc};font-weight:600;font-size:0.9rem;'>{_rl}</span>"
    f"<br><span style='color:#94A3B8;font-size:0.8rem;'>{_name} (@{_username})</span></div>",
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="status-container"><div class="status-dot"></div><div class="status-text">Hệ thống Sẵn sàng (GPU)</div></div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Cấu hình Hệ thống")

# Reset Vector DB — chỉ admin
if _user_role == "admin":
    if st.sidebar.button("🔄 Làm mới CSDL Vector"):
        if os.path.exists(PERSIST_DIR):
            try:
                shutil.rmtree(PERSIST_DIR)
                st.sidebar.success("Đã xóa CSDL cũ. Đang khởi tạo lại...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Lỗi khi làm mới: {e}")

# Expandable Information Panel — admin + user
if _user_role in ("admin", "user"):
    with st.sidebar.expander("📊 Thông tin Tài liệu & Mô hình", expanded=True):
        st.markdown(f"**Tệp PDF chính**: `{os.path.basename(FILE_PATH)}`")
        st.markdown(f"**LLM Model**: `{MODEL_NAME}`")
        st.markdown(f"**Embedding Model**: `vietnamese-bi-encoder`")
        if torch.cuda.is_available():
            st.markdown(f"**Thiết bị**: `CUDA (GPU: {torch.cuda.get_device_name(0)})`")
        else:
            st.markdown("**Thiết bị**: `CPU`")

# RAG Retriever Settings — chỉ admin
if _user_role == "admin":
    st.sidebar.subheader("🔍 Thiết lập Tìm kiếm (RAG)")
    retriever_k = st.sidebar.slider("Số lượng Chunk lấy ra (k)", min_value=1, max_value=8, value=4)
    retriever_fetch_k = st.sidebar.slider("Số lượng Chunk xét duyệt (fetch_k)", min_value=5, max_value=20, value=10)
    retriever_lambda = st.sidebar.slider("Độ đa dạng MMR (lambda_mult)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
else:
    retriever_k = 4
    retriever_fetch_k = 10
    retriever_lambda = 0.7

# LLM Parameters Settings — chỉ admin
if _user_role == "admin":
    st.sidebar.subheader("🧠 Thiết lập Sinh văn bản (LLM)")
    do_sample = st.sidebar.toggle("Bật Do Sample (Sáng tạo)", value=False)
    llm_temperature = st.sidebar.slider("Nhiệt độ (Temperature)", min_value=0.01, max_value=1.5, value=0.7, step=0.1)
    llm_max_new_tokens = st.sidebar.slider("Số Token tối đa", min_value=64, max_value=1024, value=256, step=64)
    llm_repetition_penalty = st.sidebar.slider("Phạt lặp từ (Repetition Penalty)", min_value=1.0, max_value=2.0, value=1.15, step=0.05)
else:
    do_sample = False
    llm_temperature = 0.7
    llm_max_new_tokens = 256
    llm_repetition_penalty = 1.15

# Memory depth control — admin + user
if _user_role in ("admin", "user"):
    st.sidebar.subheader("💬 Thiết lập Bộ nhớ Hội thoại")
    memory_turns = st.sidebar.slider(
        "Số lượt hội thoại được nhớ",
        min_value=0,
        max_value=10,
        value=3,
        help="0 = không nhớ lịch sử; 10 = nhớ 10 lượt trước"
    )
else:
    memory_turns = 3

# Clear chat button — mọi role
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Xóa lịch sử trò chuyện"):
    _msg_key_clear = f"messages_{_username}"
    st.session_state[_msg_key_clear] = []
    st.rerun()

# Đăng xuất
st.sidebar.markdown("---")
authenticator.logout(button_name="🚪 Đăng xuất", location="sidebar")

# ---------------------------------------------------------
# INITIALIZE SYSTEM COMPONENTS
# ---------------------------------------------------------
embeddings = load_embeddings()
model, tokenizer = load_llm_model()

# Load Vector DB
if os.path.exists(FILE_PATH):
    vector_db, total_chunks = initialize_vector_db(FILE_PATH, PERSIST_DIR, embeddings)
else:
    st.error(f"Không tìm thấy tài liệu PDF tại: {FILE_PATH}")
    st.stop()

# Sidebar display counts
st.sidebar.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Tổng số vector chunks</div>
    <div class="metric-value">{total_chunks} Chunks</div>
</div>
""", unsafe_allow_html=True)

# Create Retriever dynamically based on slider values
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": retriever_k,
        "fetch_k": retriever_fetch_k,
        "lambda_mult": retriever_lambda
    }
)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🎓 Trợ lý Tuyển sinh HSU 2025</div>
    <div class="header-subtitle">Hệ thống Hỏi đáp RAG dựa trên Đề án Tuyển sinh Đại học Hoa Sen 2025. Hãy đặt câu hỏi của bạn bên dưới!</div>
</div>
""", unsafe_allow_html=True)

# Initialize Chat History (riêng theo từng user)
_msg_key = f"messages_{_username}"
if _msg_key not in st.session_state:
    st.session_state[_msg_key] = []

# Initialize pending follow-up click (riêng theo từng user)
_followup_key = f"pending_followup_{_username}"
if _followup_key not in st.session_state:
    st.session_state[_followup_key] = None

# Display conversation
for i, msg in enumerate(st.session_state[_msg_key]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are sources associated with the assistant message, display them
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 Xem nguồn tài liệu tham khảo"):
                for idx, doc in enumerate(msg["sources"], 1):
                    page = doc.get("page", "N/A")
                    st.markdown(f"**Nguồn {idx} (Trang {page}):**")
                    st.code(doc.get("content", ""), language="text")
        # Hiển thị button gợi ý chỉ cho tin nhắn assistant cuối cùng
        if (
            msg["role"] == "assistant"
            and i == len(st.session_state[_msg_key]) - 1
            and "followups" in msg
            and msg["followups"]
        ):
            st.markdown("<div style='margin-top:10px;'><span style='color:#94A3B8;font-size:0.85rem;'>💡 Bạn có thể hỏi thêm:</span></div>", unsafe_allow_html=True)
            cols = st.columns(len(msg["followups"]))
            for col, suggestion in zip(cols, msg["followups"]):
                with col:
                    if st.button(f"❓ {suggestion}", key=f"followup_{i}_{suggestion[:20]}", use_container_width=True):
                        st.session_state[_followup_key] = suggestion
                        st.rerun()

# Xử lý click button gợi ý hoặc nhập thường
if st.session_state[_followup_key]:
    user_query = st.session_state[_followup_key]
    st.session_state[_followup_key] = None
elif typed_query := st.chat_input("Nhập câu hỏi tuyển sinh của bạn ở đây..."):
    user_query = typed_query
else:
    user_query = None

if user_query:
    # Display user query
    st.chat_message("user").markdown(user_query)
    st.session_state[_msg_key].append({"role": "user", "content": user_query})

    # Prepare response placeholder
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        spinner_placeholder = st.empty()
        
        with spinner_placeholder.container():
            st.markdown("*Đang tìm kiếm thông tin và suy nghĩ...*")
        
        try:
            # 1. Retrieve relevant contexts
            retrieved_docs = retriever.invoke(user_query)

            # ── Bước 1b: Xây dựng lịch sử hội thoại từ session_state của user hiện tại ──
            history_messages = st.session_state[_msg_key][:-1]  # bỏ câu hỏi hiện tại
            # Giữ lại tối đa memory_turns lượt (mỗi lượt gồm 2 tin nhắn: user + assistant)
            max_history_msgs = memory_turns * 2
            if max_history_msgs > 0:
                history_messages = history_messages[-max_history_msgs:]
            else:
                history_messages = []

            # Định dạng lịch sử thành chuỗi văn bản
            history_text = ""
            if history_messages:
                history_lines = []
                for msg in history_messages:
                    role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                    history_lines.append(f"{role_label}: {msg['content']}")
                history_text = "\n".join(history_lines)

            # ── Prompt template có tích hợp lịch sử hội thoại ──
            if history_text:
                CUSTOM_PROMPT_TEMPLATE = """<|im_start|>system
Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Hoa Sen (HSU).
Nhiệm vụ của bạn là trả lời câu hỏi DỰA HOÀN TOÀN vào nội dung tài liệu bên dưới.

QUY TẮC NGHIÊM NGẶT:
1. Chỉ dùng thông tin có trong [Tài liệu]. Không thêm, không suy đoán.
2. Nếu tài liệu không đề cập → trả lời: "Tôi không tìm thấy thông tin này trong tài liệu."
3. Trả lời bằng tiếng Việt, ngắn gọn, liệt kê rõ ràng nếu có nhiều mục.
4. Không lặp lại câu hỏi trong câu trả lời.
5. Diễn đạt lại bằng lời của bạn, không sao chép nguyên văn tài liệu.
6. Sử dụng [Lịch sử trò chuyện] để hiểu ngữ cảnh của câu hỏi hiện tại.

[Lịch sử trò chuyện]
{history}

[Tài liệu]
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
                custom_prompt = PromptTemplate(
                    input_variables=["history", "context", "question"],
                    template=CUSTOM_PROMPT_TEMPLATE
                )
            else:
                CUSTOM_PROMPT_TEMPLATE = """<|im_start|>system
Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Hoa Sen (HSU).
Nhiệm vụ của bạn là trả lời câu hỏi DỰA HOÀN TOÀN vào nội dung tài liệu bên dưới.

QUY TẮC NGHIÊM NGẶT:
1. Chỉ dùng thông tin có trong [Tài liệu]. Không thêm, không suy đoán.
2. Nếu tài liệu không đề cập → trả lời: "Tôi không tìm thấy thông tin này trong tài liệu."
3. Trả lời bằng tiếng Việt, ngắn gọn, liệt kê rõ ràng nếu có nhiều mục.
4. Không lặp lại câu hỏi trong câu trả lời.
5. Diễn đạt lại bằng lời của bạn, không sao chép nguyên văn tài liệu.

[Tài liệu]
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
                custom_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=CUSTOM_PROMPT_TEMPLATE
                )

            # ── Bước 2: Lấy pipeline đã cache (chỉ rebuild khi slider thay đổi) ──
            model_pipeline = load_pipeline(
                model, tokenizer,
                llm_max_new_tokens, do_sample,
                llm_temperature, llm_repetition_penalty
            )

            # ── Bước 3: Streaming với TextIteratorStreamer ──
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Xây dựng context từ retrieved docs
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Tạo full_prompt (có hoặc không có lịch sử)
            if history_text:
                full_prompt = CUSTOM_PROMPT_TEMPLATE.format(
                    history=history_text,
                    context=context_text,
                    question=user_query
                )
            else:
                full_prompt = CUSTOM_PROMPT_TEMPLATE.format(
                    context=context_text,
                    question=user_query
                )
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=llm_max_new_tokens,
                do_sample=do_sample,
                temperature=llm_temperature if do_sample else None,
                repetition_penalty=llm_repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

            # Chạy generation trong thread riêng để không block UI
            spinner_placeholder.empty()
            start_time = time.time()
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream từng token ra màn hình
            generated = ""
            for new_text in streamer:
                generated += new_text
                response_placeholder.markdown(generated + "▌")
            thread.join()
            end_time = time.time()

            # Xóa con trỏ nhấp nháy, hiển thị kết quả cuối
            answer = clean_answer(generated)
            response_placeholder.markdown(answer)

            source_docs = retrieved_docs
            
            # Form clean list of source documents for session state
            sources_list = []
            for doc in source_docs:
                sources_list.append({
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content
                })
            
            # Show output
            spinner_placeholder.empty()
            response_placeholder.markdown(answer)
            
            # Show sources in expander
            if sources_list:
                with st.expander("🔍 Xem nguồn tài liệu tham khảo"):
                    for idx, doc in enumerate(sources_list, 1):
                        st.markdown(f"**Nguồn {idx} (Trang {doc['page']}):**")
                        st.code(doc["content"], language="text")
            
            # Save assistant message to session state
            # Sinh câu hỏi gợi mở
            with st.spinner("💡 Đang tạo câu hỏi gợi ý..."):
                followup_questions = generate_followup_questions(
                    model, tokenizer, user_query, answer
                )

            st.session_state[_msg_key].append({
                "role": "assistant",
                "content": answer,
                "sources": sources_list,
                "followups": followup_questions
            })

            # Hiển thị button gợi ý ngay sau câu trả lời
            if followup_questions:
                st.markdown("<div style='margin-top:10px;'><span style='color:#94A3B8;font-size:0.85rem;'>💡 Bạn có thể hỏi thêm:</span></div>", unsafe_allow_html=True)
                cols = st.columns(len(followup_questions))
                for col, suggestion in zip(cols, followup_questions):
                    with col:
                        if st.button(f"❓ {suggestion}", key=f"new_{suggestion[:20]}", use_container_width=True):
                            st.session_state[_followup_key] = suggestion
                            st.rerun()

        except Exception as e:
            spinner_placeholder.empty()
            error_message = f"❌ Lỗi khi xử lý câu hỏi: {str(e)}"
            st.error(error_message)
            st.session_state[_msg_key].append({"role": "assistant", "content": error_message})
