# 🎓 HSU Admission Chatbot 2025 - Trợ lý Tư vấn Tuyển sinh Đại học Hoa Sen

Ứng dụng **HSU Admission Chatbot 2025** là một hệ thống hỏi đáp thông minh sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** chạy hoàn toàn local (hoặc sử dụng GPU cá nhân). Hệ thống cho phép người dùng hỏi đáp về Đề án Tuyển sinh Đại học Hoa Sen năm 2025 dựa trên nội dung tệp tài liệu PDF chính thức (`thongtin8.pdf`).

Giao diện được xây dựng bằng **Streamlit** với phong cách thiết kế hiện đại (Glassmorphism, Dark mode) tối ưu trải nghiệm người dùng và hỗ trợ phản hồi dạng streaming (luồng chữ chạy trực tiếp).

---

## 🚀 Tính năng nổi bật

1. **Hỏi đáp RAG thông minh (Strict Constraints):**
   - Trả lời câu hỏi **dựa hoàn toàn** vào nội dung tài liệu đính kèm.
   - Tránh hiện tượng ảo tưởng (hallucination) của LLM bằng các quy tắc ràng buộc nghiêm ngặt trong Prompt.
   - Nếu tài liệu không có thông tin, chatbot sẽ phản hồi rõ ràng là không tìm thấy thay vì tự suy diễn.

2. **Phản hồi thời gian thực (Real-time Streaming):**
   - Sử dụng `TextIteratorStreamer` của Hugging Face kết hợp với lập trình đa luồng (`threading.Thread`) giúp hiển thị câu trả lời dạng gõ chữ trực tiếp lên giao diện Streamlit mà không bị nghẽn (block) UI.

3. **Trích dẫn nguồn tài liệu (Source Attribution):**
   - Hiển thị trực quan các phân đoạn văn bản nguồn (chunks) được tìm thấy kèm theo **số trang** cụ thể trong file PDF để người dùng dễ dàng đối chiếu tính xác thực.

4. **Giao diện quản trị & Cấu hình trực quan (Sidebar Control Panel):**
   - **Trạng thái hệ thống:** Đèn LED nhấp nháy hiển thị trạng thái sẵn sàng và thiết bị xử lý hiện tại (GPU/CPU).
   - **Quản lý Vector DB:** Nút bấm làm mới và tái khởi tạo cơ sở dữ liệu Vector nhanh chóng.
   - **Tùy chỉnh tham số tìm kiếm (RAG):**
     - Số lượng phân đoạn văn bản cần lấy ra ($k$).
     - Số lượng phân đoạn xét duyệt ban đầu ($fetch\_k$).
     - Độ đa dạng thuật toán MMR ($lambda\_mult$).
   - **Tùy chỉnh tham số sinh văn bản (LLM):**
     - Tắt/Bật tính năng sinh văn bản ngẫu nhiên/sáng tạo (`do_sample`).
     - Nhiệt độ độ sáng tạo (`temperature`).
     - Số lượng token tối đa cho câu trả lời (`max_new_tokens`).
     - Phạt lặp từ (`repetition_penalty`).

---

## 🛠️ Kiến trúc & Công nghệ sử dụng

Hệ thống được phát triển dựa trên các thư viện mã nguồn mở hàng đầu trong lĩnh vực AI/LLMs:

- **Frontend / UI:** [Streamlit](https://streamlit.io/) (được tinh chỉnh CSS nâng cao với font chữ Google Fonts `Inter`, bảng màu tối sang trọng và hiệu ứng kính mờ glassmorphism).
- **Mô hình ngôn ngữ lớn (LLM):** `Qwen/Qwen2.5-3B-Instruct`
  - Được cấu hình định dạng **4-bit** (`BitsAndBytesConfig` với kiểu định lượng `nf4` và sử dụng `bfloat16`) giúp tiết kiệm tối đa tài nguyên phần cứng (VRAM) nhưng vẫn giữ được độ chính xác và khả năng hiểu tiếng Việt tốt.
- **Mô hình nhúng (Embeddings):** `bkai-foundation-models/vietnamese-bi-encoder`
  - Mô hình chuyên dụng cho việc nhúng từ và câu tiếng Việt của BKAI giúp tìm kiếm ngữ nghĩa chính xác cao.
- **Cơ sở dữ liệu Vector:** [Chroma DB](https://github.com/chroma-core/chroma)
  - Quản lý và lưu trữ các vector nhúng cục bộ tại thư mục `chroma_db3`.
- **Xử lý tài liệu:** `pdfplumber` dùng để trích xuất văn bản từ tệp PDF có độ chính xác cao và giữ nguyên số trang.
- **Khung RAG workflow:** [LangChain](https://python.langchain.com/) (các module Text Splitter, Document Loader, Hugging Face Integrations, v.v.).

---

## 📋 Yêu cầu hệ thống

* **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux.
* **Bộ vi xử lý:** Khuyến nghị máy tính có card đồ họa **NVIDIA GPU** (hỗ trợ CUDA) với tối thiểu **6GB VRAM** để chạy mô hình `Qwen2.5-3B` và model Embeddings mượt mà ở chế độ 4-bit.
* **Bộ nhớ RAM:** Tối thiểu 16GB RAM hệ thống.
* **Môi trường:** Python 3.10 trở lên.

---

## 💻 Hướng dẫn cài đặt

Làm theo các bước sau để thiết lập môi trường chạy ứng dụng trên máy tính của bạn:

### 1. Chuẩn bị thư mục và kích hoạt môi trường ảo

Mở terminal tại thư mục dự án và khởi tạo môi trường ảo Python:

```powershell
# Tạo môi trường ảo (venv)
python -m venv venv

# Kích hoạt môi trường ảo trên Windows (PowerShell)
.\venv\Scripts\activate

# Hoặc trên Linux/macOS
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết

Cài đặt PyTorch hỗ trợ CUDA (nếu máy bạn có GPU NVIDIA). Ví dụ cài đặt phiên bản CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Sau đó cài đặt các thư viện bổ trợ của dự án:

```bash
pip install streamlit pdfplumber transformers bitsandbytes accelerate langchain langchain-community langchain-huggingface langchain-chroma langchain-core sentence-transformers
```

*(Hoặc bạn có thể cài đặt trực tiếp từ file `requirements.txt` có sẵn bằng lệnh: `pip install -r requirements.txt`)*

---

## ⚙️ Cấu hình Đường dẫn trong Code

Trong file [app2.py](file:///d:/chatbotdemo2/app2.py), vui lòng kiểm tra và cập nhật các đường dẫn hằng số sau cho phù hợp với máy tính của bạn (từ dòng 175):

```python
# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
FILE_PATH = "D:/chatbotdemo2/thongtin8.pdf"      # Đường dẫn tới tệp PDF đề án tuyển sinh
PERSIST_DIR = "D:/chatbotdemo2/chroma_db3"     # Thư mục lưu trữ cơ sở dữ liệu Vector Chroma DB
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"         # Tên mô hình LLM trên Hugging Face
```

---

## 🏃 Chạy ứng dụng

Khi môi trường ảo đang được kích hoạt, khởi chạy ứng dụng bằng Streamlit bằng lệnh sau:

```bash
streamlit run app2.py
```

Ứng dụng sẽ tự động tải các mô hình (Embedding và LLM) từ Hugging Face trong lần chạy đầu tiên (quá trình này có thể mất vài phút tùy thuộc vào tốc độ internet của bạn). Sau đó, nó sẽ phân tích file PDF và khởi tạo Chroma DB.

Khi hoàn tất, trình duyệt web của bạn sẽ tự động mở trang ứng dụng tại địa chỉ mặc định:
👉 `http://localhost:8501`

---

## 📂 Cấu trúc mã nguồn của `app2.py`

* **Streamlit Config & Premium CSS Styling (Dòng 35-172):** Định cấu hình ứng dụng, chèn CSS tùy chỉnh giao diện tối, hiệu ứng nhấp nháy trạng thái và cấu trúc giao diện glassmorphism.
* **Cached Resource Loaders (Dòng 185-273):**
  * `load_embeddings()`: Tải và lưu cache mô hình nhúng tiếng Việt.
  * `load_llm_model()`: Tải mô hình Qwen 2.5 3B cấu hình lượng tử hóa 4-bit.
  * `load_pipeline()`: Tạo pipeline sinh văn bản dựa trên cấu hình người dùng.
  * `initialize_vector_db()`: Đọc PDF bằng `pdfplumber`, cắt đoạn văn bản với kích thước chunk 1500 ký tự (đè gối 300 ký tự) và ghi vào Chroma DB.
* **Helper Functions (Dòng 277-298):** `clean_answer()` dùng để dọn dẹp các thẻ đánh dấu đặc biệt của định dạng ChatML (như `<|im_start|>`, `<|im_end|>`) và cắt bỏ các câu dở dang ở cuối nếu LLM bị hết token đột ngột.
* **Sidebar Controls & Status (Dòng 300-376):** Hiển thị chi tiết phần cứng, cung cấp các thanh trượt điều khiển tham số RAG & LLM.
* **Main Interface (Dòng 378-526):** Quản lý luồng chat chính, vẽ lịch sử hội thoại từ `st.session_state.messages`, thực hiện tìm kiếm vector bằng thuật toán MMR và kích hoạt sinh văn bản đa luồng dạng streaming với `TextIteratorStreamer`.

---

*Chúc bạn có trải nghiệm tuyệt vời khi sử dụng **HSU Admission Chatbot 2025**!*
