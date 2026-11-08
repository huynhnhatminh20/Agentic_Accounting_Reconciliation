# 🏛️ Hệ Thống Đối Soát & Báo Cáo Kế Toán Tự Động (Agentic Accounting Reconciliation)

> **Hệ thống AI Kế toán Trưởng thông minh** hỗ trợ tự động đối soát sao kê ngân hàng, trích xuất & kiểm toán hóa đơn đa phương thức (XML/PDF/Ảnh OCR), tự chữa lỗi số học, hạch toán định khoản tự động chuẩn **Thông tư 200 & Thông tư 133/BTC**, truy vấn tài chính bằng ngôn ngữ tự nhiên (**Text-to-SQL Self-Healing**) và xuất khẩu báo cáo **Excel chuẩn MISA AMIS**.

---

## 🌟 Tính Năng Nổi Bật

### 1. 📑 Nhận Diện Đa Kênh & Tự Kiểm Toán Hóa Đơn (Multi-modal Self-Auditing OCR)
- Hỗ trợ toàn diện 3 định dạng hóa đơn phổ biến tại Việt Nam: **XML điện tử**, **PDF trích xuất văn bản**, và **Ảnh chụp/Quét (JPG/PNG)**.
- Kết hợp **VietOCR (VGG-Transformer)** và **PaddleOCR** cho tiếng Việt có dấu.
- **Kiểm định toán học tự động:** So sánh phương trình kế toán:
  $$\text{Tổng tiền thanh toán} = \text{Tiền hàng trước thuế} + \text{Tiền thuế GTGT}$$
- **Cơ chế Tự chữa lỗi OCR (Self-Healing Levenshtein):** Tự động phát hiện lỗi đọc nhầm ký tự số tương đồng (như $1 \leftrightarrow 8$, $0 \leftrightarrow 8$) và đề xuất giá trị chính xác theo kiểm toán số học.

### 2. ⚡ Kiến Trúc Đối Soát Kép: System 1 (Fast) & System 2 (ReAct Agent)
- **System 1 (Fast Symbolic Engine):** Xử lý tức thì các giao dịch chuyển khoản chuẩn 1-1 khớp chính xác số tiền và số hóa đơn.
- **System 2 (Autonomous ReAct Investigator):** Khi gặp giao dịch ngoại lệ, kích hoạt Agent LLM với chuỗi suy luận (*Thought $\rightarrow$ Action $\rightarrow$ Observation*) cùng bộ công cụ chuyên sâu:
  - `tool_solve_subset_sum`: Thuật toán nhánh cận giải bài toán 1 khoản thanh toán gom cho nhiều hóa đơn (1-to-Many).
  - `tool_lookup_alias`: Tra cứu danh bạ người ủy quyền (Giám đốc, Kế toán chuyển tiền thay công ty).
  - `tool_calculate_deductions`: Tự động tính toán chiết khấu thanh toán theo hợp đồng và bù trừ phí chuyển tiền ngân hàng.
  - `tool_recheck_ocr_discrepancy`: Điều tra đối chiếu ngược lại ảnh hóa đơn gốc khi phát hiện chênh lệch số dư.

### 3. ⚖️ Định Khoản Kế Toán Kép Cân Đối Tuyệt Đối (VAS TT 200/133)
- Tự động sinh bút toán kép Nợ/Có chuẩn tài khoản kế toán Việt Nam:
  - **TK 1121** (Tiền gửi ngân hàng)
  - **TK 131** (Phải thu của khách hàng theo từng đối tượng chi tiết)
  - **TK 635** (Chi phí tài chính - Chiết khấu thương mại, thanh toán)
  - **TK 6425** (Chi phí quản lý doanh nghiệp - Thuế, phí ngân hàng)
  - **TK 515** (Doanh thu hoạt động tài chính)
- Đảm bảo **100% nguyên tắc cân đối kế toán:** $\sum \text{Nợ} = \sum \text{Có}$.

### 4. 💬 Truy Vấn Tài Chính Bằng Ngôn Ngữ Tự Nhiên (Self-Healing Text-to-SQL)
- Kế toán viên có thể nhập câu hỏi tự do bằng tiếng Việt (ví dụ: *"Top 3 khách hàng nợ nhiều nhất"*, *"Liệt kê hóa đơn quá hạn của KH001"*, *"Có bao nhiêu hóa đơn thuế 8%"*).
- Sử dụng mô hình LLM cục bộ (`qwen2.5:3b`) qua Ollama để sinh câu lệnh SQLite SELECT.
- **Cơ chế Self-Healing:** Tự động sửa lỗi cú pháp SQLite, lọc chuẩn xác hạn nợ, và tự động nới lỏng điều kiện lọc nếu kết quả rỗng. Tự động dự phòng về Canonical Queries thông minh nếu LLM không khả dụng.

### 5. 📊 Báo Cáo Tuổi Nợ (Aging Report) & Xuất Excel MISA AMIS
- Phân tích tuổi nợ chi tiết theo 5 phân nhóm chuẩn quốc tế: *Chưa đến hạn, Quá hạn 1-30 ngày, 31-60 ngày, 61-90 ngày, và >90 ngày*.
- Xuất khẩu workbook Excel 3 Sheets chuyên nghiệp tương thích trực tiếp cấu trúc import chứng từ **MISA AMIS**:
  1. `Bang_Doi_Soat_Cong_No`: Bảng kê kết quả đối soát có tô màu phân biệt System 1 / System 2.
  2. `So_Dinh_Khoan_MISA`: Bảng định khoản hạch toán chi tiết sẵn sàng nhập khẩu vào phần mềm kế toán.
  3. `Bao_Cao_Cong_No_Phai_Thu`: Bảng tổng hợp công nợ chi tiết từng khách hàng kèm tỷ lệ chiết khấu và hạn tín dụng.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
agentic/
├── app.py                     # Giao diện người dùng Web Streamlit (4 Tabs nghiệp vụ)
├── config.py                  # Cấu hình trung tâm (Database, Ollama, Hằng số VAS)
├── setup_checkpoints.py       # Script tự động tải Model Weights (VietOCR, PhoBERT, Ollama)
├── requirements.txt           # Danh mục thư viện Python cần thiết
├── README.md                  # Tài liệu hướng dẫn sử dụng
├── .gitignore                 # Cấu hình bỏ qua file nhị phân nặng & cache
│
├── database/                  # Quản lý cơ sở dữ liệu SQLite
│   ├── db_manager.py          # Lớp kết nối, execute transaction an toàn
│   └── seed_mock_data.py      # Sinh dữ liệu mẫu chuẩn 5 kịch bản đối soát kế toán
│
├── modules/                   # Các module lõi của hệ thống
│   ├── ocr_agent.py           # Bóc tách hóa đơn XML/PDF/Ảnh & Kiểm toán số học
│   ├── system1_engine.py      # Bộ máy đối soát nhanh 1-1 (Fast Symbolic Engine)
│   ├── system2_agent.py       # Agent ReAct điều tra ngoại lệ tự động (Subset-Sum, Alias, Fee)
│   ├── w2ner_adapter.py       # Adapter trích xuất thực thể sao kê ngân hàng
│   └── sql_reporting_agent.py # Text-to-SQL Self-Healing, Báo cáo Tuổi nợ & Xuất Excel MISA
│
├── weights/                   # Thư mục lưu trữ model checkpoints (tải qua script)
│   ├── ocr/                   # Trọng số VietOCR (vgg_transformer.pth)
│   ├── phobert-base-v2/       # Mô hình nền tảng PhoBERT (safetensors)
│   └── w2ner/                 # Thư mục chứa trọng số W2NER tùy chỉnh
│
└── uploads/                   # Thư mục tạm thời lưu hóa đơn tải lên & báo cáo Excel xuất ra
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 1. Yêu Cầu Môi Trường
- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux.
- **Python:** Phiên bản `3.10` trở lên (khuyên dùng Python 3.10 hoặc 3.11).
- **Phần cứng:** Khuyên dùng GPU có từ 4GB VRAM trở lên (hỗ trợ hoàn hảo trên GPU Laptop như RTX 3050).

### 2. Cài Đặt Thư Viện Python
Mở Terminal / PowerShell tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### 3. Tự Động Tải Checkpoints & Weights
Chạy script tự động tải các model công khai (VietOCR, PhoBERT, cấu hình W2NER):
```bash
python setup_checkpoints.py
```

### 4. Khởi Động Ollama Server (Dành cho LLM Agent)
Cài đặt [Ollama](https://ollama.com/), sau đó khởi động server và tải mô hình `qwen2.5:3b`:
```bash
# Khởi động dịch vụ Ollama
ollama serve

# Tải mô hình Qwen 2.5 3B (nhẹ 1.9GB, chạy cực mượt trên GPU 4GB)
ollama pull qwen2.5:3b
```

### 5. Khởi Tạo Cơ Sở Dữ Liệu Kế Toán Mẫu
Khởi tạo cấu trúc bảng SQLite và nạp dữ liệu mô phỏng 5 kịch bản thực tế:
```bash
python -c "from database.db_manager import DatabaseManager; from database.seed_mock_data import seed_all_data; seed_all_data(DatabaseManager())"
```

### 6. Khởi Chạy Giao Diện Web Streamlit
Khởi chạy ứng dụng Web trực quan:
```bash
streamlit run app.py
```
Sau khi chạy, trình duyệt sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🖥️ Hướng Dẫn Sử Dụng Các Tab Chức Năng

1. **Tab 1 — Bảng Điều Khiển Tổng Quan (Dashboard):**
   - Theo dõi tiến độ đối soát theo thời gian thực (tổng số giao dịch, tỷ lệ tự động hóa, tổng tiền thu chi).
   - Xem ma trận 2D Word-Word W2NER và đồ thị phân bổ trạng thái dòng tiền.
2. **Tab 2 — Trích Xuất & Kiểm Toán Hóa Đơn (OCR & Audit):**
   - Tải lên file hóa đơn XML, PDF hoặc Ảnh chụp.
   - Hệ thống tự động bóc tách số HĐ, MST, ngày tháng, tiền hàng, tiền thuế và chạy kiểm tra số học Self-Healing.
3. **Tab 3 — Đối Soát & Phê Duyệt Giao Dịch (Reconciliation):**
   - Xem chi tiết từng giao dịch được System 1 và System 2 giải quyết.
   - Thẩm định giải trình của ReAct Agent (nguyên nhân chiết khấu, giải bài toán Subset-sum, danh bạ đại diện).
   - Nút **"Phê duyệt (Approve)"** và **"Phê duyệt hàng loạt"** để chốt sổ kế toán.
4. **Tab 4 — Báo Cáo Kế Toán & Text-to-SQL (Analytics):**
   - Gõ bất kỳ câu hỏi tài chính nào bằng tiếng Việt để xem kết quả và biểu đồ Plotly tương tác.
   - Xem phân tích báo cáo tuổi nợ chi tiết (Aging Report).
   - Nhấn nút **"Xuất File Excel MISA AMIS"** để tải về file sổ sách kế toán hoàn chỉnh.

---

## 📜 Giấy Phép & Tiêu Chuẩn Áp Dụng
- Hệ thống tuân thủ nghiêm ngặt **Hệ thống Chuẩn mực Kế toán Việt Nam (VAS)** theo **Thông tư 200/2014/TT-BTC** và **Thông tư 133/2016/TT-BTC**.
- Phù hợp ứng dụng làm đồ án tốt nghiệp, hệ thống PoC tự động hóa tài chính doanh nghiệp vừa và nhỏ (SME), hoặc giải pháp tích hợp phần mềm kế toán.
