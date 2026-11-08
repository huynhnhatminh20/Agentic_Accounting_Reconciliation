# ============================================================
# config.py — Cấu hình Trung tâm Hệ thống Đối soát Kế toán
# Thông tư 200 / Thông tư 133 (Bộ Tài chính Việt Nam)
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------------
# 1. ĐƯỜNG DẪN HỆ THỐNG
# ------------------------------------------------------------------

# Thư mục gốc dự án (cùng cấp với config.py)
BASE_DIR = Path(__file__).resolve().parent

# Thư mục chứa database
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn file SQLite chính
DATABASE_PATH = str(DB_DIR / "accounting.db")

# Thư mục upload tạm thời (hóa đơn PDF/XML/ảnh từ người dùng)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 2. CẤU HÌNH OLLAMA / LLM / VLM
# ------------------------------------------------------------------

# Địa chỉ Ollama server chạy local
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Mô hình ngôn ngữ lớn — dùng cho ReAct Agent (System 2) và Text-to-SQL (Phase 4)
# qwen2.5:3b vừa vặn hoàn hảo trong 4GB VRAM của GPU Laptop, phản hồi nhanh (1-2s) và không tràn RAM
LLM_MODEL: str = "qwen2.5:3b"

# Mô hình đa phương thức — dùng cho OCR hóa đơn ảnh (System 2 Tool 4)
VLM_MODEL: str = "qwen2.5-vl:7b"

# Timeout tối đa một lần gọi Ollama (giây)
OLLAMA_TIMEOUT_SECONDS: int = 120

# Số vòng lặp tối đa cho ReAct Agent (tránh vòng lặp vô hạn)
REACT_AGENT_MAX_ITERATIONS: int = 10

# ------------------------------------------------------------------
# 3. HẰNG SỐ NGHIỆP VỤ KẾ TOÁN
# ------------------------------------------------------------------

# Dung sai tiền lẻ (VNĐ) cho phép khi khớp tự động
# Áp dụng cho sai lệch do làm tròn thuế GTGT
TOLERANCE_VND: int = 1_000

# Các mức phí chuyển khoản ngân hàng phổ biến (VNĐ)
# Phí liên ngân hàng qua Napas: 11.000, 22.000, 33.000; phí trong hệ thống: 0, 5.500
COMMON_BANK_FEES: list[int] = [11_000, 22_000, 33_000, 55_000, 5_500, 0]

# Ngưỡng điểm so khớp mờ tên công ty bằng RapidFuzz (0–100)
# Giá trị 82 cân bằng giữa tránh nhầm lẫn & chấp nhận viết tắt không dấu
FUZZY_MATCH_THRESHOLD: int = 82

# Ngưỡng confidence OCR — dưới ngưỡng này cần đánh dấu kiểm tra lại
OCR_CONFIDENCE_THRESHOLD: float = 0.75

# Số ngày tối đa sao kê NH có thể lệch so với ngày hóa đơn (khi match)
DATE_TOLERANCE_DAYS: int = 90

# ------------------------------------------------------------------
# 4. HỆ THỐNG TÀI KHOẢN KẾ TOÁN CHUẨN (TT 200 / TT 133)
# ------------------------------------------------------------------
# Format: { "TK_CODE": {"name": "...", "type": "asset|liability|revenue|expense"} }

CHART_OF_ACCOUNTS: dict[str, dict] = {
    # ---- Tiền & Tương đương tiền ----
    "111":  {"name": "Tiền mặt",                              "type": "asset"},
    "1121": {"name": "Tiền gửi ngân hàng VNĐ",               "type": "asset"},
    "1122": {"name": "Tiền gửi ngân hàng ngoại tệ",          "type": "asset"},

    # ---- Phải thu ----
    "131":  {"name": "Phải thu khách hàng",                   "type": "asset"},

    # ---- Phải trả ----
    "331":  {"name": "Phải trả người bán",                    "type": "liability"},

    # ---- Doanh thu & Giảm trừ doanh thu ----
    "511":  {"name": "Doanh thu bán hàng và cung cấp dịch vụ","type": "revenue"},
    "521":  {"name": "Chiết khấu thương mại",                 "type": "revenue_deduction"},

    # ---- Chi phí ----
    "641":  {"name": "Chi phí bán hàng",                      "type": "expense"},
    "642":  {"name": "Chi phí quản lý doanh nghiệp / Phí ngân hàng",
                                                               "type": "expense"},

    # ---- Thuế GTGT ----
    "1331": {"name": "Thuế GTGT được khấu trừ",              "type": "asset"},
    "3331": {"name": "Thuế GTGT phải nộp",                   "type": "liability"},
    "33311":{"name": "Thuế GTGT đầu ra",                     "type": "liability"},
    "33312":{"name": "Thuế GTGT hàng nhập khẩu",             "type": "liability"},
}

# Mapping nhanh tên tài khoản → mã (dùng khi sinh bút toán tự động)
ACCOUNT_NAME_TO_CODE: dict[str, str] = {
    v["name"]: k for k, v in CHART_OF_ACCOUNTS.items()
}

# ------------------------------------------------------------------
# 5. TRẠNG THÁI HÓA ĐƠN & SAO KÊ
# ------------------------------------------------------------------

class InvoiceStatus:
    UNPAID  = "UNPAID"
    PAID    = "PAID"
    PARTIAL = "PARTIAL"

class ReconcileStatus:
    PENDING              = "PENDING"
    AUTO_MATCHED_SYS1    = "AUTO_MATCHED_SYS1"
    AGENT_RESOLVED_SYS2  = "AGENT_RESOLVED_SYS2"
    NEED_HUMAN_REVIEW    = "NEED_HUMAN_REVIEW"

class MatchType:
    EXACT_1_TO_1           = "EXACT_1_TO_1"
    MULTI_INVOICE_1_TO_N   = "MULTI_INVOICE_1_TO_N"
    FUZZY_NAME_1_TO_1      = "FUZZY_NAME_1_TO_1"
    SUBSET_SUM_COMBO       = "SUBSET_SUM_COMBO"
    DISCOUNT_OR_FEE_ADJUSTED = "DISCOUNT_OR_FEE_ADJUSTED"
    ALIAS_RESOLVED         = "ALIAS_RESOLVED"
    OCR_ERROR_CORRECTED    = "OCR_ERROR_CORRECTED"
    UNRESOLVED             = "UNRESOLVED"

class ResolutionSource:
    SYSTEM_1_RULE          = "SYSTEM_1_RULE"
    SYSTEM_2_REACT_AGENT   = "SYSTEM_2_REACT_AGENT"
    MANUAL                 = "MANUAL"

class SourceType:
    XML        = "XML"
    PDF        = "PDF"
    IMAGE_OCR  = "IMAGE_OCR"

# ------------------------------------------------------------------
# 6. THÔNG TIN DOANH NGHIỆP (Dùng in trên báo cáo)
# ------------------------------------------------------------------

COMPANY_NAME    = "Công ty TNHH Phần Mềm Kế Toán Demo"
COMPANY_TAX_CODE = "0123456789"
FISCAL_YEAR     = 2024

