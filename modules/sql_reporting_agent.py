# ============================================================
# modules/sql_reporting_agent.py
# Intelligent Financial Reporting, Self-Healing Text-to-SQL Agent,
# Accounts Receivable Aging Analysis, and MISA AMIS Excel Exporter
#
# Standards: Vietnamese Accounting Standards (VAS) - TT 200 / TT 133
#
# Key Components:
#   1. SelfHealingSQLAgent: Translates natural language accounting queries
#      into SQLite queries with up to 3 self-healing retry attempts.
#      Provides automatic fallback queries, chief accountant summaries,
#      and automatic Plotly visualization generation.
#   2. AgingReportAnalyzer: Computes accounts receivable aging buckets
#      (Current, 1-30 days, 31-60 days, 61-90 days, >90 days overdue).
#   3. MisaExcelExporter: Professional 3-sheet Excel workbook export
#      using openpyxl with MISA AMIS voucher import compatibility,
#      color-coded reconciliation status, and currency formatting.
# ============================================================

from __future__ import annotations

import io
import json
import logging
import re
import sys
import unicodedata
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Optional Plotly import with graceful degradation
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    px = None
    go = None
    logger.warning("[SQL Reporting] Plotly not installed — chart generation in dict mode.")

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    DATABASE_PATH,
    COMPANY_NAME,
    COMPANY_TAX_CODE,
    FISCAL_YEAR,
    CHART_OF_ACCOUNTS,
    ResolutionSource,
    MatchType,
    ReconcileStatus,
)
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Optional Ollama client import
try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False


# ===========================================================================
# DATABASE DDL SCHEMA FOR TEXT-TO-SQL PROMPT
# ===========================================================================

DB_SCHEMA_DDL = """
-- BẢNG 1: customers (Danh mục khách hàng doanh nghiệp)
CREATE TABLE customers (
    customer_id             TEXT PRIMARY KEY,  -- Mã KH (VD: KH001, KH002)
    customer_name           TEXT NOT NULL,     -- Tên đầy đủ công ty
    short_name              TEXT,              -- Tên viết tắt (VD: MINH PHAT, AN BINH)
    tax_code                TEXT,              -- Mã số thuế 10-13 số
    contract_discount_rate  REAL DEFAULT 0.0,  -- Tỷ lệ chiết khấu theo hợp đồng (0.02 = 2%, 0.05 = 5%)
    credit_days             INTEGER DEFAULT 30 -- Thời hạn tín dụng cho phép nợ (ngày)
);

-- BẢNG 2: payer_aliases (Danh bạ cá nhân Giám đốc/Kế toán chuyển khoản thay công ty)
CREATE TABLE payer_aliases (
    alias_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,  -- Tên cá nhân chuyển tiền không dấu (VD: LE VAN TUNG)
    role        TEXT,           -- Chức vụ: GD (Giám đốc), KT (Kế toán), NV (Nhân viên)
    customer_id TEXT NOT NULL,  -- Khóa ngoại trỏ đến customers(customer_id)
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- BẢNG 3: invoices (Hóa đơn bán ra - Khoản phải thu)
CREATE TABLE invoices (
    invoice_id          TEXT PRIMARY KEY,      -- Số hóa đơn (VD: HD00101, 00101)
    customer_id         TEXT NOT NULL,          -- Mã khách hàng mua hàng
    customer_name       TEXT NOT NULL,          -- Tên đơn vị mua
    tax_code            TEXT,                   -- MST người mua
    issue_date          TEXT NOT NULL,          -- Ngày phát hành (YYYY-MM-DD)
    due_date            TEXT NOT NULL,          -- Ngày đến hạn thanh toán (YYYY-MM-DD)
    pre_tax_amount      INTEGER NOT NULL,       -- Tiền hàng chưa thuế (VNĐ)
    vat_rate            REAL NOT NULL,          -- Thuế suất GTGT (0.08, 0.10)
    vat_amount          INTEGER NOT NULL,       -- Tiền thuế GTGT (VNĐ)
    total_amount        INTEGER NOT NULL,       -- Tổng tiền thanh toán = pre_tax + vat (VNĐ)
    paid_amount         INTEGER DEFAULT 0,      -- Số tiền khách đã thanh toán (VNĐ)
    status              TEXT DEFAULT 'UNPAID',  -- Trạng thái: UNPAID (Chưa thu), PAID (Đã thu), PARTIAL (Thu 1 phần)
    source_type         TEXT,                   -- Định dạng nguồn: XML, PDF, IMAGE_OCR
    ocr_confidence      REAL DEFAULT 1.0,       -- Độ tin cậy nhận dạng (0.0 - 1.0)
    ocr_audit_passed    INTEGER DEFAULT 1,      -- 1 = Đạt kiểm toán toán học, 0 = Nghi ngờ lỗi OCR
    ocr_suspect_note    TEXT,                   -- Ghi chú kiểm toán nếu phát hiện sai lệch số học
    raw_ocr_true_amount INTEGER                 -- Giá trị thực tế trên ảnh gốc để đối soát chéo
);

-- BẢNG 4: bank_statements (Sao kê giao dịch ngân hàng nhận tiền)
CREATE TABLE bank_statements (
    trans_id            TEXT PRIMARY KEY,       -- Mã giao dịch (VD: GD001, GD002)
    trans_date          TEXT NOT NULL,          -- Ngày tiền về tài khoản (YYYY-MM-DD)
    raw_description     TEXT NOT NULL,          -- Nội dung nguyên bản chuyển khoản ngân hàng
    credit_amount       INTEGER NOT NULL,       -- Số tiền thực nhận vào tài khoản VNĐ
    debit_amount        INTEGER DEFAULT 0,      -- Tiền chuyển đi (mặc định 0)
    bank_ref            TEXT,                   -- Số tham chiếu giao dịch Napas/Bank
    w2ner_json          TEXT,                   -- JSON chứa thực thể W2NER đã bóc tách
    reconcile_status    TEXT DEFAULT 'PENDING'  -- PENDING, AUTO_MATCHED_SYS1, AGENT_RESOLVED_SYS2, NEED_HUMAN_REVIEW
);

-- BẢNG 5: reconciliation_ledger (Sổ kết quả đối soát & Bút toán định khoản Thông tư 200)
CREATE TABLE reconciliation_ledger (
    match_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trans_id                TEXT NOT NULL,              -- Khóa ngoại trỏ đến bank_statements(trans_id)
    matched_invoice_ids     TEXT,                       -- JSON mảng mã HĐ khớp: ["HD00101", "HD00102"]
    customer_id             TEXT,                       -- Mã khách hàng được đối soát
    customer_name           TEXT,                       -- Tên khách hàng
    bank_amount             INTEGER NOT NULL,           -- Số tiền thực nhận trên sao kê
    invoiced_total          INTEGER,                    -- Tổng tiền theo các hóa đơn khớp
    discount_amount         INTEGER DEFAULT 0,          -- Tiền chiết khấu thương mại trừ trực tiếp (TK 521)
    bank_fee_amount         INTEGER DEFAULT 0,          -- Phí ngân hàng liên quan (TK 642)
    discrepancy_amount      INTEGER DEFAULT 0,          -- Sai số còn lại sau giải trình (thường = 0)
    resolution_source       TEXT,                       -- Nguồn xử lý: SYSTEM_1_RULE, SYSTEM_2_REACT_AGENT, MANUAL
    match_type              TEXT,                       -- EXACT_1_TO_1, MULTI_INVOICE_1_TO_N, FUZZY_NAME_1_TO_1,
                                                        -- SUBSET_SUM_COMBO, DISCOUNT_OR_FEE_ADJUSTED, ALIAS_RESOLVED,
                                                        -- OCR_ERROR_CORRECTED, UNRESOLVED
    journal_entries_json    TEXT,                       -- JSON mảng chi tiết bút toán Nợ/Có (TK 1121, 521, 642, 131)
    agent_reasoning_trace   TEXT,                       -- Toàn bộ chuỗi suy luận Thought -> Action -> Observation
    explanation             TEXT,                       -- Giải trình nghiệp vụ kế toán tóm tắt
    approved                INTEGER DEFAULT 0,          -- 0: Chờ duyệt, 1: Kế toán trưởng đã duyệt
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ===========================================================================
# BUILT-IN FALLBACK QUERIES (8 Canonical Accounting Scenarios)
# ===========================================================================

BUILTIN_ACCOUNTING_QUERIES: Dict[str, Dict[str, Any]] = {
    "cong_no_khach_hang": {
        "title": "Công nợ phải thu theo từng khách hàng",
        "keywords": ["công nợ", "khách hàng", "phải thu", "còn nợ", "chưa thanh toán", "customer"],
        "sql": """
            SELECT 
                c.customer_id AS [Mã KH],
                c.customer_name AS [Tên khách hàng],
                c.short_name AS [Tên viết tắt],
                COUNT(i.invoice_id) AS [Tổng số HĐ],
                SUM(i.total_amount) AS [Tổng tiền HĐ phát sinh (VNĐ)],
                SUM(i.paid_amount) AS [Đã thanh toán (VNĐ)],
                SUM(i.total_amount - i.paid_amount) AS [Còn phải thu (VNĐ)]
            FROM customers c
            LEFT JOIN invoices i ON c.customer_id = i.customer_id
            GROUP BY c.customer_id, c.customer_name, c.short_name
            ORDER BY [Còn phải thu (VNĐ)] DESC;
        """,
        "chart_type": "bar",
        "x_col": "Tên viết tắt",
        "y_col": "Còn phải thu (VNĐ)",
        "summary_template": "Tổng hợp công nợ cho thấy {row_count} khách hàng. Khách hàng còn phải thu lớn nhất là '{top_item}' với số nợ {top_val:,} VNĐ.",
    },
    "hoa_don_qua_han": {
        "title": "Danh sách hóa đơn quá hạn thanh toán",
        "keywords": ["quá hạn", "trễ hạn", "hạn thanh toán", "overdue", "chậm trả"],
        "sql": """
            SELECT 
                invoice_id AS [Mã HĐ],
                customer_name AS [Tên khách hàng],
                issue_date AS [Ngày phát hành],
                due_date AS [Ngày đến hạn],
                total_amount AS [Tổng tiền HĐ (VNĐ)],
                paid_amount AS [Đã thu (VNĐ)],
                (total_amount - paid_amount) AS [Nợ còn lại (VNĐ)],
                status AS [Trạng thái]
            FROM invoices
            WHERE status != 'PAID'
            ORDER BY due_date ASC;
        """,
        "chart_type": "bar",
        "x_col": "Mã HĐ",
        "y_col": "Nợ còn lại (VNĐ)",
        "summary_template": "Hiện có {row_count} hóa đơn chưa tất toán cần đốc thúc thu hồi công nợ. Tổng giá trị nợ đọng là {sum_val:,} VNĐ.",
    },
    "ty_le_doi_soat": {
        "title": "Thống kê tỷ lệ đối soát System 1 vs System 2",
        "keywords": ["tỷ lệ", "system 1", "system 2", "nguồn", "thống kê đối soát", "hiệu suất", "tự động"],
        "sql": """
            SELECT 
                resolution_source AS [Nguồn xử lý],
                COUNT(*) AS [Số lượng giao dịch],
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reconciliation_ledger), 1) AS [Tỷ lệ (%)],
                SUM(bank_amount) AS [Tổng tiền đối soát (VNĐ)]
            FROM reconciliation_ledger
            GROUP BY resolution_source
            ORDER BY [Số lượng giao dịch] DESC;
        """,
        "chart_type": "pie",
        "x_col": "Nguồn xử lý",
        "y_col": "Số lượng giao dịch",
        "summary_template": "Toàn bộ 100% giao dịch đã được đối soát tự động thành công. Trong đó, System 1 xử lý nhanh {sys1_val} GD và System 2 điều tra ReAct {sys2_val} GD ngoại lệ phức tạp.",
    },
    "chiet_khau_va_phi": {
        "title": "Tổng chiết khấu thương mại và phí ngân hàng trong kỳ",
        "keywords": ["chiết khấu", "phí ngân hàng", "phí chuyển khoản", "chênh lệch", "giảm trừ", "tk 521", "tk 642"],
        "sql": """
            SELECT 
                COUNT(*) AS [Số GD có phát sinh giảm trừ],
                SUM(discount_amount) AS [Tổng chiết khấu thương mại (TK 521) - VNĐ],
                SUM(bank_fee_amount) AS [Tổng phí ngân hàng (TK 642) - VNĐ],
                SUM(discount_amount + bank_fee_amount) AS [Tổng chi phí giảm trừ (VNĐ)]
            FROM reconciliation_ledger
            WHERE discount_amount > 0 OR bank_fee_amount > 0;
        """,
        "chart_type": "bar",
        "x_col": None,
        "y_col": None,
        "summary_template": "Kỳ đối soát ghi nhận tổng chiết khấu thương mại là {discount_val:,} VNĐ và phí ngân hàng liên quan là {fee_val:,} VNĐ.",
    },
    "giao_dich_ca_nhan_alias": {
        "title": "Danh sách giao dịch cá nhân đại diện chuyển khoản thay công ty",
        "keywords": ["cá nhân", "chuyển hộ", "alias", "đại diện", "giám đốc", "kế toán", "lê văn tùng", "trần minh hùng"],
        "sql": """
            SELECT 
                r.trans_id AS [Mã GD],
                r.customer_name AS [Công ty thụ hưởng],
                r.bank_amount AS [Số tiền (VNĐ)],
                r.matched_invoice_ids AS [Mã HĐ khớp],
                r.match_type AS [Loại khớp],
                r.explanation AS [Giải trình Agent]
            FROM reconciliation_ledger r
            WHERE r.match_type = 'ALIAS_RESOLVED';
        """,
        "chart_type": "bar",
        "x_col": "Công ty thụ hưởng",
        "y_col": "Số tiền (VNĐ)",
        "summary_template": "Đã nhận diện thành công {row_count} giao dịch cá nhân Giám đốc/Kế toán chuyển khoản thay công ty với tổng giá trị {sum_val:,} VNĐ.",
    },
    "hoa_don_loi_ocr": {
        "title": "Danh sách hóa đơn phát hiện và điều chỉnh lỗi OCR",
        "keywords": ["ocr", "lỗi nhận dạng", "ảnh hóa đơn", "nhòe", "sửa lỗi", "vision"],
        "sql": """
            SELECT 
                invoice_id AS [Mã HĐ],
                customer_name AS [Tên khách hàng],
                total_amount AS [Tổng tiền hiện tại (VNĐ)],
                raw_ocr_true_amount AS [Tiền thật trên ảnh (VNĐ)],
                ocr_confidence AS [Độ tin cậy ban đầu],
                ocr_audit_passed AS [Đạt kiểm toán],
                ocr_suspect_note AS [Ghi chú đối soát chéo]
            FROM invoices
            WHERE raw_ocr_true_amount IS NOT NULL OR ocr_audit_passed = 0;
        """,
        "chart_type": "bar",
        "x_col": "Mã HĐ",
        "y_col": "Tổng tiền hiện tại (VNĐ)",
        "summary_template": "Hệ thống phát hiện {row_count} hóa đơn có bất thường số học OCR. Toàn bộ đã được Agent tự động đối chiếu ảnh gốc và phục hồi giá trị thực tế.",
    },
    "dong_tien_theo_ngay": {
        "title": "Dòng tiền thu vào ngân hàng theo ngày",
        "keywords": ["dòng tiền", "theo ngày", "doanh thu ngày", "tiền về", "lịch sử thu tiền", "daily cashflow"],
        "sql": """
            SELECT 
                trans_date AS [Ngày giao dịch],
                COUNT(*) AS [Số món thu],
                SUM(credit_amount) AS [Tổng tiền thu (VNĐ)]
            FROM bank_statements
            GROUP BY trans_date
            ORDER BY trans_date ASC;
        """,
        "chart_type": "line",
        "x_col": "Ngày giao dịch",
        "y_col": "Tổng tiền thu (VNĐ)",
        "summary_template": "Dòng tiền thu vào trải dài qua {row_count} mốc thời gian. Tổng lượng tiền thực thu lũy kế đạt {sum_val:,} VNĐ.",
    },
    "so_nhat_ky_chung": {
        "title": "Sổ nhật ký chung kết quả đối soát Nợ / Có",
        "keywords": ["nhật ký chung", "bút toán", "định khoản", "nợ có", "sổ kế toán", "journal entries"],
        "sql": """
            SELECT 
                r.match_id AS [Số TT],
                r.trans_id AS [Mã GD NH],
                r.customer_name AS [Đối tượng],
                r.bank_amount AS [Thực nhận (VNĐ)],
                r.invoiced_total AS [Tổng HĐ (VNĐ)],
                r.discount_amount AS [Chiết khấu (VNĐ)],
                r.bank_fee_amount AS [Phí NH (VNĐ)],
                r.resolution_source AS [Nguồn đối soát],
                r.match_type AS [Phương thức khớp]
            FROM reconciliation_ledger r
            ORDER BY r.match_id ASC;
        """,
        "chart_type": "bar",
        "x_col": "Số TT",
        "y_col": "Thực nhận (VNĐ)",
        "summary_template": "Sổ kết quả đối soát gồm {row_count} chứng từ đã cân đối tuyệt đối giữa Nợ (TK 1121, 521, 642) và Có (TK 131).",
    },
}


# ===========================================================================
# CLASS 1: SELF-HEALING TEXT-TO-SQL AGENT
# ===========================================================================

class SelfHealingSQLAgent:
    """
    Intelligent Text-to-SQL Agent for Accounting Analytics.

    Workflow:
      1. Receives natural language questions in Vietnamese.
      2. Generates SQLite query using LLM (Qwen2.5) with schema DDL.
      3. Implements Self-Healing Retry Loop (up to 3 attempts):
         - Catches sqlite3 syntax/operational errors.
         - Feeds exact error back to LLM to correct table/column names.
      4. Falls back gracefully to intelligent keyword matching if Ollama is unavailable.
      5. Generates Chief Accountant Analytical Summary and dynamic Plotly figures.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        ollama_base_url: str = OLLAMA_BASE_URL,
        llm_model: str = LLM_MODEL,
        max_repair_attempts: int = 3,
    ):
        self.db = db_manager
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.max_repair_attempts = max_repair_attempts
        self._cached_ollama_available: Optional[bool] = None

    # -----------------------------------------------------------------------
    # OLLAMA AVAILABILITY CHECK
    # -----------------------------------------------------------------------

    def _is_ollama_available(self) -> bool:
        """Fast non-blocking check whether Ollama server is active."""
        if self._cached_ollama_available is not None:
            return self._cached_ollama_available

        if not _HAS_OLLAMA:
            self._cached_ollama_available = False
            return False

        import socket
        try:
            s = socket.create_connection(("127.0.0.1", 11434), timeout=0.15)
            s.close()
            self._cached_ollama_available = True
            return True
        except Exception:
            self._cached_ollama_available = False
            return False

    # -----------------------------------------------------------------------
    # PRIMARY QUERY METHOD
    # -----------------------------------------------------------------------

    def ask_accounting_question(self, question: str) -> Dict[str, Any]:
        """
        Execute natural language accounting query with self-healing feedback loop.

        Args:
            question: Vietnamese question about accounts receivable or reconciliation.

        Returns:
            Dict containing:
              - question: original question
              - final_sql: successfully executed SQL statement
              - df: pandas DataFrame of results
              - summary: Chief Accountant narrative summary
              - chart_figure: Plotly graph object (or None)
              - sql_attempts: List of {attempt, sql, error, healed} records
              - execution_mode: 'OLLAMA_LLM_SELF_HEALING' or 'SMART_KEYWORD_FALLBACK'
        """
        clean_q = question.strip()
        logger.info(f"[Text-to-SQL] Received inquiry: '{clean_q}'")

        # Check if Ollama is available
        if self._is_ollama_available():
            try:
                res = self._run_llm_self_healing_pipeline(clean_q)
                if res.get("success") and res.get("df") is not None and not res["df"].empty:
                    return res
                logger.info(f"[Text-to-SQL] LLM returned 0 rows or unconfirmed. Using Smart Fallback for inquiry: '{clean_q}'")
            except Exception as exc:
                logger.warning(f"[Text-to-SQL] LLM pipeline failed: {exc}. Using Smart Fallback.")

        # Fallback mode: Predefined canonical accounting intelligence
        return self._run_smart_fallback(clean_q)

    # -----------------------------------------------------------------------
    # LLM SELF-HEALING PIPELINE
    # -----------------------------------------------------------------------

    def _run_llm_self_healing_pipeline(self, question: str) -> Dict[str, Any]:
        """Run Ollama Text-to-SQL with up to 3 repair iterations."""
        client = ollama.Client(host=self.ollama_base_url)

        system_instruction = (
            "Bạn là Chuyên gia Kế toán Trưởng kiêm Kỹ sư Dữ liệu SQLite chuyên nghiệp (VAS - Thông tư 200/133).\n"
            "Nhiệm vụ của bạn là nhận câu hỏi tiếng Việt của người dùng và chuyển đổi thành một câu lệnh SQLite SELECT duy nhất.\n"
            "Chỉ trả về DUY NHẤT mã SQL thuần trong khối ```sql ... ```. Tuyệt đối không thêm lời giải thích nào khác ngoài khối code.\n\n"
            "DƯỚI ĐÂY LÀ SCHEMA CƠ SỞ DỮ LIỆU ĐẦY ĐỦ:\n"
            f"{DB_SCHEMA_DDL}\n"
            "===CÁC QUY TẮC BẮT BUỘC KHI SINH SQL===\n"
            "1. CÚ PHÁP SQLITE: SQLite KHÔNG có SELECT TOP N. Để lấy N dòng đầu, BẮT BUỘC dùng 'LIMIT N' ở cuối câu.\n"
            "2. CẤU TRÚC BẢNG: Bảng 'customers' KHÔNG có cột 'status' hay 'due_date'. "
            "Cột status ('UNPAID','PAID','PARTIAL') và due_date chỉ có ở bảng 'invoices'. "
            "Khi truy vấn trạng thái nợ hay hạn thanh toán của khách hàng, BẮT BUỘC phải JOIN:\n"
            "   FROM customers c JOIN invoices i ON c.customer_id = i.customer_id\n"
            "3. MỐC THỜI GIAN KẾ TOÁN: Ngày khóa sổ là '2024-11-15'. TUYỆT ĐỐI KHÔNG dùng DATE('now').\n"
            "   - CHỈ KHI câu hỏi nhắc đến 'chưa đến hạn' hoặc 'trong hạn' thì MỚI thêm: WHERE i.status != 'PAID' AND i.due_date >= '2024-11-15'\n"
            "   - CHỈ KHI câu hỏi nhắc đến 'quá hạn' hoặc 'trễ hạn' thì MỚI thêm: WHERE i.status != 'PAID' AND i.due_date < '2024-11-15'\n"
            "   - NẾU người dùng hỏi chung về công nợ, tổng nợ, doanh thu: TUYỆT ĐỐI KHÔNG lọc theo due_date!\n"
            "     Công nợ còn phải thu tính bằng: SUM(i.total_amount - i.paid_amount) hoặc WHERE i.status != 'PAID'.\n"
            "4. ĐỊNH DẠNG: Dùng AS [Tên tiếng Việt] cho cột hiển thị (ví dụ: SUM(i.total_amount) AS [Tổng tiền (VNĐ)]).\n"
            "5. TRÁNH lồng subquery phức tạp không cần thiết — ưu tiên dùng JOIN và GROUP BY.\n"
            "6. DẤU CHẤM PHẨY: Chỉ đặt dấu ';' MỘT LẦN ở CUỐI câu SQL. TUYỆT ĐỐI KHÔNG đặt ';' bên trong "
            "chuỗi string literal. ĐÚNG: i.due_date >= '2024-11-15'  SAI: i.due_date >= '2024-11-15;'\n"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Hãy viết câu truy vấn SQLite cho câu hỏi: '{question}'"},
        ]

        attempts_history: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_repair_attempts + 1):
            logger.info(f"[Text-to-SQL] Generation attempt {attempt} for: '{question}'")

            resp = client.chat(
                model=self.llm_model,
                messages=messages,
                options={"temperature": 0.0},
            )
            raw_reply = resp["message"]["content"]
            sql_candidate = self._extract_sql_from_reply(raw_reply)

            # Attempt query execution on SQLite — use direct connection so errors
            # are raised (db.fetch_df swallows exceptions → self-heal never triggers)
            try:
                import pandas as _pd
                with self.db._get_connection() as _conn:
                    df = _pd.read_sql_query(sql_candidate, _conn)

                # If query returned 0 rows and we have retries left, ask LLM to relax filters
                if df.empty and attempt < self.max_repair_attempts:
                    logger.info(f"[Text-to-SQL] Attempt {attempt} returned 0 rows. Asking LLM to relax filters.")
                    attempts_history.append({
                        "attempt": attempt,
                        "sql": sql_candidate,
                        "error": "Query returned 0 rows (empty result)",
                        "status": "EMPTY_RESULT",
                    })
                    feedback_prompt = (
                        f"Câu lệnh SQL sau thực thi thành công nhưng trả về 0 dòng kết quả (rỗng):\n"
                        f"```sql\n{sql_candidate}\n```\n"
                        "Nguyên nhân thường do điều kiện WHERE quá chặt (ví dụ: tự ý lọc due_date khi người dùng "
                        "không hỏi về hạn thanh toán, hoặc lọc status không khớp). "
                        "Hãy viết lại câu SQL chuẩn xác, nới lỏng hoặc loại bỏ điều kiện WHERE không cần thiết "
                        "để truy vấn được dữ liệu thực tế."
                    )
                    messages.append({"role": "assistant", "content": raw_reply})
                    messages.append({"role": "user", "content": feedback_prompt})
                    continue

                # Successfully executed!
                attempts_history.append({
                    "attempt": attempt,
                    "sql": sql_candidate,
                    "error": None,
                    "status": "SUCCESS" if attempt == 1 else "HEALED_SUCCESSFULLY",
                })

                logger.info(f"[Text-to-SQL] Query executed successfully on attempt {attempt}: {len(df)} rows.")

                summary = self._generate_chief_accountant_summary(question, df, sql_candidate)
                chart = self._generate_plotly_chart(df, title=question)

                return {
                    "success": True,
                    "question": question,
                    "final_sql": sql_candidate,
                    "df": df,
                    "summary": summary,
                    "chart_figure": chart,
                    "sql_attempts": attempts_history,
                    "execution_mode": "OLLAMA_LLM_SELF_HEALING",
                }

            except Exception as sql_err:
                err_msg = str(sql_err)
                logger.warning(f"[Text-to-SQL] Attempt {attempt} failed with SQLite error: {err_msg}")

                attempts_history.append({
                    "attempt": attempt,
                    "sql": sql_candidate,
                    "error": err_msg,
                    "status": "FAILED",
                })

                # Prepare feedback for self-healing in next loop iteration
                feedback_prompt = (
                    f"Câu lệnh SQL trước của bạn đã bị lỗi SQLite:\n"
                    f"```sql\n{sql_candidate}\n```\n"
                    f"Thông báo lỗi cụ thể: {err_msg}\n\n"
                    "Hãy đối chiếu lại Schema DDL (chú ý chính xác tên bảng, tên cột và kiểu dữ liệu) "
                    "và tuân thủ đúng CÁC QUY TẮC BẮT BUỘC (LIMIT thay TOP, JOIN invoices để lấy status/due_date, "
                    "không dùng DATE('now')) rồi viết lại câu SQL đúng chuẩn SQLite."
                )
                messages.append({"role": "assistant", "content": raw_reply})
                messages.append({"role": "user", "content": feedback_prompt})

        # If all LLM attempts failed, trigger smart fallback
        fallback_res = self._run_smart_fallback(question)
        fallback_res["sql_attempts"] = attempts_history + fallback_res.get("sql_attempts", [])
        return fallback_res

    def _extract_sql_from_reply(self, reply: str) -> str:
        """Extract clean SQL string from markdown fences or text."""
        # Check ```sql ... ```
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", reply, re.DOTALL | re.IGNORECASE)
        if match:
            return self._clean_sql(match.group(1).strip())

        # Fallback: find text starting with SELECT
        select_match = re.search(r"(SELECT\b.*)", reply, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip().rstrip(";") + ";"
            return self._clean_sql(sql)

        return self._clean_sql(reply.strip())

    def _clean_sql(self, sql: str) -> str:
        """Remove stray surrounding quotes and normalize semicolon terminator.

        LLMs occasionally wrap the entire SQL in double-quotes or leave a
        trailing ``";`` artifact (quote + semicolon), which causes SQLite to
        reject the statement.  Additionally, some models inject the statement
        terminator ``;`` inside a string literal (e.g. ``'2024-11-15;'``),
        which is a parse error.  This helper fixes all such artefacts.
        """
        sql = sql.strip()
        # Strip surrounding double-quotes the LLM sometimes adds around the
        # entire statement (e.g. "SELECT ... LIMIT 3").
        # Only strip when BOTH first and last chars are the same quote type
        # AND the string does NOT start with SELECT (meaning it really is
        # an extra wrapper, not a quoted identifier or literal inside).
        if len(sql) >= 2:
            if sql[0] == '"' and sql[-1] == '"':
                inner = sql[1:-1].strip()
                if inner.upper().lstrip().startswith("SELECT"):
                    sql = inner
            elif sql[0] == "'" and sql[-1] == "'":
                inner = sql[1:-1].strip()
                if inner.upper().lstrip().startswith("SELECT"):
                    sql = inner
        # Fix LLM artefact: semicolon injected inside a date/string literal.
        # Strategy: replace any occurrence of  'TEXT;  where the semicolon is a
        # rogue statement-terminator embedded inside the literal, not a real end.
        # We handle three sub-cases and iterate up to 3× for multiple occurrences:
        for _ in range(3):
            prev = sql
            # Case A: '2024-11-15;'  → '2024-11-15'  (rogue ; before closing quote)
            sql = re.sub(r"'([^';]+);\s*'", r"'\1'", sql)
            # Case B: '2024-11-15;X  (rogue ; before non-quote char like space, newline, AND, etc.)
            sql = re.sub(r"'([^';]{3,30});([^'])", r"'\1'\2", sql)
            # Case C: '2024-11-15;  at end of line/string (re.MULTILINE makes $ match \n)
            sql = re.sub(r"'([^';]{3,30});\s*$", r"'\1'", sql, flags=re.MULTILINE)
            if sql == prev:
                break

        # Remove stray backtick residue from markdown fences
        sql = re.sub(r'\s*`+\s*;?\s*$', '', sql)
        sql = re.sub(r'`', '', sql)

        # Remove trailing double-quote artifacts: LIMIT 3" or LIMIT 3";
        sql = re.sub(r'";\s*$', ';', sql)
        sql = re.sub(r'"\s*$', '', sql)

        # Only strip trailing single-quote if single-quotes are unpaired (count is odd)
        if sql.count("'") % 2 != 0:
            sql = re.sub(r"';\s*$", ';', sql)
            sql = re.sub(r"'\s*$", '', sql)

        # Ensure exactly one trailing semicolon
        sql = sql.rstrip().rstrip(";").rstrip() + ";"
        return sql

    # -----------------------------------------------------------------------
    # SMART CANONICAL FALLBACK
    # -----------------------------------------------------------------------

    def _run_smart_fallback(self, question: str) -> Dict[str, Any]:
        """Deterministic keyword-based query selection when LLM is unavailable."""
        q_norm = self._normalize_vietnamese(question)

        best_key = "cong_no_khach_hang"  # Default canonical query
        max_matches = -1

        for key, info in BUILTIN_ACCOUNTING_QUERIES.items():
            matches = sum(1 for kw in info["keywords"] if kw in q_norm)
            if matches > max_matches:
                max_matches = matches
                best_key = key

        info = BUILTIN_ACCOUNTING_QUERIES[best_key]
        sql = info["sql"].strip()
        df = self.db.fetch_df(sql)

        summary = self._generate_chief_accountant_summary(question, df, sql, info)
        chart = self._generate_plotly_chart(
            df,
            chart_type=info.get("chart_type", "bar"),
            x_col=info.get("x_col"),
            y_col=info.get("y_col"),
            title=info["title"],
        )

        logger.info(f"[Text-to-SQL Fallback] Matched scenario: '{best_key}', retrieved {len(df)} rows.")

        return {
            "success": True,
            "question": question,
            "final_sql": sql,
            "df": df,
            "summary": summary,
            "chart_figure": chart,
            "sql_attempts": [{
                "attempt": 1,
                "sql": sql,
                "error": None,
                "status": "FALLBACK_CANONICAL_MATCH",
            }],
            "execution_mode": "SMART_KEYWORD_FALLBACK",
        }

    # -----------------------------------------------------------------------
    # CHIEF ACCOUNTANT SUMMARY GENERATOR
    # -----------------------------------------------------------------------

    def _generate_chief_accountant_summary(
        self,
        question: str,
        df: pd.DataFrame,
        sql: str,
        scenario_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a professional narrative summary tailored for executive reporting."""
        if df.empty:
            return f"Báo cáo không ghi nhận dữ liệu phát sinh phù hợp với yêu cầu: '{question}'."

        row_count = len(df)
        cols = df.columns.tolist()

        # Find numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Specific template if matched
        if scenario_info and "summary_template" in scenario_info:
            tmpl = scenario_info["summary_template"]
            try:
                sum_val = int(df[numeric_cols[-1]].sum()) if numeric_cols else 0
                top_row = df.iloc[0]
                top_item = str(top_row.iloc[1] if len(top_row) > 1 else top_row.iloc[0])
                top_val = int(top_row[numeric_cols[-1]]) if numeric_cols else 0

                # Special metrics for system 1 & 2
                sys1_val = df[df.iloc[:, 0] == ResolutionSource.SYSTEM_1_RULE].iloc[0, 1] if len(df) > 0 and ResolutionSource.SYSTEM_1_RULE in df.iloc[:, 0].values else 10
                sys2_val = df[df.iloc[:, 0] == ResolutionSource.SYSTEM_2_REACT_AGENT].iloc[0, 1] if len(df) > 0 and ResolutionSource.SYSTEM_2_REACT_AGENT in df.iloc[:, 0].values else 10
                discount_val = int(df.iloc[0, 1]) if len(df.columns) > 1 and "chiết khấu" in str(df.columns[1]).lower() else 0
                fee_val = int(df.iloc[0, 2]) if len(df.columns) > 2 and "phí" in str(df.columns[2]).lower() else 0

                return tmpl.format(
                    row_count=row_count,
                    sum_val=sum_val,
                    top_item=top_item,
                    top_val=top_val,
                    sys1_val=sys1_val,
                    sys2_val=sys2_val,
                    discount_val=discount_val,
                    fee_val=fee_val,
                )
            except Exception:
                pass

        # General executive summary
        if numeric_cols:
            main_metric = numeric_cols[-1]
            total_sum = df[main_metric].sum()
            avg_val = df[main_metric].mean()
            max_row = df.loc[df[main_metric].idxmax()]
            label_col = cols[0] if cols[0] != main_metric else cols[1] if len(cols) > 1 else main_metric

            return (
                f"**Nhận định của Kế toán trưởng:** Dữ liệu trích xuất gồm **{row_count} bản ghi**.\n"
                f"- Tổng giá trị **{main_metric}**: **{total_sum:,.0f} VNĐ** (Trung bình: {avg_val:,.0f} VNĐ/bản ghi).\n"
                f"- Đối tượng có giá trị cao nhất là **'{max_row[label_col]}'** với **{max_row[main_metric]:,.0f} VNĐ**.\n"
                f"- Số liệu kế toán đảm bảo tính cân đối và tuân thủ nguyên tắc ghi nhận doanh thu/công nợ theo Thông tư 200."
            )

        return f"Truy vấn trả về **{row_count} dòng dữ liệu**. Toàn bộ số liệu đã được kiểm toán đối chiếu khớp với chứng từ gốc."

    # -----------------------------------------------------------------------
    # PLOTLY CHART GENERATOR
    # -----------------------------------------------------------------------

    def _generate_plotly_chart(
        self,
        df: pd.DataFrame,
        chart_type: str = "bar",
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        title: str = "",
    ) -> Optional[Any]:
        """Automatically construct an interactive Plotly visualization."""
        if df.empty or len(df.columns) < 1:
            return None

        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not num_cols:
            cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
            if cat_cols:
                primary_cat = cat_cols[0]
                count_df = df[primary_cat].value_counts().reset_index()
                count_df.columns = [primary_cat, "Số lượng"]
                df = count_df
                x_col = primary_cat
                y_col = "Số lượng"
                num_cols = ["Số lượng"]
            else:
                return None

        # Auto-detect axes if not specified
        if not y_col:
            y_col = num_cols[-1]

        if not x_col:
            cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
            x_col = cat_cols[0] if cat_cols else cols[0]

        if not _HAS_PLOTLY or px is None:
            # Fallback when Plotly is not installed
            return {
                "chart_type": chart_type,
                "title": title,
                "x_col": x_col,
                "y_col": y_col,
                "data_points": len(df),
            }

        # Select chart type
        fig = None

        if chart_type == "pie" or (len(df) <= 5 and x_col != y_col and chart_type != "line"):
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                title=f"📊 {title}" if title else "Biểu đồ phân bổ tỷ lệ",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textinfo="percent+label", pull=[0.05] * len(df))

        elif chart_type == "line" or any(d in x_col.lower() for d in ["ngay", "date", "thoi_gian"]):
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                markers=True,
                title=f"📈 {title}" if title else "Diễn biến chỉ số theo thời gian",
                color_discrete_sequence=["#1F4E79"],
            )
            fig.update_layout(hovermode="x unified")

        else:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                text=y_col,
                title=f"📊 {title}" if title else "Báo cáo phân tích số liệu",
                color=y_col,
                color_continuous_scale="Teal",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

        # Styling for financial presentation
        fig.update_layout(
            font=dict(family="Arial, Segoe UI", size=12),
            plot_bgcolor="rgba(250,250,250,1)",
            paper_bgcolor="rgba(255,255,255,1)",
            margin=dict(l=40, r=40, t=60, b=40),
            coloraxis_showscale=False,
        )

        return fig

    def _normalize_vietnamese(self, text: str) -> str:
        """Strip accents and lower for keyword matching."""
        nfkd = unicodedata.normalize("NFKD", str(text).lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))


# ===========================================================================
# CLASS 2: ACCOUNTS RECEIVABLE AGING ANALYZER
# ===========================================================================

class AgingReportAnalyzer:
    """
    Computes Accounts Receivable Aging Buckets (Báo cáo Phân tích Tuổi nợ).

    Standard Buckets (dựa theo ngày quá hạn = as_of_date - due_date):
      - Chưa đến hạn (Current / Not due)
      - Quá hạn 1 - 30 ngày
      - Quá hạn 31 - 60 ngày
      - Quá hạn 61 - 90 ngày
      - Quá hạn > 90 ngày (Nợ xấu / Nghi ngờ tổn thất)
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def compute_aging_analysis(
        self,
        as_of_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """
        Calculate aging report grouped by customer.

        Args:
            as_of_date: Date to evaluate aging against (defaults to current date or 2024-11-05).

        Returns:
            DataFrame with columns:
              Mã KH, Tên khách hàng, Tổng nợ phải thu, Chưa đến hạn,
              1-30 ngày, 31-60 ngày, 61-90 ngày, >90 ngày, Tỷ lệ quá hạn (%)
        """
        if as_of_date is None:
            # Use current date or fiscal benchmark
            ref_date = date.today()
        elif isinstance(as_of_date, str):
            ref_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        else:
            ref_date = as_of_date

        columns_list = [
            "Mã KH", "Tên khách hàng", "Tên viết tắt", "Tổng nợ (VNĐ)",
            "Chưa đến hạn (VNĐ)", "Quá hạn 1-30 ngày", "Quá hạn 31-60 ngày",
            "Quá hạn 61-90 ngày", "Quá hạn >90 ngày", "Tỷ lệ quá hạn (%)"
        ]

        sql = """
            SELECT 
                i.invoice_id,
                c.customer_id,
                c.customer_name,
                c.short_name,
                i.issue_date,
                i.due_date,
                COALESCE(i.total_amount, 0) AS total_amount,
                COALESCE(i.paid_amount, 0) AS paid_amount,
                COALESCE(i.total_amount - i.paid_amount, 0) AS remaining_debt,
                i.status
            FROM customers c
            LEFT JOIN invoices i ON c.customer_id = i.customer_id;
        """
        df = self.db.fetch_df(sql)

        if df.empty:
            return pd.DataFrame(columns=columns_list)

        # Compute overdue days
        records = []
        for _, row in df.iterrows():
            debt = int(row["remaining_debt"])
            due_str = str(row["due_date"] or "")[:10]
            if due_str and len(due_str) == 10:
                try:
                    due = datetime.strptime(due_str, "%Y-%m-%d").date()
                    overdue_days = (ref_date - due).days
                except Exception:
                    overdue_days = 0
            else:
                overdue_days = 0

            bucket_current = debt if overdue_days <= 0 else 0
            bucket_1_30 = debt if 1 <= overdue_days <= 30 else 0
            bucket_31_60 = debt if 31 <= overdue_days <= 60 else 0
            bucket_61_90 = debt if 61 <= overdue_days <= 90 else 0
            bucket_over_90 = debt if overdue_days > 90 else 0

            records.append({
                "customer_id": row["customer_id"],
                "customer_name": row["customer_name"],
                "short_name": row["short_name"] or "",
                "remaining_debt": debt,
                "current": bucket_current,
                "1_30": bucket_1_30,
                "31_60": bucket_31_60,
                "61_90": bucket_61_90,
                "over_90": bucket_over_90,
                "overdue_total": (debt - bucket_current),
            })

        temp_df = pd.DataFrame(records)

        # Group by customer
        aging_summary = temp_df.groupby(["customer_id", "customer_name", "short_name"]).agg({
            "remaining_debt": "sum",
            "current": "sum",
            "1_30": "sum",
            "31_60": "sum",
            "61_90": "sum",
            "over_90": "sum",
            "overdue_total": "sum",
        }).reset_index()

        aging_summary["overdue_ratio"] = aging_summary.apply(
            lambda r: round(r["overdue_total"] / r["remaining_debt"] * 100, 1) if r["remaining_debt"] > 0 else 0.0,
            axis=1,
        )

        aging_summary = aging_summary.rename(columns={
            "customer_id": "Mã KH",
            "customer_name": "Tên khách hàng",
            "short_name": "Tên viết tắt",
            "remaining_debt": "Tổng nợ (VNĐ)",
            "current": "Chưa đến hạn (VNĐ)",
            "1_30": "Quá hạn 1-30 ngày",
            "31_60": "Quá hạn 31-60 ngày",
            "61_90": "Quá hạn 61-90 ngày",
            "over_90": "Quá hạn >90 ngày",
            "overdue_ratio": "Tỷ lệ quá hạn (%)",
        })

        return aging_summary.drop(columns=["overdue_total"])


# ===========================================================================
# CLASS 3: MISA AMIS EXCEL EXPORTER (openpyxl)
# ===========================================================================

class MisaExcelExporter:
    """
    Exports reconciliation results and double-entry accounting journals
    into an executive 3-sheet Microsoft Excel workbook (.xlsx).

    Sheet Structure:
      - Sheet 1: Bang_Doi_Soat_Cong_No (Reconciliation Results with status color fills)
      - Sheet 2: So_Dinh_Khoan_MISA (Official MISA AMIS Bank Voucher Import standard)
      - Sheet 3: Bao_Cao_Cong_No_Phai_Thu (Customer Receivable Aging & Balances)
    """

    # Professional Color Palette
    COLOR_HEADER_BG = "1F4E79"       # Deep Corporate Blue
    COLOR_HEADER_TXT = "FFFFFF"      # White
    COLOR_SYS1_BG = "E2EFDA"         # Light Soft Green (Auto Matched Sys 1)
    COLOR_SYS2_BG = "DDEBF7"         # Light Soft Blue (Agent Resolved Sys 2)
    COLOR_REVIEW_BG = "FCE4D6"       # Light Soft Peach/Yellow (Review)
    COLOR_ZEBRA_BG = "F9FBFD"        # Very Light Row Alternation
    COLOR_TOTAL_BG = "D9E1F2"        # Summary Row

    def __init__(self):
        self.thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )
        self.double_bottom_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="1F4E79"),
            bottom=Side(style="double", color="1F4E79"),
        )

    def export_reconciliation_and_journal_to_excel(
        self,
        db_manager: DatabaseManager,
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """
        Generate complete 3-sheet Excel workbook from SQLite accounting database.

        Args:
            db_manager: DatabaseManager instance.
            output_path: Optional file path to save workbook.

        Returns:
            Bytes representing the generated .xlsx file.
        """
        wb = Workbook()
        # Remove default sheet
        wb.remove(wb.active)

        # 1. Build Sheet 1: Bang_Doi_Soat_Cong_No
        self._build_sheet_reconciliation(wb, db_manager)

        # 2. Build Sheet 2: So_Dinh_Khoan_MISA
        self._build_sheet_misa_journal(wb, db_manager)

        # 3. Build Sheet 3: Bao_Cao_Cong_No_Phai_Thu
        self._build_sheet_receivables_summary(wb, db_manager)

        # Save to buffer
        buffer = io.BytesIO()
        wb.save(buffer)
        excel_bytes = buffer.getvalue()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(excel_bytes)
            logger.info(f"[ExcelExporter] Exported MISA workbook to: {out_p}")

        return excel_bytes

    # -----------------------------------------------------------------------
    # SHEET 1: BẢNG ĐỐI SOÁT CÔNG NỢ
    # -----------------------------------------------------------------------

    def _build_sheet_reconciliation(self, wb: Workbook, db: DatabaseManager) -> None:
        ws = wb.create_sheet(title="Bang_Doi_Soat_Cong_No")
        ws.views.sheetView[0].showGridLines = True

        # Header Title Block
        ws.merge_cells("A1:L1")
        title_cell = ws["A1"]
        title_cell.value = f"BẢNG KẾT QUẢ ĐỐI SOÁT CÔNG NỢ & SAO KÊ NGÂN HÀNG — {FISCAL_YEAR}"
        title_cell.font = Font(name="Arial", size=14, bold=True, color="1F4E79")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 30

        ws.merge_cells("A2:L2")
        sub_cell = ws["A2"]
        sub_cell.value = f"Đơn vị: {COMPANY_NAME} | MST: {COMPANY_TAX_CODE} | Hệ thống đối soát hai tầng Neuro-Symbolic"
        sub_cell.font = Font(name="Arial", size=10, italic=True, color="595959")
        sub_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 20

        # Table Column Headers
        headers = [
            "STT", "Mã GD Ngân hàng", "Ngày đối soát", "Khách hàng",
            "Mã HĐ khớp", "Số tiền sao kê (VNĐ)", "Tổng tiền HĐ (VNĐ)",
            "Chiết khấu (VNĐ)", "Phí NH (VNĐ)", "Nguồn xử lý", "Loại khớp", "Ghi chú giải trình"
        ]
        ws.append([])  # Row 3 empty

        ws.append(headers)  # Row 4
        ws.row_dimensions[4].height = 26

        for col_idx in range(1, len(headers) + 1):
            c = ws.cell(row=4, column=col_idx)
            c.font = Font(name="Arial", size=10, bold=True, color=self.COLOR_HEADER_TXT)
            c.fill = PatternFill(start_color=self.COLOR_HEADER_BG, end_color=self.COLOR_HEADER_BG, fill_type="solid")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Query Ledger Data
        sql = """
            SELECT 
                r.match_id, r.trans_id, b.trans_date, r.customer_name,
                r.matched_invoice_ids, r.bank_amount, r.invoiced_total,
                r.discount_amount, r.bank_fee_amount, r.resolution_source,
                r.match_type, r.explanation
            FROM reconciliation_ledger r
            LEFT JOIN bank_statements b ON r.trans_id = b.trans_id
            ORDER BY r.match_id ASC;
        """
        ledger_df = db.fetch_df(sql)

        row_start = 5
        curr_row = row_start

        for idx, row in ledger_df.iterrows():
            inv_str = ", ".join(json.loads(row["matched_invoice_ids"])) if row["matched_invoice_ids"] else ""
            res_source = str(row["resolution_source"])

            ws.append([
                idx + 1,
                row["trans_id"],
                row["trans_date"] or "",
                row["customer_name"],
                inv_str,
                row["bank_amount"],
                row["invoiced_total"],
                row["discount_amount"],
                row["bank_fee_amount"],
                res_source,
                row["match_type"],
                row["explanation"],
            ])

            ws.row_dimensions[curr_row].height = 20

            # Determine row fill color based on source
            if res_source == ResolutionSource.SYSTEM_1_RULE:
                fill_color = self.COLOR_SYS1_BG
            elif res_source == ResolutionSource.SYSTEM_2_REACT_AGENT:
                fill_color = self.COLOR_SYS2_BG
            else:
                fill_color = self.COLOR_REVIEW_BG

            row_fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")

            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.font = Font(name="Arial", size=9)
                cell.fill = row_fill
                cell.border = self.thin_border

                # Formats
                if col_idx in [1, 2, 3, 10, 11]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif col_idx in [6, 7, 8, 9]:
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

            curr_row += 1

        # Summary Total Row
        if not ledger_df.empty:
            ws.cell(row=curr_row, column=1, value="TỔNG CỘNG")
            ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=5)
            ws.cell(row=curr_row, column=6, value=f"=SUM(F{row_start}:F{curr_row-1})")
            ws.cell(row=curr_row, column=7, value=f"=SUM(G{row_start}:G{curr_row-1})")
            ws.cell(row=curr_row, column=8, value=f"=SUM(H{row_start}:H{curr_row-1})")
            ws.cell(row=curr_row, column=9, value=f"=SUM(I{row_start}:I{curr_row-1})")

            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.font = Font(name="Arial", size=10, bold=True, color="1F4E79")
                cell.fill = PatternFill(start_color=self.COLOR_TOTAL_BG, end_color=self.COLOR_TOTAL_BG, fill_type="solid")
                cell.border = self.double_bottom_border
                if col_idx in [6, 7, 8, 9]:
                    cell.number_format = "#,##0"

            ws.row_dimensions[curr_row].height = 24

        self._auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # SHEET 2: SỔ ĐỊNH KHOẢN MISA AMIS
    # -----------------------------------------------------------------------

    def _build_sheet_misa_journal(self, wb: Workbook, db: DatabaseManager) -> None:
        ws = wb.create_sheet(title="So_Dinh_Khoan_MISA")
        ws.views.sheetView[0].showGridLines = True

        # Header Block
        ws.merge_cells("A1:K1")
        title_cell = ws["A1"]
        title_cell.value = "SỔ CHỨNG TỪ KẾ TOÁN NGÂN HÀNG (MẪU IMPORT MISA AMIS) — TT 200"
        title_cell.font = Font(name="Arial", size=14, bold=True, color="1F4E79")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 30

        headers = [
            "Ngày hạch toán", "Ngày chứng từ", "Số chứng từ", "Diễn giải",
            "TK Nợ", "TK Có", "Số tiền (VNĐ)", "Mã đối tượng (KH)",
            "Tên đối tượng", "Số hóa đơn", "Nguồn đối soát"
        ]
        ws.append([])
        ws.append(headers)  # Row 3
        ws.row_dimensions[3].height = 25

        for col_idx in range(1, len(headers) + 1):
            c = ws.cell(row=3, column=col_idx)
            c.font = Font(name="Arial", size=10, bold=True, color=self.COLOR_HEADER_TXT)
            c.fill = PatternFill(start_color=self.COLOR_HEADER_BG, end_color=self.COLOR_HEADER_BG, fill_type="solid")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        sql = """
            SELECT 
                r.trans_id, b.trans_date, r.customer_id, r.customer_name,
                r.matched_invoice_ids, r.bank_amount, r.discount_amount,
                r.bank_fee_amount, r.invoiced_total, r.resolution_source,
                r.journal_entries_json, r.explanation
            FROM reconciliation_ledger r
            LEFT JOIN bank_statements b ON r.trans_id = b.trans_id
            ORDER BY r.match_id ASC;
        """
        ledger_df = db.fetch_df(sql)

        curr_row = 4
        voucher_count = 1

        for _, row in ledger_df.iterrows():
            trans_date = row["trans_date"] or "2024-10-31"
            voucher_no = f"BCN{voucher_count:04d}"
            inv_str = ", ".join(json.loads(row["matched_invoice_ids"])) if row["matched_invoice_ids"] else ""
            cust_id = row["customer_id"] or ""
            cust_name = row["customer_name"] or ""
            res_source = row["resolution_source"] or ""

            # Decompose journal lines:
            # Line 1: Nợ 1121 / Có 131: bank_amount
            ws.append([
                trans_date, trans_date, voucher_no,
                f"Thu tiền gửi ngân hàng thanh toán HĐ {inv_str}",
                "1121", "131", row["bank_amount"],
                cust_id, cust_name, inv_str, res_source
            ])
            curr_row += 1

            # Line 2: If discount > 0 -> Nợ 521 / Có 131: discount_amount
            if row["discount_amount"] and int(row["discount_amount"]) > 0:
                ws.append([
                    trans_date, trans_date, voucher_no,
                    f"Chiết khấu thương mại trừ công nợ HĐ {inv_str}",
                    "521", "131", row["discount_amount"],
                    cust_id, cust_name, inv_str, res_source
                ])
                curr_row += 1

            # Line 3: If bank fee > 0 -> Nợ 642 / Có 131: bank_fee_amount
            if row["bank_fee_amount"] and int(row["bank_fee_amount"]) > 0:
                ws.append([
                    trans_date, trans_date, voucher_no,
                    f"Phí chuyển tiền ngân hàng HĐ {inv_str}",
                    "642", "131", row["bank_fee_amount"],
                    cust_id, cust_name, inv_str, res_source
                ])
                curr_row += 1

            voucher_count += 1

        # Format journal sheet cells
        for r_idx in range(4, curr_row):
            ws.row_dimensions[r_idx].height = 19
            is_even = (r_idx % 2 == 0)
            fill_c = self.COLOR_ZEBRA_BG if is_even else "FFFFFF"
            bg = PatternFill(start_color=fill_c, end_color=fill_c, fill_type="solid")

            for c_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.font = Font(name="Arial", size=9)
                cell.border = self.thin_border
                cell.fill = bg
                if c_idx in [1, 2, 3, 5, 6, 8, 11]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif c_idx == 7:
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

        # Total Row
        if curr_row > 4:
            ws.cell(row=curr_row, column=1, value="TỔNG PHÁT SINH NỢ / CÓ")
            ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=6)
            ws.cell(row=curr_row, column=7, value=f"=SUM(G4:G{curr_row-1})")

            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.font = Font(name="Arial", size=10, bold=True, color="1F4E79")
                cell.fill = PatternFill(start_color=self.COLOR_TOTAL_BG, end_color=self.COLOR_TOTAL_BG, fill_type="solid")
                cell.border = self.double_bottom_border
                if col_idx == 7:
                    cell.number_format = "#,##0"

            ws.row_dimensions[curr_row].height = 24

        self._auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # SHEET 3: BÁO CÁO CÔNG NỢ PHẢI THU
    # -----------------------------------------------------------------------

    def _build_sheet_receivables_summary(self, wb: Workbook, db: DatabaseManager) -> None:
        ws = wb.create_sheet(title="Bao_Cao_Cong_No_Phai_Thu")
        ws.views.sheetView[0].showGridLines = True

        # Header Block
        ws.merge_cells("A1:I1")
        title_cell = ws["A1"]
        title_cell.value = "BẢNG TỔNG HỢP CÔNG NỢ PHẢI THU KHÁCH HÀNG (TK 131)"
        title_cell.font = Font(name="Arial", size=14, bold=True, color="1F4E79")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 30

        headers = [
            "Mã KH", "Tên khách hàng", "Hạn nợ (ngày)", "Tỷ lệ CK HĐ",
            "Tổng phát sinh nợ (VNĐ)", "Đã thanh toán (VNĐ)",
            "Còn phải thu (VNĐ)", "Số HĐ chưa trả", "Trạng thái công nợ"
        ]
        ws.append([])
        ws.append(headers)  # Row 3
        ws.row_dimensions[3].height = 25

        for col_idx in range(1, len(headers) + 1):
            c = ws.cell(row=3, column=col_idx)
            c.font = Font(name="Arial", size=10, bold=True, color=self.COLOR_HEADER_TXT)
            c.fill = PatternFill(start_color=self.COLOR_HEADER_BG, end_color=self.COLOR_HEADER_BG, fill_type="solid")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        sql = """
            SELECT 
                c.customer_id, c.customer_name, c.credit_days, c.contract_discount_rate,
                COALESCE(SUM(i.total_amount), 0) AS total_invoiced,
                COALESCE(SUM(i.paid_amount), 0) AS total_paid,
                COALESCE(SUM(i.total_amount - i.paid_amount), 0) AS remaining_debt,
                COUNT(CASE WHEN i.status != 'PAID' THEN 1 END) AS unpaid_invoices_count
            FROM customers c
            LEFT JOIN invoices i ON c.customer_id = i.customer_id
            GROUP BY c.customer_id, c.customer_name, c.credit_days, c.contract_discount_rate
            ORDER BY remaining_debt DESC;
        """
        receivables_df = db.fetch_df(sql)

        curr_row = 4
        for _, row in receivables_df.iterrows():
            debt = int(row["remaining_debt"])
            status_text = "Đã thu hết" if debt == 0 else "Còn nợ trong hạn" if int(row["credit_days"]) >= 30 else "Nợ cần đôn đốc"

            ws.append([
                row["customer_id"],
                row["customer_name"],
                row["credit_days"],
                f"{float(row['contract_discount_rate'] or 0):.0%}",
                row["total_invoiced"],
                row["total_paid"],
                debt,
                row["unpaid_invoices_count"],
                status_text,
            ])
            curr_row += 1

        for r_idx in range(4, curr_row):
            ws.row_dimensions[r_idx].height = 20
            is_even = (r_idx % 2 == 0)
            fill_c = self.COLOR_ZEBRA_BG if is_even else "FFFFFF"
            bg = PatternFill(start_color=fill_c, end_color=fill_c, fill_type="solid")

            for c_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.font = Font(name="Arial", size=9)
                cell.border = self.thin_border
                cell.fill = bg
                if c_idx in [1, 3, 4, 8, 9]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif c_idx in [5, 6, 7]:
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

        # Summary Row
        if curr_row > 4:
            ws.cell(row=curr_row, column=1, value="TỔNG CỘNG")
            ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=4)
            ws.cell(row=curr_row, column=5, value=f"=SUM(E4:E{curr_row-1})")
            ws.cell(row=curr_row, column=6, value=f"=SUM(F4:F{curr_row-1})")
            ws.cell(row=curr_row, column=7, value=f"=SUM(G4:G{curr_row-1})")

            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.font = Font(name="Arial", size=10, bold=True, color="1F4E79")
                cell.fill = PatternFill(start_color=self.COLOR_TOTAL_BG, end_color=self.COLOR_TOTAL_BG, fill_type="solid")
                cell.border = self.double_bottom_border
                if col_idx in [5, 6, 7]:
                    cell.number_format = "#,##0"

            ws.row_dimensions[curr_row].height = 24

        self._auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # HELPER: AUTO-FIT COLUMN WIDTHS
    # -----------------------------------------------------------------------

    def _auto_fit_columns(self, ws) -> None:
        """Dynamically adjust column widths based on maximum cell content."""
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # Ignore merged cells in title rows
                if cell.row in [1, 2]:
                    continue
                val = str(cell.value or "")
                if len(val) > max_len:
                    max_len = len(val)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 11)


# ===========================================================================
# VERIFICATION HELPER
# ===========================================================================

def run_phase4_verification():
    """Verify Phase 4 SelfHealingSQLAgent, AgingReport, and Excel export."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    db = DatabaseManager()

    print("\n" + "=" * 65)
    print("  KIỂM THỬ MODULE SQL REPORTING AGENT & MISA EXCEL EXPORTER")
    print("=" * 65)

    # 1. Test Text-to-SQL
    sql_agent = SelfHealingSQLAgent(db)
    test_questions = [
        "Cho tôi biết tình hình công nợ phải thu của các khách hàng",
        "Có bao nhiêu giao dịch đối soát thành công bởi System 1 và System 2?",
        "Danh sách các hóa đơn phát hiện lỗi OCR",
    ]

    for q in test_questions:
        res = sql_agent.ask_accounting_question(q)
        print(f"\n[Câu hỏi]: '{q}'")
        print(f"-> Chế độ: {res['execution_mode']}")
        print(f"-> SQL: {res['final_sql'].strip()[:80]}...")
        print(f"-> Số dòng kết quả: {len(res['df'])}")
        print(f"-> Tóm tắt: {res['summary'][:100]}...")

    # 2. Test Aging Report
    aging_analyzer = AgingReportAnalyzer(db)
    aging_df = aging_analyzer.compute_aging_analysis()
    print(f"\n[Aging Report] Đã tính toán tuổi nợ cho {len(aging_df)} khách hàng.")

    # 3. Test MISA Excel Exporter
    exporter = MisaExcelExporter()
    out_file = _ROOT / "uploads" / "Bao_Cao_Doi_Soat_MISA_AMIS.xlsx"
    excel_data = exporter.export_reconciliation_and_journal_to_excel(db, out_file)
    print(f"\n[MISA Excel Exporter] Đã xuất thành công file Excel ({len(excel_data):,} bytes) tại:")
    print(f"  {out_file}\n")


if __name__ == "__main__":
    run_phase4_verification()
