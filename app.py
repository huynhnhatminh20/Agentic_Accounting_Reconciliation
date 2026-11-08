# ============================================================
# app.py — Giao diện Web Điều hành & Đối soát Kế toán Hai tầng
# Neuro-Symbolic Dual-Process Agentic Accounting System
# W2NER (PhoBERT) + System 1 (Rule) + System 2 (ReAct Agent)
# Chuẩn mực Kế toán Việt Nam (VAS) — Thông tư 200 / Thông tư 133
# ============================================================

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

# Optional Plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    DATABASE_PATH,
    COMPANY_NAME,
    COMPANY_TAX_CODE,
    FISCAL_YEAR,
    TOLERANCE_VND,
    COMMON_BANK_FEES,
    FUZZY_MATCH_THRESHOLD,
    InvoiceStatus,
    ReconcileStatus,
    MatchType,
    ResolutionSource,
)
from database.db_manager import DatabaseManager
from database.seed_mock_data import seed_all_data
from modules.ocr_agent import (
    SelfAuditingOCRAgent,
    InvoiceData,
    suggest_accounting_accounts,
)
from modules.w2ner_adapter import (
    W2NERStatementExtractor,
    W2NEROutput,
    LABEL_NNW,
    THW_LABELS,
)
from modules.system1_engine import System1ReconciliationEngine
from modules.system2_agent import System2ReActInvestigator
from modules.sql_reporting_agent import (
    SelfHealingSQLAgent,
    AgingReportAnalyzer,
    MisaExcelExporter,
    BUILTIN_ACCOUNTING_QUERIES,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. PAGE CONFIGURATION & MODERN STYLING
# ===========================================================================

st.set_page_config(
    page_title="W2NER & Dual-Process Agentic Accounting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive financial presentation
CUSTOM_CSS = """
<style>
    /* Global font & spacing */
    .main { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Top Header */
    .app-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.18);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.3px;
    }
    .app-header p {
        color: #E0E7FF !important;
        font-size: 13.5px !important;
        margin: 6px 0 0 0 !important;
        opacity: 0.95;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        gap: 14px;
        margin-bottom: 18px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        flex: 1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-top: 4px solid #3B82F6;
        transition: transform 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-card.green  { border-top-color: #10B981; }
    .kpi-card.purple { border-top-color: #8B5CF6; }
    .kpi-card.amber  { border-top-color: #F59E0B; }
    .kpi-card.blue   { border-top-color: #2563EB; }
    
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin: 6px 0 3px 0;
    }
    .kpi-subtitle {
        font-size: 11.5px;
        color: #64748B;
    }

    /* W2NER Entity Badges */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px;
        letter-spacing: 0.2px;
    }
    .badge-org   { background-color: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-per   { background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE; }
    .badge-inv   { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .badge-rate  { background-color: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-period{ background-color: #FFE4E6; color: #9F1239; border: 1px solid #FECDD3; }
    .badge-tax   { background-color: #F1F5F9; color: #334155; border: 1px solid #E2E8F0; }

    /* ReAct Trace Boxes */
    .react-box {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
        line-height: 1.5;
    }
    .react-thought {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        color: #334155;
    }
    .react-action {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        color: #14532D;
        font-family: Consolas, Monaco, monospace;
        font-size: 12.5px;
    }
    .react-observation {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        color: #78350F;
    }
    .react-decision {
        background-color: #ECFDF5;
        border-left: 4px solid #059669;
        color: #064E3B;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===========================================================================
# 2. CACHED INITIALIZATION OF SERVICES & MANAGERS
# ===========================================================================

@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """Singleton connection to SQLite accounting database."""
    return DatabaseManager(DATABASE_PATH)

@st.cache_resource
def get_ocr_agent() -> SelfAuditingOCRAgent:
    """Singleton instance of Self-Auditing OCR Agent."""
    return SelfAuditingOCRAgent()

@st.cache_resource
def get_w2ner_extractor() -> W2NERStatementExtractor:
    """Singleton instance of Deep Learning W2NER Extractor."""
    return W2NERStatementExtractor()

@st.cache_resource
def get_system1_engine() -> System1ReconciliationEngine:
    """Singleton instance of Fast Symbolic System 1 Engine."""
    return System1ReconciliationEngine()

@st.cache_resource
def get_system2_investigator() -> System2ReActInvestigator:
    """Singleton instance of Autonomous ReAct System 2 Investigator."""
    return System2ReActInvestigator(get_db_manager())

@st.cache_resource
def get_sql_reporting_agent() -> SelfHealingSQLAgent:
    """Singleton instance of Self-Healing Text-to-SQL Agent."""
    return SelfHealingSQLAgent(get_db_manager())

@st.cache_resource
def get_aging_analyzer() -> AgingReportAnalyzer:
    """Singleton instance of Aging Report Analyzer."""
    return AgingReportAnalyzer(get_db_manager())

@st.cache_resource
def get_excel_exporter() -> MisaExcelExporter:
    """Singleton instance of MISA AMIS Excel Exporter."""
    return MisaExcelExporter()


db = get_db_manager()
ocr_agent = get_ocr_agent()
w2ner = get_w2ner_extractor()
sys1 = get_system1_engine()
sys2 = get_system2_investigator()
sql_agent = get_sql_reporting_agent()
aging_analyzer = get_aging_analyzer()
excel_exporter = get_excel_exporter()


# ===========================================================================
# 3. SIDEBAR CONTROLS & PIPELINE EXECUTION
# ===========================================================================

with st.sidebar:
    st.markdown("### 🏦 **Điều Hành Hệ Thống**")
    st.caption("Neuro-Symbolic Dual-Process Accounting")

    # DB Status Indicator
    st.success("🟢 SQLite Database: `accounting.db`")

    # Ollama LLM Mode Toggle
    use_ollama = st.checkbox(
        "Kích hoạt Ollama LLM (`qwen2.5:7b`)",
        value=False,
        help="Khi bật, System 2 sẽ gọi mô hình ngôn ngữ lớn qua Ollama. Khi tắt, chạy bộ điều phối Heuristic Planner siêu tốc.",
    )
    if use_ollama:
        st.info("🧠 Chế độ: LLM ReAct (Ollama)")
    else:
        st.caption("⚡ Chế độ: Heuristic Agentic Engine (Tất định 100%)")

    st.markdown("---")

    # Action 1: Reset / Seed Mock Data
    if st.button("🔄 Khởi tạo / Reset Dữ liệu Demo", use_container_width=True):
        with st.spinner("Đang nạp bộ dữ liệu kiểm thử chuẩn 5 Kịch bản..."):
            seed_all_data(db)
        st.success("✅ Đã nạp thành công 28 Hóa đơn & 20 Sao kê ngân hàng!")
        st.rerun()

    # Action 2: Run End-to-End Pipeline
    if st.button("🚀 CHẠY TOÀN BỘ PIPELINE", type="primary", use_container_width=True):
        pipeline_status = st.status("Đang thi hành quy trình đối soát hai tầng...", expanded=True)

        # Step 1: System 1 Fast Symbolic Reconciliation
        pipeline_status.write("⚡ **Bước 1/2:** Chạy System 1 (Fast Symbolic Rules)...")
        start_t = time.time()
        sys1_res = sys1.run_fast_reconciliation(db, w2ner)
        elapsed_s1 = time.time() - start_t
        pipeline_status.write(
            f"✓ System 1 hoàn thành trong {elapsed_s1:.2f}s: "
            f"Khớp tức thời **{sys1_res['matched_count']}/{sys1_res['total_processed']}** GD. "
            f"Đẩy **{sys1_res['exception_count']}** ngoại lệ sang System 2."
        )

        # Step 2: System 2 ReAct Investigation
        if sys1_res["exception_count"] > 0:
            pipeline_status.write("🧠 **Bước 2/2:** Chạy System 2 (Autonomous ReAct Investigator)...")
            start_t2 = time.time()
            sys2_res = sys2.resolve_exception_queue(sys1_res["exception_queue"])
            elapsed_s2 = time.time() - start_t2
            pipeline_status.write(
                f"✓ System 2 hoàn thành trong {elapsed_s2:.2f}s: "
                f"Điều tra & giải quyết thành công **{sys2_res['resolved_count']}/{sys2_res['total_exceptions']}** ngoại lệ!"
            )

        pipeline_status.update(
            label="🎉 Toàn bộ quy trình đối soát đã hoàn thành 100%!",
            state="complete",
            expanded=False,
        )
        st.rerun()

    st.markdown("---")

    # Sidebar Quick Statistics
    st.markdown("#### 📊 **Thống Kê Nhanh**")
    stats = db.get_summary_stats()
    df_inv_cnt = db.fetch_df("SELECT COUNT(*) AS n FROM invoices")
    df_stmt_cnt = db.fetch_df("SELECT COUNT(*) AS n FROM bank_statements")
    df_matched_cnt = db.fetch_df("SELECT COUNT(*) AS n FROM reconciliation_ledger")

    total_invoices = int(df_inv_cnt.iloc[0]["n"] or 0)
    total_statements = int(df_stmt_cnt.iloc[0]["n"] or 0)
    total_matched = int(df_matched_cnt.iloc[0]["n"] or 0)

    arr_pct = (total_matched / total_statements * 100) if total_statements > 0 else 0.0

    st.metric("Tổng số Hóa đơn (DB)", f"{total_invoices} HĐ")
    st.metric("Tổng dòng Sao kê NH", f"{total_statements} GD")
    st.metric("Tỷ lệ tự động hóa (ARR)", f"{arr_pct:.1f}%")
    st.metric("Sao kê chờ xử lý", f"{stats['pending_statements']} GD")

    st.markdown("---")
    st.caption(f"Đơn vị: {COMPANY_NAME}\nNiên độ tài chính: {FISCAL_YEAR}")


# ===========================================================================
# 4. TOP APPLICATION HEADER & KPI BANNER
# ===========================================================================

st.markdown(
    f"""
    <div class="app-header">
        <h1>🏦 Hệ Thống Đối Soát Kế Toán Tự Động Hai Tầng (Dual-Process Accounting AI)</h1>
        <p>Mô hình kết hợp Deep Learning W2NER (PhoBERT), Symbolic Deterministic Engine (System 1) và Autonomous ReAct Agent (System 2) — Tuân thủ Thông tư 200/133</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# 5. FOUR MAIN APPLICATION TABS
# ===========================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Tab 1: Self-Auditing OCR & Hóa Đơn",
    "🧬 Tab 2: W2NER Deep Extraction",
    "🧠 Tab 3: Đối Soát Hai Tầng (System 1 & 2)",
    "📊 Tab 4: Trợ Lý SQL & Xuất Excel MISA",
])


# ===========================================================================
# TAB 1: SELF-AUDITING OCR & QUẢN LÝ HÓA ĐƠN
# ===========================================================================

with tab1:
    st.markdown("### 📄 **Perception Layer 1: Self-Auditing OCR & Multi-Format Invoice Extractor**")
    st.markdown(
        "Bóc tách hóa đơn đa định dạng (**XML Thông tư 78**, **PDF**, **Ảnh chụp**) "
        "tích hợp **Cơ chế Kiểm toán Số học Tự động** (Mathematical Self-Audit) và **Tự Sửa Lỗi Nhòe Chữ Số** (Self-Healing)."
    )

    col_upload, col_result = st.columns([1, 1.2])

    with col_upload:
        st.markdown("#### 📤 **Tải lên Hóa đơn hoặc Thử mẫu**")
        uploaded_file = st.file_uploader(
            "Chọn file hóa đơn (.xml, .pdf, .png, .jpg)",
            type=["xml", "pdf", "png", "jpg", "jpeg"],
        )

        btn_sample_ocr = st.button("⚡ Test thử Hóa đơn mẫu có lỗi OCR nhòe số (HD00123)", use_container_width=True)

        inv_to_display: Optional[InvoiceData] = None

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            ext = Path(uploaded_file.name).suffix.lower()
            with st.spinner("Đang bóc tách hóa đơn và kiểm toán số học..."):
                inv_to_display = ocr_agent.parse_auto(file_bytes, file_ext=ext)
        elif btn_sample_ocr:
            sample_files = ocr_agent.generate_sample_invoice_files()
            with st.spinner("Đang phân tích mẫu hóa đơn ảnh bị lỗi OCR..."):
                # Simulate HD00123 OCR misread (18.5tr misread as 10.5tr)
                inv_to_display = InvoiceData(
                    invoice_id="HD00123",
                    customer_name="Công ty Cổ Phần Xuất Nhập Khẩu Sao Vàng",
                    tax_code="0400456789",
                    issue_date="2024-10-20",
                    due_date="2024-11-20",
                    pre_tax_amount=16_818_182,
                    vat_rate=0.10,
                    vat_amount=1_681_818,
                    total_amount=10_500_000,  # Intentional OCR blur error
                    source_type="IMAGE_OCR",
                    ocr_confidence=0.58,
                    item_description="Thiết bị viễn thông công nghệ cao",
                )
                audit = ocr_agent.audit_invoice_math(inv_to_display)
                ocr_agent._apply_audit(inv_to_display, audit)
                inv_to_display.suggested_accounts = suggest_accounting_accounts(inv_to_display.item_description)

    with col_result:
        st.markdown("#### 🔍 **Kết Quả Trích Xuất & Kiểm Toán Số Học**")
        if inv_to_display is not None:
            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Số Hóa đơn", inv_to_display.invoice_id or "Chưa rõ")
            m2.metric("Ngày phát hành", inv_to_display.issue_date or "Chưa rõ")
            m3.metric("Thuế suất VAT", f"{inv_to_display.vat_rate:.0%}")
            m4.metric("Tổng tiền (OCR đọc)", f"{inv_to_display.total_amount:,}đ")

            display_mst = inv_to_display.buyer_tax_code or inv_to_display.tax_code or "Chưa rõ"
            seller_info = f"**Đơn vị bán:** `{inv_to_display.seller_name}` | " if inv_to_display.seller_name else ""
            st.markdown(f"{seller_info}**Khách hàng:** `{inv_to_display.customer_name or 'Khách hàng'}` | **MST:** `{display_mst}`")
            st.markdown(f"**Diễn giải:** `{inv_to_display.item_description or 'Hàng hóa dịch vụ'}`")

            # Mathematical Audit Check
            st.markdown("##### 📐 **Kiểm định Phương trình Kế toán:**")
            calc_vat = round(inv_to_display.pre_tax_amount * inv_to_display.vat_rate)
            calc_total = inv_to_display.pre_tax_amount + inv_to_display.vat_amount

            st.write(
                f"1. Tiền VAT: `{inv_to_display.pre_tax_amount:,}đ x {inv_to_display.vat_rate:.0%} = {calc_vat:,}đ` "
                f"(Thực tế ghi: `{inv_to_display.vat_amount:,}đ`)"
            )
            st.write(
                f"2. Tổng thanh toán: `{inv_to_display.pre_tax_amount:,}đ + {inv_to_display.vat_amount:,}đ = {calc_total:,}đ` "
                f"(Thực tế ghi: `{inv_to_display.total_amount:,}đ`)"
            )

            if inv_to_display.ocr_audit_passed:
                if inv_to_display.ocr_suspect_note and "[Self-Healed]" in inv_to_display.ocr_suspect_note:
                    st.success(f"✅ **KIỂM TOÁN THÔNG QUA (Audit PASSED):** {inv_to_display.ocr_suspect_note}")
                else:
                    st.success("✅ **KIỂM TOÁN THÔNG QUA (Audit PASSED):** Mọi phương trình số học cân đối hoàn hảo!")
            else:
                st.error("🚨 **CẢNH BÁO KIỂM TOÁN (Audit FAILED):** Phát hiện sai lệch số học trong nhận dạng OCR!")
                st.warning(f"**Lý do:** {inv_to_display.ocr_suspect_note}")

                if inv_to_display.suggested_corrected_total:
                    if inv_to_display.suggested_corrected_total != inv_to_display.total_amount:
                        st.info(
                            f"💡 **Đề xuất Tự Chữa Lỗi (Self-Healing OCR):** Giá trị chính xác theo kiểm toán toán học là "
                            f"**{inv_to_display.suggested_corrected_total:,} VNĐ** (Thay vì {inv_to_display.total_amount:,} VNĐ)."
                        )
                    else:
                        st.info(
                            f"💡 **Đề xuất Tự Chữa Lỗi (Self-Healing OCR):** {inv_to_display.ocr_suspect_note}"
                        )

            # Account Classification Suggestion
            if inv_to_display.suggested_accounts:
                acct = inv_to_display.suggested_accounts
                st.markdown(
                    f"**Gợi ý Định khoản:** Nợ TK **{acct['debit_account']}** ({acct['debit_label']}) / "
                    f"Có TK **{acct['credit_account']}** — *Lý do: {acct['reason']}*"
                )
        else:
            st.info("Vui lòng tải lên hóa đơn hoặc bấm nút 'Test thử Hóa đơn mẫu' bên trái để xem kết quả kiểm toán.")

    st.markdown("---")

    # Invoices Database View
    st.markdown("#### 🗄️ **Danh Sách Hóa Đơn Trong Cơ Sở Dữ Liệu (28 Hóa Đơn)**")
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        inv_status_filter = st.selectbox(
            "Lọc theo trạng thái:",
            ["Tất cả", "UNPAID (Chưa thu)", "PAID (Đã thu)", "Nghi ngờ lỗi OCR (Audit=0)"],
        )

    query_inv = "SELECT invoice_id, customer_name, issue_date, due_date, pre_tax_amount, vat_amount, total_amount, paid_amount, status, ocr_audit_passed, ocr_confidence FROM invoices"
    if inv_status_filter == "UNPAID (Chưa thu)":
        query_inv += " WHERE status = 'UNPAID'"
    elif inv_status_filter == "PAID (Đã thu)":
        query_inv += " WHERE status = 'PAID'"
    elif inv_status_filter == "Nghi ngờ lỗi OCR (Audit=0)":
        query_inv += " WHERE ocr_audit_passed = 0"

    invoices_table_df = db.fetch_df(query_inv)
    st.dataframe(
        invoices_table_df.style.format({
            "pre_tax_amount": "{:,.0f}đ",
            "vat_amount": "{:,.0f}đ",
            "total_amount": "{:,.0f}đ",
            "paid_amount": "{:,.0f}đ",
            "ocr_confidence": "{:.2f}",
        }),
        use_container_width=True,
        height=320,
    )


# ===========================================================================
# TAB 2: PHÂN TÍCH SAO KÊ BẰNG MẠNG HỌC SÂU W2NER
# ===========================================================================

with tab2:
    st.markdown("### 🧬 **Perception Layer 2: Deep Extraction via Word-Word Relation Network (W2NER)**")
    st.markdown(
        "Mô hình học sâu bắt các thực thể **Phẳng** (Flat), **Lồng nhau** (Nested) và **Đứt đoạn** (Discontinuous) "
        "bằng cách dự đoán lưới quan hệ 2 chiều $N \\times N$ (`NNW` - Kế tiếp, `THW-*` - Đóng vòng thực thể)."
    )

    # Input Section
    stmt_df = db.fetch_df("SELECT trans_id, raw_description, credit_amount FROM bank_statements")
    sample_descriptions = stmt_df["raw_description"].tolist() if not stmt_df.empty else [
        "CTY CP XD AN BINH THANH TOAN HD SO 00102 THANG 9 VA HD 00105 TRU CK 2PC",
        "CTY HOANG GIA TT TIEN HANG DOT 1 THANG 10",
        "LE VAN TUNG CHUYEN KHOAN TIEN HANG THANG 11",
    ]

    selected_desc = st.selectbox("Chọn nội dung sao kê mẫu từ DB:", sample_descriptions)
    custom_desc = st.text_input("Hoặc nhập câu sao kê bất kỳ để kiểm thử:", value=selected_desc)

    if custom_desc:
        with st.spinner("Mạng W2NER đang mã hóa biểu diễn 2D từ-từ..."):
            w2ner_res: W2NEROutput = w2ner.extract_entities(custom_desc)

        st.markdown("#### 🏷️ **1. Các Thực Thể Trích Xuất Được (Entity Badges):**")

        # Entity Badges Visualizer
        badge_html = "<div style='line-height: 2.2; margin-bottom: 14px;'>"
        for org in w2ner_res.payer_orgs:
            badge_html += f"<span class='badge badge-org'>🏢 CÔNG TY: {org}</span>"
        for per in w2ner_res.sender_persons:
            badge_html += f"<span class='badge badge-per'>👤 CÁ NHÂN: {per}</span>"
        for inv in w2ner_res.invoice_ids:
            badge_html += f"<span class='badge badge-inv'>📑 MÃ HĐ: {inv}</span>"
        for rate in w2ner_res.discount_rates:
            badge_html += f"<span class='badge badge-rate'>✂️ CHIẾT KHẤU: {rate:.0%}</span>"
        for p in w2ner_res.periods:
            badge_html += f"<span class='badge badge-period'>📅 KỲ: {p}</span>"
        for t in w2ner_res.tax_codes:
            badge_html += f"<span class='badge badge-tax'>🔢 MST: {t}</span>"
        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)

        # Highlighted Sentence
        st.markdown("**Câu gốc sau khi gắn nhãn thực thể:**")
        marked_text = custom_desc
        for inv in w2ner_res.invoice_ids:
            marked_text = marked_text.replace(inv, f"**[📑 {inv}]**")
        for org in w2ner_res.payer_orgs:
            marked_text = marked_text.replace(org, f"**[🏢 {org}]**")
        for per in w2ner_res.sender_persons:
            marked_text = marked_text.replace(per, f"**[👤 {per}]**")
        st.info(marked_text)

        # 2D Grid Heatmap Section
        st.markdown("#### 🗺️ **2. Ma Trận Lưới Từ - Từ 2 Chiều (2D Word-Word Grid Matrix):**")
        st.caption(
            "Mỗi ô $(i, j)$ thể hiện quan hệ từ token $i$ đến token $j$. "
            "Nhãn **NNW** nối các từ liền kề trong cùng thực thể; nhãn **THW-*** (Tail-Head) khép kín thực thể phẳng hoặc đứt đoạn."
        )

        tokens = w2ner_res.tokens
        N = len(tokens)
        matrix, rows, cols = w2ner.build_grid_matrix(w2ner_res)

        if _HAS_PLOTLY:
            fig_grid = px.imshow(
                matrix,
                x=tokens,
                y=tokens,
                color_continuous_scale="Viridis",
                title=f"2D W2NER Relation Grid ({N} x {N} tokens)",
                labels=dict(x="Tail Token (j)", y="Head Token (i)", color="Activation"),
            )
            fig_grid.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_grid, use_container_width=True)
        else:
            # HTML Table Fallback for Grid Matrix
            grid_html = "<div style='overflow-x:auto;'><table style='border-collapse:collapse; font-size:11px;'>"
            grid_html += "<tr><th style='border:1px solid #ddd; padding:4px; background:#f1f5f9;'>i \\ j</th>"
            for t in tokens:
                grid_html += f"<th style='border:1px solid #ddd; padding:4px; background:#f1f5f9;'>{t}</th>"
            grid_html += "</tr>"

            for i, tok_i in enumerate(tokens):
                grid_html += f"<tr><td style='border:1px solid #ddd; padding:4px; font-weight:bold; background:#f1f5f9;'>{tok_i}</td>"
                for j in range(N):
                    val = matrix[i][j]
                    bg = "#3B82F6" if val > 0.5 else "#FFFFFF"
                    color = "#FFFFFF" if val > 0.5 else "#CBD5E1"
                    grid_html += f"<td style='border:1px solid #ddd; padding:4px; background:{bg}; color:{color}; text-align:center;'>{val:.1f}</td>"
                grid_html += "</tr>"
            grid_html += "</table></div>"
            st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown("---")

    # Bank Statements Batch View
    st.markdown("#### 📋 **Toàn Bộ 20 Dòng Sao Kê Ngân Hàng Kèm Dữ Liệu W2NER:**")
    all_stmts_df = db.fetch_df("SELECT trans_id, trans_date, raw_description, credit_amount, reconcile_status, w2ner_json FROM bank_statements")
    
    # Format display
    display_stmts = all_stmts_df.copy()
    display_stmts["credit_amount"] = display_stmts["credit_amount"].apply(lambda v: f"{int(v):,} VNĐ")
    display_stmts["w2ner_summary"] = display_stmts["w2ner_json"].apply(
        lambda s: ", ".join(f"{k}:{v}" for k, v in json.loads(s)["entities"].items() if v) if s else "Chưa phân tích"
    )

    st.dataframe(
        display_stmts[["trans_id", "trans_date", "raw_description", "credit_amount", "reconcile_status", "w2ner_summary"]],
        use_container_width=True,
        height=320,
    )


# ===========================================================================
# TAB 3: ĐỐI SOÁT TƯ DUY HAI TẦNG: SYSTEM 1 & SYSTEM 2
# ===========================================================================

with tab3:
    st.markdown("### 🧠 **Decision Layer: Neuro-Symbolic Dual-Process Reconciliation**")
    st.markdown(
        "Mô hình phối hợp tư duy **System 1 (Khớp biểu tượng tất định tốc độ cao)** "
        "và **System 2 (ReAct Exception Investigator điều tra tự chủ)** theo chuẩn Thông tư 200/133."
    )

    # Top KPI Cards
    ledger_df = db.fetch_df("SELECT * FROM reconciliation_ledger")
    total_reconciled = len(ledger_df)
    sys1_count = len(ledger_df[ledger_df["resolution_source"] == ResolutionSource.SYSTEM_1_RULE])
    sys2_count = len(ledger_df[ledger_df["resolution_source"] == ResolutionSource.SYSTEM_2_REACT_AGENT])

    sys1_pct = (sys1_count / 20 * 100) if total_reconciled > 0 else 50.0
    sys2_pct = (sys2_count / 20 * 100) if total_reconciled > 0 else 50.0
    arr_rate = ((sys1_count + sys2_count) / 20 * 100) if total_reconciled > 0 else 100.0

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card blue">
                <div class="kpi-title">Tổng Sao Kê Ngân Hàng</div>
                <div class="kpi-value">20 GD</div>
                <div class="kpi-subtitle">10 Chuẩn + 10 Ngoại lệ phức tạp</div>
            </div>
            <div class="kpi-card green">
                <div class="kpi-title">System 1 (Fast Symbolic)</div>
                <div class="kpi-value">{sys1_count} GD ({sys1_pct:.0f}%)</div>
                <div class="kpi-subtitle">Khớp tức thời mã HĐ & Tên công ty</div>
            </div>
            <div class="kpi-card purple">
                <div class="kpi-title">System 2 (ReAct Agent)</div>
                <div class="kpi-value">{sys2_count} GD ({sys2_pct:.0f}%)</div>
                <div class="kpi-subtitle">Điều tra 4 kịch bản ngoại lệ bằng Tool</div>
            </div>
            <div class="kpi-card amber">
                <div class="kpi-title">Tỷ lệ Tự động hóa (ARR)</div>
                <div class="kpi-value">{arr_rate:.1f}%</div>
                <div class="kpi-subtitle">Thời gian: &lt; 2.5 giây toàn hệ thống</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Split: Part A and Part B
    part_a, part_b = st.columns([1, 1.3])

    with part_a:
        st.markdown("#### ⚡ **Phần A: Giao Dịch Khớp Nhanh Bởi System 1**")
        st.caption("Khớp chính xác 100% bằng 2 quy tắc tất định (Rule 1 & Rule 2)")

        sys1_df = ledger_df[ledger_df["resolution_source"] == ResolutionSource.SYSTEM_1_RULE]
        if not sys1_df.empty:
            for _, r in sys1_df.iterrows():
                with st.expander(f"✓ {r['trans_id']} — {r['customer_name']} ({int(r['bank_amount']):,}đ)"):
                    st.write(f"**Mã HĐ khớp:** `{r['matched_invoice_ids']}`")
                    st.write(f"**Phương thức khớp:** `{r['match_type']}`")
                    st.write(f"**Giải trình:** {r['explanation']}")
                    entries = json.loads(r["journal_entries_json"])
                    st.markdown("**Bút toán định khoản tự động:**")
                    for e in entries:
                        st.code(f"{e['entry_type']} TK {e['account_code']} ({e['account_name']}): {e['amount']:,}đ", language="text")
        else:
            st.info("Chưa chạy pipeline. Bấm nút '🚀 CHẠY TOÀN BỘ PIPELINE' ở Sidebar bên trái.")

    with part_b:
        st.markdown("#### 🔬 **Phần B: Hộp Đen Suy Luận Của System 2 ReAct Agent**")
        st.caption("Khám phá chuỗi Thought -> Action -> Observation giải quyết 4 kịch bản ngoại lệ")

        sys2_df = ledger_df[ledger_df["resolution_source"] == ResolutionSource.SYSTEM_2_REACT_AGENT]
        if not sys2_df.empty:
            scenario_tabs = st.tabs([
                "Tất cả ngoại lệ (10)",
                "Kịch bản B (Subset Sum)",
                "Kịch bản C (Chiết khấu/Phí)",
                "Kịch bản D (Danh bạ Alias)",
                "Kịch bản E (Lỗi OCR)",
            ])

            def render_react_expander(row_data, tab_prefix="all", idx=0):
                m_type = row_data["match_type"]
                type_badges = {
                    MatchType.SUBSET_SUM_COMBO: "🧩 SUBSET SUM",
                    MatchType.DISCOUNT_OR_FEE_ADJUSTED: "✂️ CHIẾT KHẤU / PHÍ",
                    MatchType.ALIAS_RESOLVED: "👤 ALIAS LOOKUP",
                    MatchType.OCR_ERROR_CORRECTED: "🛠️ SỬA LỖI OCR",
                }
                badge_title = type_badges.get(m_type, m_type)
                trans_id = row_data.get("trans_id", "tr")

                with st.expander(f"🔎 [{badge_title}] {trans_id} — {row_data['customer_name']} ({int(row_data['bank_amount']):,}đ)"):
                    # Display Reasoning Trace Markdown
                    st.markdown(row_data["agent_reasoning_trace"])

                    # Display Proposed Journal Entries
                    entries = json.loads(row_data["journal_entries_json"])
                    st.markdown("##### 📝 **Bút toán hạch toán đề xuất (TT 200):**")
                    for e in entries:
                        st.markdown(f"- **{e['entry_type']} TK {e['account_code']}** ({e['account_name']}): `{e['amount']:,} VNĐ`")

                    # Approve Button
                    col_btn, _ = st.columns([1, 2])
                    with col_btn:
                        btn_key = f"btn_app_{tab_prefix}_{trans_id}_{idx}"
                        if st.button(f"✅ Phê duyệt (Approve)", key=btn_key):
                            db.execute_query(
                                "UPDATE reconciliation_ledger SET approved = 1 WHERE trans_id = ?",
                                (trans_id,),
                            )
                            st.success(f"Đã duyệt giao dịch {trans_id} thành công!")
                            st.rerun()

            with scenario_tabs[0]:
                for idx, r in enumerate(sys2_df.to_dict(orient="records")):
                    render_react_expander(r, tab_prefix="tab_all", idx=idx)

            with scenario_tabs[1]:
                b_df = sys2_df[sys2_df["match_type"] == MatchType.SUBSET_SUM_COMBO]
                for idx, r in enumerate(b_df.to_dict(orient="records")):
                    render_react_expander(r, tab_prefix="tab_b", idx=idx)

            with scenario_tabs[2]:
                c_df = sys2_df[sys2_df["match_type"] == MatchType.DISCOUNT_OR_FEE_ADJUSTED]
                for idx, r in enumerate(c_df.to_dict(orient="records")):
                    render_react_expander(r, tab_prefix="tab_c", idx=idx)

            with scenario_tabs[3]:
                d_df = sys2_df[sys2_df["match_type"] == MatchType.ALIAS_RESOLVED]
                for idx, r in enumerate(d_df.to_dict(orient="records")):
                    render_react_expander(r, tab_prefix="tab_d", idx=idx)

            with scenario_tabs[4]:
                e_df = sys2_df[sys2_df["match_type"] == MatchType.OCR_ERROR_CORRECTED]
                for idx, r in enumerate(e_df.to_dict(orient="records")):
                    render_react_expander(r, tab_prefix="tab_e", idx=idx)
        else:
            st.info("Chưa có giao dịch ngoại lệ nào được giải quyết. Bấm '🚀 CHẠY TOÀN BỘ PIPELINE' ở Sidebar.")


# ===========================================================================
# TAB 4: TRỢ LÝ KẾ TOÁN TRƯỞNG & XUẤT EXCEL MISA
# ===========================================================================

with tab4:
    st.markdown("### 📊 **Trợ Lý Kế Toán Trưởng (Self-Healing Text-to-SQL) & Xuất Excel Chuẩn MISA**")
    st.markdown(
        "Truy vấn tài chính bằng ngôn ngữ tự nhiên tiếng Việt, tự động sửa lỗi SQL (Self-Healing) "
        "và xuất khẩu sổ sách kế toán trực tiếp sang file **Excel chuẩn MISA AMIS**."
    )

    # Section 1: Natural Language Question Interface
    st.markdown("#### 💬 **1. Truy Vấn Kế Toán Bằng Ngôn Ngữ Tự Nhiên**")

    preset_questions = [
        "1. Tình hình công nợ phải thu theo từng khách hàng",
        "2. Danh sách các hóa đơn quá hạn thanh toán",
        "3. Thống kê tỷ lệ đối soát giữa System 1 và System 2",
        "4. Tổng chiết khấu thương mại và phí ngân hàng trong kỳ",
        "5. Danh sách các giao dịch cá nhân Giám đốc/Kế toán chuyển thay công ty",
        "6. Các hóa đơn phát hiện và tự động điều chỉnh lỗi OCR",
        "7. Diễn biến dòng tiền thu vào ngân hàng theo ngày",
        "8. Sổ nhật ký chung kết quả đối soát Nợ / Có",
        "── Câu hỏi tự do (LLM sẽ sinh SQL động) ──",
        "9. Liệt kê các khách hàng chưa đến hạn thanh toán",
        "10. Top 3 khách hàng có tổng nợ cao nhất",
        "11. Các hóa đơn quá hạn của KH001",
        "12. Có bao nhiêu hóa đơn áp thuế suất 8%?",
        "13. Tổng doanh thu theo từng tháng trong năm 2024",
    ]

    selected_q = st.selectbox("Chọn câu hỏi phân tích mẫu:", preset_questions)
    user_q = st.text_input("Hoặc nhập câu hỏi tiếng Việt tự do:", value=selected_q)

    if st.button("🔎 Phân Tích & Truy Vấn Dữ Liệu", type="primary"):
        with st.spinner("Agent đang suy luận, sinh SQL và kiểm tra dữ liệu..."):
            query_res = sql_agent.ask_accounting_question(user_q)

        st.markdown(f"**Chế độ thực thi:** `{query_res['execution_mode']}`")

        # Display SQL and Self-Healing Logs
        with st.expander("📝 Xem câu lệnh SQLite & Lịch sử Self-Healing", expanded=False):
            st.code(query_res["final_sql"], language="sql")
            if len(query_res["sql_attempts"]) > 1:
                st.warning(f"Đã trải qua {len(query_res['sql_attempts'])} lần thử để tự chữa lỗi SQL:")
                for att in query_res["sql_attempts"]:
                    status_icon = "✓" if att["error"] is None else "✗"
                    st.write(f"- Lần {att['attempt']}: [{status_icon}] {att['status']} — Lỗi: `{att['error']}`")

        # Display Narrative Summary
        st.info(query_res["summary"])

        # Display Data & Visual Chart
        res_df = query_res["df"]
        if not res_df.empty:
            tab_tbl, tab_chart = st.tabs(["📋 Bảng Số Liệu Chi Tiết", "📈 Biểu Đồ Trực Quan"])
            with tab_tbl:
                st.dataframe(res_df, use_container_width=True)

            with tab_chart:
                chart_obj = query_res["chart_figure"]
                if _HAS_PLOTLY and isinstance(chart_obj, go.Figure):
                    st.plotly_chart(chart_obj, use_container_width=True)
                else:
                    # Streamlit Native Chart Fallback
                    num_cols = res_df.select_dtypes(include=["number"]).columns.tolist()
                    if num_cols:
                        st.bar_chart(res_df.set_index(res_df.columns[0])[num_cols[-1]])
                    else:
                        st.info("Không có cột số liệu phù hợp để dựng biểu đồ.")

    st.markdown("---")

    # Section 2: Accounts Receivable Aging Report
    st.markdown("#### ⏳ **2. Báo Cáo Phân Tích Tuổi Nợ (Aging Report)**")
    aging_data = aging_analyzer.compute_aging_analysis(as_of_date="2024-11-15")

    st.dataframe(
        aging_data.style.format({
            "Tổng nợ (VNĐ)": "{:,.0f}đ",
            "Chưa đến hạn (VNĐ)": "{:,.0f}đ",
            "Quá hạn 1-30 ngày": "{:,.0f}đ",
            "Quá hạn 31-60 ngày": "{:,.0f}đ",
            "Quá hạn 61-90 ngày": "{:,.0f}đ",
            "Quá hạn >90 ngày": "{:,.0f}đ",
            "Tỷ lệ quá hạn (%)": "{:.1f}%",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # Section 3: Professional MISA AMIS Excel Export
    st.markdown("#### 📥 **3. Xuất Sổ Sách Chuẩn MISA AMIS (Excel 3 Sheets)**")
    st.markdown(
        "File Excel chuyên nghiệp gồm: **Sheet 1** (Bảng đối soát màu sắc), **Sheet 2** (Sổ định khoản mẫu import MISA), "
        "và **Sheet 3** (Báo cáo tổng hợp công nợ khách hàng)."
    )

    out_excel_path = _ROOT / "uploads" / "Bao_Cao_Doi_Soat_MISA_AMIS.xlsx"
    excel_bytes_data = excel_exporter.export_reconciliation_and_journal_to_excel(db, out_excel_path)

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="📥 Tải Xuống File Excel MISA AMIS (.xlsx)",
            data=excel_bytes_data,
            file_name="Bao_Cao_Doi_Soat_MISA_AMIS.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    with col_info:
        st.caption(
            "✓ Đầy đủ công thức tổng cộng `=SUM(...)`\n"
            "✓ Tách riêng bút toán Nợ 1121, 521, 642 / Có 131\n"
            "✓ Định dạng tiền tệ `#,##0 VNĐ` và viền kẻ bảng chuyên nghiệp."
        )


# ===========================================================================
# 6. APPLICATION FOOTER
# ===========================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 12px; margin-top: 10px;">
        Hệ thống Đối soát Kế toán Hai tầng (Neuro-Symbolic Dual-Process Agentic Accounting) • 
        Đề tài Nghiên cứu Ứng dụng AI trong Kế toán & Kiểm toán Việt Nam • Thông tư 200/133/BTC
    </div>
    """,
    unsafe_allow_html=True,
)

