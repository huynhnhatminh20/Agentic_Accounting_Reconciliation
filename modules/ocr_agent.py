# ============================================================
# modules/ocr_agent.py
# Self-Auditing OCR & Multi-format Invoice Extractor
#
# Supports: XML (Thong tu 78 / VNPT e-invoice standard)
#           PDF (pdfplumber)
#           Image (PaddleOCR / VietOCR / Ollama VLM fallback)
#
# Key features:
#   - Mathematical self-audit with auto-heal suggestion
#   - Levenshtein-based digit confusion correction
#   - Accounting account auto-classification (TT200/133)
#   - Sample invoice file generator for Streamlit demo
# ============================================================

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import base64
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    OLLAMA_BASE_URL,
    VLM_MODEL,
    CHART_OF_ACCOUNTS,
    OCR_CONFIDENCE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    from lxml import etree as _lxml_etree
    _HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as _lxml_etree   # type: ignore
    _HAS_LXML = False

try:
    import pdfplumber as _pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False
    logger.warning("[OCR] pdfplumber not found — PDF parsing unavailable.")

try:
    from paddleocr import PaddleOCR as _PaddleOCR   # type: ignore
    _HAS_PADDLE = True
except ImportError:
    _HAS_PADDLE = False

try:
    from vietocr.tool.predictor import Predictor as _VietOCRPredictor  # type: ignore
    from vietocr.tool.config import Cfg as _VietOCRCfg                 # type: ignore
    _HAS_VIETOCR = True
except ImportError:
    _HAS_VIETOCR = False

try:
    import ollama as _ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# Standard Vietnamese e-invoice XML tag mapping (Thong tu 78 / VNPT / MISA)
# ---------------------------------------------------------------------------
XML_TAG_MAP: Dict[str, str] = {
    # Invoice header
    "SHDon":    "invoice_id",
    "KHHDon":   "invoice_series",    # Ky hieu
    "NLap":     "issue_date",
    "HVTNMBan": "seller_name",
    "MST":      "tax_code",
    # Buyer
    "Ten":      "customer_name",
    "MSTNMua":  "buyer_tax_code",
    "DChi":     "buyer_address",
    # Amounts
    "TgTCThue": "pre_tax_amount",    # Tong gia tri chua thue
    "TgTThue":  "vat_amount",        # Tong tien thue
    "TgTTtbSo": "total_amount",      # Tong tien thanh toan
    "TSuat":    "vat_rate",          # Thue suat %
    # Line items
    "TenHHDV":  "item_description",
    "DVTinh":   "unit",
    "SLuong":   "quantity",
    "DGia":     "unit_price",
    "ThTien":   "line_total",
}

# Common alternative tag names from different ERP vendors
XML_TAG_ALIASES: Dict[str, str] = {
    # VNPT
    "SoHD":     "invoice_id",
    "TenNMua":  "customer_name",
    "MaThue":   "tax_code",
    "TienThue": "vat_amount",
    "TongTien": "total_amount",
    "TienHang": "pre_tax_amount",
    # MISA
    "InvNo":    "invoice_id",
    "BuyerName":"customer_name",
    "TaxAmount":"vat_amount",
    "TotalAmount":"total_amount",
}
XML_TAG_MAP.update(XML_TAG_ALIASES)

# VAT rates recognized in Vietnam (fractional, e.g. 0.08)
KNOWN_VAT_RATES: List[float] = [0.0, 0.05, 0.08, 0.10]

# Digit confusion matrix for OCR self-heal
# Maps (ocr_digit -> likely_true_digit) based on visual similarity
DIGIT_CONFUSIONS: Dict[str, List[str]] = {
    "0": ["8", "6"],
    "1": ["7", "l", "I"],
    "5": ["6", "S", "s"],
    "6": ["0", "5"],
    "7": ["1", "T"],
    "8": ["0", "3", "9"],
    "9": ["8", "4"],
    "3": ["8"],
}

# Boilerplate terms in Vietnamese invoices that must not be used as company / person names
BOILERPLATE_NAMES = {
    "NGƯỜI BÁN HÀNG", "NGUOI BAN HANG", "ĐƠN VỊ BÁN HÀNG", "DON VI BAN HANG",
    "NGƯỜI MUA HÀNG", "NGUOI MUA HANG", "ĐƠN VỊ MUA HÀNG", "DON VI MUA HANG",
    "NGƯỜI MUA", "NGUOI MUA", "NGƯỜI BÁN", "NGUOI BAN",
    "BUYER", "SELLER", "(BUYER)", "(SELLER)",
    "KÝ BỞI", "KY BOI", "SIGNATURE VALID", "KÝ SỐ",
    "CHƯA RÕ", "KHÁCH HÀNG", "KHACH HANG", "NGƯỜI KÝ", "NGUOI KY",
    "THỦ TRƯỞNG ĐƠN VỊ", "KẾ TOÁN TRƯỞNG", "NGƯỜI LẬP BIỂU",
    "SỐ CCCD", "SO CCCD", "SỐ CCCD (ID CODE)", "SO CCCD (ID CODE)", "CCCD", "ID CODE",
    "SỐ CMND", "SO CMND", "CMND", "HÌNH THỨC THANH TOÁN", "HINH THUC THANH TOAN",
}


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class InvoiceData:
    """Normalized invoice record after extraction + audit."""
    invoice_id:        str   = ""
    invoice_series:    str   = ""
    customer_name:     str   = ""
    buyer_tax_code:    str   = ""
    seller_name:       str   = ""
    tax_code:          str   = ""
    issue_date:        str   = ""
    due_date:          str   = ""
    pre_tax_amount:    int   = 0
    vat_rate:          float = 0.10
    vat_amount:        int   = 0
    total_amount:      int   = 0
    item_description:  str   = ""
    source_type:       str   = "UNKNOWN"    # XML | PDF | IMAGE_OCR
    ocr_confidence:    float = 1.0
    ocr_audit_passed:  bool  = True
    ocr_suspect_note:  str   = ""
    raw_ocr_true_amount: Optional[int] = None
    suggested_corrected_total: Optional[int] = None
    suggested_accounts: Dict[str, str] = field(default_factory=dict)
    raw_fields:        Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_db_tuple(self) -> tuple:
        """Return tuple matching invoices table column order."""
        return (
            self.invoice_id,
            "",                       # customer_id — resolved later
            self.customer_name,
            self.buyer_tax_code or self.tax_code,
            self.issue_date,
            self.due_date or self.issue_date,
            self.pre_tax_amount,
            self.vat_rate,
            self.vat_amount,
            self.total_amount,
            0,                        # paid_amount
            "UNPAID",
            self.source_type,
            self.ocr_confidence,
            int(self.ocr_audit_passed),
            self.ocr_suspect_note or None,
            self.raw_ocr_true_amount,
        )


@dataclass
class AuditResult:
    passed: bool
    vat_check_ok: bool
    total_check_ok: bool
    expected_vat: int
    expected_total: int
    actual_vat: int
    actual_total: int
    note: str = ""
    suggested_corrected_total: Optional[int] = None
    corrected_field: str = ""


# ===========================================================================
# Levenshtein distance (char-level) — no external dependency
# ===========================================================================

def _levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


def _candidate_corrections(amount_int: int) -> List[int]:
    """
    Given an integer amount, return candidate corrected amounts
    by substituting each digit with its visually similar confusables.
    E.g. 10500000 -> [18500000, 10800000, ...]
    """
    s = str(amount_int)
    candidates: List[int] = []
    for idx, ch in enumerate(s):
        for sub in DIGIT_CONFUSIONS.get(ch, []):
            if sub.isdigit():
                new_s = s[:idx] + sub + s[idx + 1:]
                try:
                    candidates.append(int(new_s))
                except ValueError:
                    pass
    return list(set(candidates))


# ===========================================================================
# XML helpers
# ===========================================================================

def _parse_amount(text: Optional[str]) -> int:
    """Strip non-numeric chars and convert to int VND."""
    if not text:
        return 0
    cleaned = re.sub(r"[^\d]", "", str(text).strip())
    return int(cleaned) if cleaned else 0


def _parse_rate(text: Optional[str]) -> float:
    """Parse vat_rate string like '10%', '0.1', '8' -> float fraction."""
    if not text:
        return 0.10
    s = str(text).strip().replace("%", "").replace(",", ".")
    try:
        v = float(s)
        return v / 100.0 if v > 1 else v
    except ValueError:
        return 0.10


def _normalize_date(raw: str) -> str:
    """Try to convert various date formats to YYYY-MM-DD."""
    if not raw:
        return ""
    raw = raw.strip()
    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    cleaned = re.sub(r"\s+", "", raw)
    m = re.match(r"^(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", cleaned)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 1990 <= y <= 2040:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # YYYY/MM/DD or YYYY-MM-DD with slashes/dots
    m_iso2 = re.match(r"^(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})", cleaned)
    if m_iso2:
        y, mo, d = int(m_iso2.group(1)), int(m_iso2.group(2)), int(m_iso2.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 1990 <= y <= 2040:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # YYYYMMDD
    m2 = re.match(r"^(\d{4})(\d{2})(\d{2})", cleaned)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"
    return raw


def _xml_find_text(root: Any, tag: str) -> Optional[str]:
    """Find first element with *local name* matching tag (namespace-agnostic)."""
    # Try direct
    el = root.find(f".//{tag}")
    if el is not None and el.text:
        return el.text.strip()
    # Namespace-agnostic search
    for el in root.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local == tag and el.text:
            return el.text.strip()
    return None


# ===========================================================================
# Account classification rules
# ===========================================================================

ACCOUNT_KEYWORDS: List[Tuple[str, str, str]] = [
    # (keyword_pattern, account_code, account_name_short)
    # Fixed assets
    (r"may tinh|laptop|server|thiet bi|may moc|xe o to|oto|phuong tien", "211", "Tai san co dinh"),
    (r"phan mem|software|ban quyen", "213", "Tai san vo hinh"),
    # Goods / inventory
    (r"hang hoa|vat tu|nguyen lieu|hang mua|san pham", "156", "Hang hoa / Vat tu"),
    # Utilities
    (r"dien|nuoc|vien thong|internet|dien thoai", "642", "Chi phi quan ly - Tien ich"),
    # Office / stationery
    (r"van phong pham|in an|photo|muc in|giay|bao bi", "642", "Chi phi quan ly - VP pham"),
    # Advertising / marketing
    (r"quang cao|marketing|truyen thong|pr |media", "641", "Chi phi ban hang - QC"),
    # Freight / shipping
    (r"van chuyen|logistic|cuoc phi|freight|chuyen phat", "641", "Chi phi ban hang - Van chuyen"),
    # Consulting / audit
    (r"tu van|kiem toan|ke toan|luat|dich vu chuyen mon", "642", "Chi phi quan ly - DV chuyen mon"),
    # Repair / maintenance
    (r"sua chua|bao tri|bao duong|maintenance", "642", "Chi phi quan ly - Sua chua"),
    # Meals / accommodation
    (r"an uong|khach san|resort|du lich|nha hang|mon an", "642", "Chi phi quan ly - Tiep khach"),
    # Construction
    (r"xay dung|cong trinh|thi cong|ket cau|be tong", "241", "Xay dung co ban do dang"),
    # Medicine / pharma
    (r"duoc pham|thuoc|y te|benh|vaccine|sinh pham", "156", "Hang hoa - Duoc pham"),
    # Insurance
    (r"bao hiem|insurance", "642", "Chi phi quan ly - Bao hiem"),
    # Default
    (r".*", "642", "Chi phi quan ly DN (mac dinh)"),
]


def suggest_accounting_accounts(item_description: str) -> Dict[str, str]:
    """
    Rule-based account classification from item description.
    Returns {"debit_account": "...", "credit_account": "...", "reason": "..."}.
    """
    desc_lower = _vn_remove_accent(item_description.lower())
    for pattern, code, label in ACCOUNT_KEYWORDS:
        if re.search(pattern, desc_lower):
            return {
                "debit_account":  code,
                "debit_label":    label,
                "credit_account": "1121",
                "credit_label":   "Tien gui ngan hang VND",
                "reason":         f"Khop pattern '{pattern}' -> TK {code} ({label})",
            }
    return {
        "debit_account":  "642",
        "debit_label":    "Chi phi quan ly DN",
        "credit_account": "1121",
        "credit_label":   "Tien gui ngan hang VND",
        "reason":         "Mac dinh TK642",
    }


def _vn_remove_accent(text: str) -> str:
    """Strip Vietnamese diacritics using unicodedata NFKD decomposition."""
    import unicodedata as _ud
    nfkd = _ud.normalize("NFKD", text)
    return "".join(c for c in nfkd if not _ud.combining(c))


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class SelfAuditingOCRAgent:
    """
    Multi-format Vietnamese invoice parser with built-in math auditing.

    Usage:
        agent = SelfAuditingOCRAgent()
        inv   = agent.parse_xml_invoice("path/to/invoice.xml")
        inv   = agent.parse_pdf_invoice(pdf_bytes)
        inv   = agent.parse_image_invoice(img_bytes)
    """

    def __init__(
        self,
        ollama_base_url: str = OLLAMA_BASE_URL,
        vlm_model: str = VLM_MODEL,
        confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
    ):
        self.ollama_base_url = ollama_base_url
        self.vlm_model = vlm_model
        self.confidence_threshold = confidence_threshold
        self._paddle_ocr = None   # Lazy init
        self._vietocr   = None    # Lazy init
        logger.info("[OCR] SelfAuditingOCRAgent initialized.")

    # -----------------------------------------------------------------------
    # 1. XML PARSER
    # -----------------------------------------------------------------------

    def parse_xml_invoice(
        self, source: Union[str, bytes, Path]
    ) -> InvoiceData:
        """
        Parse a Vietnamese e-invoice XML file (TT78 / VNPT / MISA format).

        Args:
            source: file path (str/Path) or raw XML bytes.

        Returns:
            InvoiceData with source_type="XML".
        """
        try:
            if isinstance(source, (str, Path)) and Path(source).exists():
                with open(source, "rb") as f:
                    raw_bytes = f.read()
            elif isinstance(source, bytes):
                raw_bytes = source
            else:
                raw_bytes = str(source).encode("utf-8")

            # Parse with lxml for namespace handling
            if _HAS_LXML:
                parser = _lxml_etree.XMLParser(recover=True)
                root = _lxml_etree.fromstring(raw_bytes, parser=parser)
            else:
                root = _lxml_etree.fromstring(raw_bytes)

            raw: Dict[str, Any] = {}

            # Map all known tags
            for tag, field_name in XML_TAG_MAP.items():
                val = _xml_find_text(root, tag)
                if val and field_name not in raw:
                    raw[field_name] = val

            # Build InvoiceData
            inv = InvoiceData(
                invoice_id       = raw.get("invoice_id", ""),
                invoice_series   = raw.get("invoice_series", ""),
                customer_name    = raw.get("customer_name", ""),
                buyer_tax_code   = raw.get("buyer_tax_code", ""),
                seller_name      = raw.get("seller_name", ""),
                tax_code         = raw.get("tax_code", ""),
                issue_date       = _normalize_date(raw.get("issue_date", "")),
                pre_tax_amount   = _parse_amount(raw.get("pre_tax_amount")),
                vat_rate         = _parse_rate(raw.get("vat_rate", "10")),
                vat_amount       = _parse_amount(raw.get("vat_amount")),
                total_amount     = _parse_amount(raw.get("total_amount")),
                item_description = raw.get("item_description", ""),
                source_type      = "XML",
                ocr_confidence   = 1.0,
                raw_fields       = raw,
            )

            # Math audit
            audit = self.audit_invoice_math(inv)
            self._apply_audit(inv, audit)

            # Account suggestion
            inv.suggested_accounts = suggest_accounting_accounts(inv.item_description)

            logger.info(
                f"[OCR] XML parsed: {inv.invoice_id} | "
                f"total={inv.total_amount:,} | audit={'OK' if inv.ocr_audit_passed else 'FAIL'}"
            )
            return inv

        except Exception as exc:
            logger.error(f"[OCR] XML parse error: {exc}")
            raise

    # -----------------------------------------------------------------------
    # 2. PDF PARSER
    # -----------------------------------------------------------------------

    def parse_pdf_invoice(
        self, source: Union[str, bytes, Path]
    ) -> InvoiceData:
        """
        Extract invoice fields from a PDF using pdfplumber.
        Falls back to plain regex extraction from the full text.
        """
        if not _HAS_PDFPLUMBER:
            raise ImportError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            )

        try:
            if isinstance(source, (str, Path)):
                pdf_obj = _pdfplumber.open(source)
            else:
                pdf_obj = _pdfplumber.open(io.BytesIO(source))

            full_text = ""
            all_tables: List[List] = []

            with pdf_obj as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    full_text += text + "\n"
                    tables = page.extract_tables()
                    for tbl in tables:
                        all_tables.extend(tbl)

            raw = self._regex_extract_from_text(full_text)

            # Try to refine amounts from table cells
            self._enrich_from_tables(raw, all_tables)

            inv = InvoiceData(
                invoice_id       = raw.get("invoice_id", ""),
                customer_name    = raw.get("customer_name", ""),
                buyer_tax_code   = raw.get("buyer_tax_code", ""),
                seller_name      = raw.get("seller_name", ""),
                tax_code         = raw.get("tax_code", ""),
                issue_date       = _normalize_date(raw.get("issue_date", "")),
                pre_tax_amount   = raw.get("pre_tax_amount", 0),
                vat_rate         = raw.get("vat_rate", 0.10),
                vat_amount       = raw.get("vat_amount", 0),
                total_amount     = raw.get("total_amount", 0),
                item_description = raw.get("item_description", ""),
                source_type      = "PDF",
                ocr_confidence   = 0.90,
                raw_fields       = raw,
            )

            audit = self.audit_invoice_math(inv)
            self._apply_audit(inv, audit)
            inv.suggested_accounts = suggest_accounting_accounts(inv.item_description)

            logger.info(
                f"[OCR] PDF parsed: {inv.invoice_id} | total={inv.total_amount:,} | "
                f"audit={'OK' if inv.ocr_audit_passed else 'FAIL'}"
            )
            return inv

        except Exception as exc:
            logger.warning(f"[OCR] PDF parse error ({exc}) — falling back to plain-text regex extraction.")
            try:
                if isinstance(source, (str, Path)) and Path(source).exists():
                    full_text = Path(source).read_text(encoding="utf-8", errors="ignore")
                elif isinstance(source, bytes):
                    full_text = source.decode("utf-8", errors="ignore")
                else:
                    full_text = str(source)

                raw = self._regex_extract_from_text(full_text)
                inv = InvoiceData(
                    invoice_id       = raw.get("invoice_id", ""),
                    customer_name    = raw.get("customer_name", ""),
                    buyer_tax_code   = raw.get("buyer_tax_code", ""),
                    seller_name      = raw.get("seller_name", ""),
                    tax_code         = raw.get("tax_code", ""),
                    issue_date       = _normalize_date(raw.get("issue_date", "")),
                    pre_tax_amount   = raw.get("pre_tax_amount", 0),
                    vat_rate         = raw.get("vat_rate", 0.10),
                    vat_amount       = raw.get("vat_amount", 0),
                    total_amount     = raw.get("total_amount", 0),
                    item_description = raw.get("item_description", ""),
                    source_type      = "PDF",
                    ocr_confidence   = 0.85,
                    raw_fields       = raw,
                )
                audit = self.audit_invoice_math(inv)
                self._apply_audit(inv, audit)
                inv.suggested_accounts = suggest_accounting_accounts(inv.item_description)
                return inv
            except Exception:
                raise exc

    def _regex_extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract fields from plain text using robust multi-pass regex patterns."""
        raw: Dict[str, Any] = {}

        # 1. Invoice number
        # Pass 1: Labeled with standard invoice keywords
        m_inv = re.search(
            r"(?:số\s*(?:\(no\.?\)|\/no\.?|no\.?|hóa\s*đơn|hoa\s*don|hđ|hd|inv)?|invoice\s*no\.?|hóa\s*đơn\s*số|hoa\s*don\s*so|no\.?)[:\s#]+([A-Z0-9\-/]{3,20})",
            text, re.IGNORECASE,
        )
        if m_inv:
            cand = m_inv.group(1).strip(" .,-:#\t\n")
            # Must contain at least one digit and not be a common non-invoice word
            if any(c.isdigit() for c in cand) and cand.upper() not in {"PHA", "NGAY", "THANG", "NAM", "BAN", "MUA", "VND", "TIEN"}:
                raw["invoice_id"] = cand

        # Pass 2: Standalone 6-8 digit number (typical for Vietnamese e-invoices)
        if not raw.get("invoice_id"):
            m_stand = re.search(r"\b0*([1-9]\d{4,7}|\d{7,8})\b", text)
            if m_stand:
                raw["invoice_id"] = m_stand.group(0)

        # 2. Issue Date
        # Pass 1: Vietnamese words 'ngày ... tháng ... năm ...' with optional bilingual tags
        m_words = re.search(
            r"(?:ngày|ngay)(?:\s*\([^\)]*\))?[:\s]*(\d{1,2})\s*(?:tháng|thang)(?:\s*\([^\)]*\))?[:\s]*(\d{1,2})\s*(?:năm|nam)(?:\s*\([^\)]*\))?[:\s]*(\d{4})",
            text, re.IGNORECASE,
        )
        if m_words:
            d, mo, y = int(m_words.group(1)), int(m_words.group(2)), int(m_words.group(3))
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1990 <= y <= 2040:
                raw["issue_date"] = f"{y:04d}-{mo:02d}-{d:02d}"

        # Pass 2: Labeled date (Ngày lập, Ký ngày, Date, Ngày: ...)
        if not raw.get("issue_date"):
            m_labeled = re.search(
                r"(?:ngày\s*lập|ngay\s*lap|ngày\s*hđ|ngay\s*hd|ký\s*ngày|ky\s*ngay|ngày|ngay|date)(?:\s*\([^\)]*\))?[:\s]*(\d{1,2}[\s/\-\.]+\d{1,2}[\s/\-\.]+\d{4})",
                text, re.IGNORECASE,
            )
            if m_labeled:
                raw["issue_date"] = _normalize_date(m_labeled.group(1))

        # Pass 3: Labeled ISO format (2026-08-04)
        if not raw.get("issue_date"):
            m_iso = re.search(
                r"(?:ngày\s*lập|ngay\s*lap|ký\s*ngày|ky\s*ngay|ngày|ngay|date)(?:\s*\([^\)]*\))?[:\s]*(\d{4}[\s/\-\.]+\d{1,2}[\s/\-\.]+\d{1,2})",
                text, re.IGNORECASE,
            )
            if m_iso:
                raw["issue_date"] = _normalize_date(m_iso.group(1))

        # Pass 4: Any standalone DD/MM/YYYY date anywhere in document
        if not raw.get("issue_date"):
            all_dates = re.findall(r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b", text)
            for d_str, m_str, y_str in all_dates:
                d, mo, y = int(d_str), int(m_str), int(y_str)
                if 1 <= d <= 31 and 1 <= mo <= 12 and 2015 <= y <= 2035:
                    raw["issue_date"] = f"{y:04d}-{mo:02d}-{d:02d}"
                    break

        # 3. Tax codes (seller and buyer)
        tax_matches = re.findall(
            r"(?:mã số thuế|ma so thue|mst|tax code)[^\d\n]*[:\s]*([\d\s]{10,20})",
            text, re.IGNORECASE,
        )
        valid_taxes = []
        for tm in tax_matches:
            clean = re.sub(r"\s+", "", tm)
            if len(clean) in (10, 13, 14):
                valid_taxes.append(clean)
        # Fallback: check 10-digit numbers starting with 0
        if not valid_taxes:
            for tm in re.findall(r"\b(0\d{9})\b", text):
                if tm not in valid_taxes:
                    valid_taxes.append(tm)
        if valid_taxes:
            raw["tax_code"] = valid_taxes[0]
            if len(valid_taxes) > 1:
                raw["buyer_tax_code"] = valid_taxes[1]

        # 4. Amounts — look for VND numbers
        amounts: List[int] = []
        for m2 in re.finditer(r"(\d{1,3}(?:[.,]\d{3})+)", text):
            val = _parse_amount(m2.group(1))
            if val >= 10_000:
                amounts.append(val)

        amounts.sort(reverse=True)

        if len(amounts) >= 3:
            total = amounts[0]
            pre = amounts[1]
            vat_cand = amounts[2]
            if abs((pre + vat_cand) - total) <= 10:
                raw["total_amount"]   = total
                raw["pre_tax_amount"] = pre
                raw["vat_amount"]     = vat_cand
            else:
                raw["total_amount"]   = total
                raw["pre_tax_amount"] = pre
                raw["vat_amount"]     = total - pre
        elif len(amounts) == 2:
            raw["total_amount"]   = amounts[0]
            raw["pre_tax_amount"] = amounts[1]
            raw["vat_amount"]     = amounts[0] - amounts[1]
        elif len(amounts) == 1:
            raw["total_amount"]   = amounts[0]

        # 5. VAT rate — explicit text match + mathematical inference
        m_rate = re.search(r"(\d+)\s*%\s*(?:thuế suất|thue suat|vat|thue gtgt|thue)", text, re.IGNORECASE)
        if not m_rate:
            m_rate = re.search(r"(?:vat rate|thuế suất|thue suat|thuế|thue)[^%\n]*[:\s]*(\d+)\s*%", text, re.IGNORECASE)
        explicit_rate = float(m_rate.group(1)) / 100.0 if m_rate else None

        # Intelligent deduction from amounts: if pre_tax and vat_amount are known
        pre = raw.get("pre_tax_amount", 0)
        vat = raw.get("vat_amount", 0)
        inferred_rate = None
        if pre > 0 and vat > 0:
            for std in [0.10, 0.08, 0.05]:
                if abs(round(pre * std) - vat) <= 15:
                    inferred_rate = std
                    break

        if inferred_rate is not None:
            raw["vat_rate"] = inferred_rate
        elif explicit_rate is not None:
            raw["vat_rate"] = explicit_rate
        else:
            raw["vat_rate"] = 0.10

        # 6. Company & Seller names
        m_seller = re.search(
            r"(?:đơn vị bán hàng\s*(?:\(seller\))?|don vi ban hang|người bán hàng|nguoi ban hang|đơn vị bán|don vi ban)[:\s]+(.{4,90}?)(?:\n|mst|mã số|ma so|$)",
            text, re.IGNORECASE,
        )
        if m_seller:
            s_cand = re.sub(r"^\([^)]*\)\s*", "", m_seller.group(1)).strip(" :-\t")
            if s_cand.upper() not in BOILERPLATE_NAMES and len(s_cand) >= 3:
                raw["seller_name"] = s_cand

        # Priority 1: Match explicit company name of buyer
        m_buyer = re.search(
            r"(?:tên đơn vị\s*(?:\(company\))?|ten don vi\s*(?:\(company\))?|đơn vị mua hàng|don vi mua hang|tên đơn vị mua|ten don vi mua|tên công ty|ten cong ty)[:\s]+(.{4,90}?)(?:\n|mst|mã số|ma so|$)",
            text, re.IGNORECASE,
        )
        if not m_buyer:
            # Priority 2: Generic buyer / customer label
            m_buyer = re.search(
                r"(?:khách hàng|khach hang|người mua hàng\s*(?:\(buyer\))?|nguoi mua hang|họ tên người mua hàng|ho ten nguoi mua hang)[:\s]+(.{4,90}?)(?:\n|mst|mã số|ma so|$)",
                text, re.IGNORECASE,
            )
        if m_buyer:
            b_cand = re.sub(r"^\([^)]*\)\s*", "", m_buyer.group(1)).strip(" :-\t")
            if b_cand.upper() not in BOILERPLATE_NAMES and not any(bp in b_cand.upper() for bp in ["CCCD", "CMND", "ID CODE"]) and len(b_cand) >= 3:
                raw["customer_name"] = b_cand

        # Fallback for companies if not extracted
        if not raw.get("customer_name") or raw.get("customer_name").upper() in BOILERPLATE_NAMES:
            companies = re.findall(
                r"\b((?:CÔNG TY|CONG TY|CTY|DOANH NGHIỆP|DOANH NGHIEP|TẬP ĐOÀN|TAP DOAN)\s+[\w\s\.\,\-\&]{5,70})",
                text, re.IGNORECASE,
            )
            cleaned_companies = []
            for comp in companies:
                comp_clean = comp.strip(" .,-:\n\t")
                if len(comp_clean) >= 8 and comp_clean.upper() not in BOILERPLATE_NAMES:
                    cleaned_companies.append(comp_clean)

            if len(cleaned_companies) >= 2:
                if not raw.get("seller_name"):
                    raw["seller_name"] = cleaned_companies[0]
                raw["customer_name"] = cleaned_companies[1]
            elif len(cleaned_companies) == 1:
                if raw.get("seller_name") and raw["seller_name"] != cleaned_companies[0]:
                    raw["customer_name"] = cleaned_companies[0]
                elif not raw.get("customer_name"):
                    raw["customer_name"] = cleaned_companies[0]

        if not raw.get("customer_name") or raw.get("customer_name").upper() in BOILERPLATE_NAMES:
            raw["customer_name"] = "Khách hàng doanh nghiệp"

        # 7. Item description
        m_desc = re.search(
            r"(?:tên hàng hóa[,\s]*dịch vụ|ten hang hoa[,\s]*dich vu|hàng hóa dịch vụ|hang hoa dich vu|tên hàng hóa|ten hang hoa|dịch vụ|dich vu)[:\s]+(.{3,120}?)(?:\n|số lượng|so luong|đvt|dvt|$)",
            text, re.IGNORECASE,
        )
        if m_desc:
            raw["item_description"] = m_desc.group(1).strip()
        else:
            raw["item_description"] = "Hàng hóa dịch vụ"

        return raw

    def _enrich_from_tables(
        self, raw: Dict[str, Any], tables: List[List]
    ) -> None:
        """Scan extracted table cells to refine pre_tax / vat / total."""
        keywords_total    = {"tong cong", "total", "tong tien thanh toan", "thanh toan"}
        keywords_pretax   = {"chua thue", "truoc thue", "subtotal", "tong tien hang"}
        keywords_vat      = {"thue gtgt", "vat", "tien thue"}

        for row in tables:
            if not row:
                continue
            label = str(row[0] or "").lower().strip()
            # Find numeric cell in row
            for cell in row[1:]:
                val = _parse_amount(str(cell or ""))
                if val < 100_000:
                    continue
                if any(k in label for k in keywords_total):
                    raw["total_amount"] = val
                elif any(k in label for k in keywords_pretax):
                    raw["pre_tax_amount"] = val
                elif any(k in label for k in keywords_vat):
                    raw["vat_amount"] = val

    # -----------------------------------------------------------------------
    # 3. IMAGE / OCR PARSER
    # -----------------------------------------------------------------------

    def parse_image_invoice(
        self, source: Union[str, bytes, Path]
    ) -> InvoiceData:
        """
        Extract invoice from image using PaddleOCR + VietOCR pipeline.
        Falls back to Ollama VLM (qwen2.5-vl:7b) if vision models unavailable.
        Falls back to simulation mode if neither is available.
        """
        # Load image bytes
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                img_bytes = f.read()
        else:
            img_bytes = source

        ocr_text = ""
        confidence = 0.80

        # --- Attempt 1: PaddleOCR ---
        if _HAS_PADDLE and _HAS_PIL:
            try:
                ocr_text, confidence = self._run_paddle_ocr(img_bytes)
                logger.info(f"[OCR] PaddleOCR success, conf={confidence:.2f}")
            except Exception as e:
                logger.warning(f"[OCR] PaddleOCR failed: {e}")

        # --- Attempt 2: VietOCR refinement ---
        if not ocr_text and _HAS_VIETOCR and _HAS_PIL:
            try:
                ocr_text, confidence = self._run_vietocr(img_bytes)
                logger.info(f"[OCR] VietOCR success, conf={confidence:.2f}")
            except Exception as e:
                logger.warning(f"[OCR] VietOCR failed: {e}")

        # --- Attempt 3: Ollama VLM ---
        if not ocr_text and _HAS_OLLAMA:
            try:
                ocr_text, confidence = self._run_vlm_ocr(img_bytes)
                logger.info(f"[OCR] Ollama VLM success, conf={confidence:.2f}")
            except Exception as e:
                logger.warning(f"[OCR] Ollama VLM failed: {e}")

        # --- Fallback: simulation ---
        if not ocr_text:
            logger.warning("[OCR] All OCR engines unavailable — using simulated OCR text.")
            ocr_text = self._simulate_ocr_output()
            confidence = 0.60

        raw = self._regex_extract_from_text(ocr_text)
        inv = InvoiceData(
            invoice_id       = raw.get("invoice_id", ""),
            customer_name    = raw.get("customer_name", ""),
            buyer_tax_code   = raw.get("buyer_tax_code", ""),
            seller_name      = raw.get("seller_name", ""),
            tax_code         = raw.get("tax_code", ""),
            issue_date       = _normalize_date(raw.get("issue_date", "")),
            pre_tax_amount   = raw.get("pre_tax_amount", 0),
            vat_rate         = raw.get("vat_rate", 0.10),
            vat_amount       = raw.get("vat_amount", 0),
            total_amount     = raw.get("total_amount", 0),
            item_description = raw.get("item_description", ""),
            source_type      = "IMAGE_OCR",
            ocr_confidence   = confidence,
            raw_fields       = {"ocr_full_text": ocr_text, **raw},
        )

        audit = self.audit_invoice_math(inv)
        self._apply_audit(inv, audit)
        inv.suggested_accounts = suggest_accounting_accounts(inv.item_description)

        logger.info(
            f"[OCR] Image parsed: {inv.invoice_id} | total={inv.total_amount:,} | "
            f"conf={inv.ocr_confidence:.2f} | audit={'OK' if inv.ocr_audit_passed else 'FAIL'}"
        )
        return inv

    def _run_paddle_ocr(self, img_bytes: bytes) -> Tuple[str, float]:
        """Run PaddleOCR on image bytes, return (text, confidence)."""
        if self._paddle_ocr is None:
            self._paddle_ocr = _PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)

        img = _PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            result = self._paddle_ocr.ocr(tmp.name, cls=True)
        os.unlink(tmp.name)

        lines, confs = [], []
        for page in result:
            if page:
                for line in page:
                    text_info = line[1]
                    lines.append(text_info[0])
                    confs.append(float(text_info[1]))

        combined_text = "\n".join(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return combined_text, avg_conf

    def _run_vietocr(self, img_bytes: bytes) -> Tuple[str, float]:
        """
        Segment invoice image into text lines/cells using OpenCV and recognize
        each line with VietOCR transformer checkpoint.
        """
        if self._vietocr is None:
            base_yml = _ROOT / "weights" / "ocr" / "base.yml"
            vgg_yml = _ROOT / "weights" / "ocr" / "vgg_transformer.yml"
            local_weights = _ROOT / "weights" / "ocr" / "vgg_transformer.pth"

            if base_yml.exists() and vgg_yml.exists():
                import yaml
                with open(base_yml, encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f)
                with open(vgg_yml, encoding="utf-8") as f:
                    cfg_dict.update(yaml.safe_load(f))
                cfg = _VietOCRCfg(cfg_dict)
            else:
                cfg = _VietOCRCfg.load_config_from_name("vgg_transformer")

            cfg["cnn"]["pretrained"] = False
            if local_weights.exists():
                cfg["weights"] = str(local_weights)
            cfg["device"] = "cpu"
            self._vietocr = _VietOCRPredictor(cfg)

        import cv2
        import numpy as np

        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil_img = _PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Binarize with threshold
        _, bin_inv = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)

        # 2. Filter out horizontal & vertical table lines
        h_line = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)))
        v_line = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
        lines_mask = cv2.add(h_line, v_line)
        text_mask = cv2.subtract(bin_inv, lines_mask)

        # 3. Suppress outer edge decoration
        margin = 16
        if h > margin * 2 and w > margin * 2:
            text_mask[:margin, :] = 0
            text_mask[-margin:, :] = 0
            text_mask[:, :margin] = 0
            text_mask[:, -margin:] = 0

        # 4. Connect characters horizontally into text lines/word blocks
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 2))
        connected = cv2.dilate(text_mask, kernel_connect, iterations=1)

        # 5. Extract bounding boxes
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw >= 14 and 7 <= bh <= 60 and bw < w * 0.96:
                boxes.append((x, y, bw, bh))

        # Sort boxes top to bottom, then left to right
        boxes = sorted(boxes, key=lambda b: (b[1] // 12, b[0]))

        recognized_lines: List[str] = []
        if len(boxes) >= 2:
            for x, y, bw, bh in boxes:
                pad_y, pad_x = 3, 4
                crop = img_bgr[max(0, y - pad_y):min(h, y + bh + pad_y), max(0, x - pad_x):min(w, x + bw + pad_x)]
                crop_pil = _PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                try:
                    txt = self._vietocr.predict(crop_pil).strip()
                    if txt:
                        recognized_lines.append(txt)
                except Exception:
                    pass

        if not recognized_lines:
            # Fallback on full image
            full_pil = _PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            txt = self._vietocr.predict(full_pil).strip()
            if txt:
                recognized_lines.append(txt)

        combined_text = "\n".join(recognized_lines)
        return combined_text, 0.88

    def _run_vlm_ocr(self, img_bytes: bytes) -> Tuple[str, float]:
        """Call Ollama VLM (qwen2.5-vl) to transcribe invoice image."""
        b64 = base64.b64encode(img_bytes).decode()
        prompt = (
            "Day la anh hoa don ban hang cua Viet Nam. "
            "Hay doc va tra ve chinh xac cac truong sau: "
            "So hoa don, ten nguoi mua, ma so thue, ngay lap, "
            "tong tien chua thue, tien thue GTGT, tong tien thanh toan. "
            "Tra ve dang JSON thuan tuy."
        )
        client = _ollama.Client(host=self.ollama_base_url)
        resp = client.chat(
            model=self.vlm_model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [b64],
            }],
        )
        content = resp["message"]["content"]

        # Try to parse JSON response
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                raw = {
                    "invoice_id":    data.get("so_hoa_don", data.get("invoice_id", "")),
                    "customer_name": data.get("ten_nguoi_mua", data.get("customer_name", "")),
                    "tax_code":      data.get("ma_so_thue", data.get("tax_code", "")),
                    "issue_date":    data.get("ngay_lap", data.get("issue_date", "")),
                    "pre_tax_amount":_parse_amount(str(data.get("tong_tien_chua_thue", "0"))),
                    "vat_amount":    _parse_amount(str(data.get("tien_thue", "0"))),
                    "total_amount":  _parse_amount(str(data.get("tong_tien", "0"))),
                }
                # Build text from raw for downstream parsing
                ocr_text = json.dumps(raw, ensure_ascii=False)
                return ocr_text, 0.75
        except Exception:
            pass
        return content, 0.65

    def _simulate_ocr_output(self) -> str:
        """Simulated OCR text for demo/testing when no OCR engine available."""
        return (
            "HOA DON GIA TRI GIA TANG\n"
            "So hoa don: 0000123\n"
            "Ngay: 20/10/2024\n"
            "Ten don vi mua hang: Cong ty TNHH Thuong Mai Minh Phat\n"
            "Ma so thue: 0100123456\n"
            "Ten hang hoa dich vu: Hang hoa thuong mai\n"
            "Tong tien hang (chua thue): 20,000,000\n"
            "Thue GTGT (10%): 2,000,000\n"
            "Tong tien thanh toan: 22,000,000\n"
        )

    # -----------------------------------------------------------------------
    # 4. MATH AUDITING
    # -----------------------------------------------------------------------

    def audit_invoice_math(self, inv: InvoiceData) -> AuditResult:
        """
        Verify the two fundamental accounting equations:
          1. vat_amount = round(pre_tax_amount * vat_rate)   ± 10 VND
          2. total_amount = pre_tax_amount + vat_amount       ± 10 VND

        If either fails, trigger self_heal_ocr_discrepancy.
        """
        TOLERANCE = 10  # VND

        expected_vat   = round(inv.pre_tax_amount * inv.vat_rate)
        expected_total = inv.pre_tax_amount + inv.vat_amount

        vat_ok   = abs(expected_vat   - inv.vat_amount)   <= TOLERANCE
        total_ok = abs(expected_total - inv.total_amount) <= TOLERANCE

        passed = vat_ok and total_ok

        note = ""
        suggested_total = None
        corrected_field = ""

        # Self-heal vat_rate if total balances and a standard VAT rate matches vat_amount
        if not vat_ok and total_ok and inv.pre_tax_amount > 0:
            for rate in KNOWN_VAT_RATES:
                if abs(round(inv.pre_tax_amount * rate) - inv.vat_amount) <= TOLERANCE:
                    old_rate = inv.vat_rate
                    inv.vat_rate = rate
                    expected_vat = round(inv.pre_tax_amount * rate)
                    vat_ok = True
                    passed = True
                    note = (
                        f"[Self-Healed] Đã tự động hiệu chỉnh thuế suất VAT ({old_rate:.0%} → {rate:.0%}) "
                        f"do tiền VAT ({inv.vat_amount:,}đ) và tổng tiền ({inv.total_amount:,}đ) hoàn toàn cân đối theo phương trình kế toán."
                    )
                    break

        if not passed:
            heal = self.self_heal_ocr_discrepancy(inv)
            suggested_total = heal.get("suggested_corrected_total")
            corrected_field = heal.get("corrected_field", "")
            note = heal.get("note", "")

        return AuditResult(
            passed            = passed,
            vat_check_ok      = vat_ok,
            total_check_ok    = total_ok,
            expected_vat      = expected_vat,
            expected_total    = expected_total,
            actual_vat        = inv.vat_amount,
            actual_total      = inv.total_amount,
            note              = note,
            suggested_corrected_total = suggested_total,
            corrected_field   = corrected_field,
        )

    def self_heal_ocr_discrepancy(self, inv: InvoiceData) -> Dict[str, Any]:
        """
        Attempt to identify which field (pre_tax, vat, or total) is likely
        misread by OCR, using:
          - Standard VAT rate consistency check (8% / 10%)
          - Levenshtein digit confusion on numeric strings

        Returns dict with keys: suggested_corrected_total, corrected_field, note.
        """
        result: Dict[str, Any] = {
            "suggested_corrected_total": None,
            "corrected_field": "",
            "note": "",
            "candidates": [],
        }

        pre   = inv.pre_tax_amount
        vat   = inv.vat_amount
        total = inv.total_amount

        # --- Strategy 1: Check if total is consistent with known vat rates ---
        for rate in KNOWN_VAT_RATES:
            expected_vat   = round(pre * rate)
            expected_total = pre + expected_vat
            if abs(total - expected_total) <= 10:
                result["note"] = (
                    f"[Heal-1] pre_tax={pre:,} * rate={rate:.0%} -> "
                    f"vat={expected_vat:,} total={expected_total:,} khop voi total_amount. "
                    f"vat_rate co the dang sai ({inv.vat_rate:.0%} -> {rate:.0%})."
                )
                result["suggested_corrected_total"] = expected_total
                result["corrected_field"] = "vat_rate"
                return result

        # --- Strategy 2: Maybe vat_amount is misread; correct from pre * rate ---
        expected_vat_at_rate = round(pre * inv.vat_rate)
        corrected_total_from_vat = pre + expected_vat_at_rate
        if abs(corrected_total_from_vat - total) > 10:
            result["candidates"].append({
                "field": "vat_amount",
                "original": vat,
                "corrected": expected_vat_at_rate,
                "new_total": corrected_total_from_vat,
                "score": _levenshtein(str(vat), str(expected_vat_at_rate)),
            })

        # --- Strategy 3: Maybe total is misread — generate digit confusions ---
        total_candidates = _candidate_corrections(total)
        for cand_total in total_candidates:
            # Check if cand_total = pre + vat
            if abs(cand_total - (pre + vat)) <= 10:
                lev = _levenshtein(str(total), str(cand_total))
                result["candidates"].append({
                    "field": "total_amount",
                    "original": total,
                    "corrected": cand_total,
                    "new_total": cand_total,
                    "score": lev,
                })

        # --- Strategy 4: Maybe pre_tax is misread ---
        pre_candidates = _candidate_corrections(pre)
        for cand_pre in pre_candidates:
            expected_vat2 = round(cand_pre * inv.vat_rate)
            expected_total2 = cand_pre + expected_vat2
            if abs(expected_total2 - total) <= 10 or abs(cand_pre + vat - total) <= 10:
                lev = _levenshtein(str(pre), str(cand_pre))
                result["candidates"].append({
                    "field": "pre_tax_amount",
                    "original": pre,
                    "corrected": cand_pre,
                    "new_total": expected_total2,
                    "score": lev,
                })

        # Pick best candidate (lowest Levenshtein distance)
        if result["candidates"]:
            best = min(result["candidates"], key=lambda x: x["score"])
            result["suggested_corrected_total"] = best["new_total"]
            result["corrected_field"] = best["field"]
            result["note"] = (
                f"[Heal-{3 if best['field']=='total_amount' else 4}] "
                f"OCR co the doc nham truong '{best['field']}': "
                f"{best['original']:,} -> {best['corrected']:,} "
                f"(Levenshtein={best['score']}). "
                f"Tong tien de xuat chinh xac: {best['new_total']:,} VND."
            )
        else:
            result["note"] = (
                f"[Heal-FAIL] Khong tim duoc cach chinh sua hop ly. "
                f"pre={pre:,} vat={vat:,} total={total:,} vat_rate={inv.vat_rate:.0%}. "
                f"Can kiem tra thu cong anh hoa don goc."
            )

        return result

    def _apply_audit(self, inv: InvoiceData, audit: AuditResult) -> None:
        """Write audit result back into InvoiceData."""
        inv.ocr_audit_passed = audit.passed
        inv.ocr_suspect_note = audit.note
        if not audit.passed:
            inv.ocr_confidence = min(inv.ocr_confidence, 0.55)
            inv.suggested_corrected_total = audit.suggested_corrected_total
            inv.raw_ocr_true_amount = audit.suggested_corrected_total
            logger.warning(f"[OCR] Audit FAILED for {inv.invoice_id}: {audit.note}")
        else:
            if audit.note:
                logger.info(f"[OCR] Audit PASSED with self-healing for {inv.invoice_id}: {audit.note}")

    # -----------------------------------------------------------------------
    # 5. SAMPLE FILE GENERATOR (for Streamlit demo)
    # -----------------------------------------------------------------------

    def generate_sample_invoice_files(
        self, output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """
        Generate sample invoice files (XML, PDF-like text, PNG placeholder)
        for Streamlit demo testing.

        Returns:
            Dict with keys "xml_path", "pdf_path", "image_path"
            pointing to generated temp files.
        """
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "agentic_invoices"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        # -- Sample XML --
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HDon>
  <DLHDon>
    <TTChung>
      <SHDon>00101</SHDon>
      <KHHDon>AA/24E</KHHDon>
      <NLap>2024-09-05</NLap>
    </TTChung>
    <NDHDon>
      <NBan>
        <Ten>Cong ty TNHH Phan Mem Ke Toan Demo</Ten>
        <MST>0123456789</MST>
      </NBan>
      <NMua>
        <Ten>Cong ty TNHH Thuong Mai Minh Phat</Ten>
        <MSTNMua>0100123456</MSTNMua>
        <DChi>123 Nguyen Hue, Q1, TP.HCM</DChi>
      </NMua>
      <DSHHDVu>
        <HHDVu>
          <TenHHDV>Hang hoa thuong mai</TenHHDV>
          <DVTinh>Cai</DVTinh>
          <SLuong>10</SLuong>
          <DGia>2000000</DGia>
          <ThTien>20000000</ThTien>
        </HHDVu>
      </DSHHDVu>
      <TToan>
        <TgTCThue>20000000</TgTCThue>
        <DSLThue>
          <LThue>
            <TSuat>10%</TSuat>
            <TgTThue>2000000</TgTThue>
          </LThue>
        </DSLThue>
        <TgTTtbSo>22000000</TgTTtbSo>
      </TToan>
    </NDHDon>
  </DLHDon>
</HDon>"""
        xml_path = output_dir / "sample_invoice_HD00101.xml"
        xml_path.write_text(xml_content, encoding="utf-8")
        paths["xml_path"] = str(xml_path)

        # -- Sample PDF-like text (actual PDF needs reportlab; use .txt as placeholder) --
        pdf_text = (
            "HOA DON GIA TRI GIA TANG\n"
            "So hoa don: 00123\n"
            "Ngay lap: 10/10/2024\n"
            "Ten don vi mua hang: Cong ty Co Phan Xay Dung An Binh\n"
            "Ma so thue: 0200234567\n"
            "Ten hang hoa dich vu: Vat lieu xay dung\n"
            "Tong tien hang (chua thue): 30,000,000\n"
            "Thue suat GTGT: 10%\n"
            "Tien thue GTGT: 3,000,000\n"
            "Tong tien thanh toan: 33,000,000\n"
        )
        pdf_path = output_dir / "sample_invoice_HD00102.txt"
        pdf_path.write_text(pdf_text, encoding="utf-8")
        paths["pdf_path"] = str(pdf_path)

        # -- Sample PNG (simple text image using PIL if available) --
        if _HAS_PIL:
            img = _PILImage.new("RGB", (600, 400), color=(255, 255, 255))
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                lines = [
                    "HOA DON GTGT - SO: 00123",
                    "Ngay: 20/10/2024",
                    "Nguoi mua: CTY TNHH DUOC PHAM HOANG GIA",
                    "MST: 0300345678",
                    "Tong tien chua thue: 18,500,000",   # Note: intentional OCR blur test
                    "Tien thue (10%):       1,850,000",
                    "TONG TIEN THANH TOAN: 10,500,000",  # Intentionally "wrong" (blurred 8->0)
                ]
                y = 20
                for line in lines:
                    draw.text((20, y), line, fill=(0, 0, 0))
                    y += 40
            except Exception:
                pass  # draw without custom font

            img_path = output_dir / "sample_invoice_HD00123_ocr_test.png"
            img.save(str(img_path))
            paths["image_path"] = str(img_path)
        else:
            paths["image_path"] = ""

        logger.info(f"[OCR] Sample files generated: {paths}")
        return paths

    # -----------------------------------------------------------------------
    # 6. Convenience: parse auto-detect format
    # -----------------------------------------------------------------------

    def parse_auto(
        self,
        source: Union[str, bytes, Path],
        file_ext: str = "",
    ) -> InvoiceData:
        """
        Auto-detect format from file extension or byte signature
        and dispatch to appropriate parser.
        """
        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
        else:
            # Sniff bytes
            sig = source[:8] if source else b""
            if sig.startswith(b"<?xml") or sig.startswith(b"<HDon") or b"<SHDon" in source[:200]:
                ext = ".xml"
            elif sig.startswith(b"%PDF"):
                ext = ".pdf"
            else:
                ext = ".png"

        ext = file_ext.lower() or ext

        if ext == ".xml":
            return self.parse_xml_invoice(source)
        elif ext == ".pdf":
            return self.parse_pdf_invoice(source)
        else:
            return self.parse_image_invoice(source)
