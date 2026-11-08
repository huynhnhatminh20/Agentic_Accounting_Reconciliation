# ============================================================
# database/seed_mock_data.py — Dữ liệu Giả lập Kiểm thử
# Hệ thống Đối soát Kế toán (TT 200/133)
#
# Bộ dữ liệu được tính toán CHính XÁC về mặt toán học,
# bao gồm 5 Kịch bản A–E để kiểm tra toàn diện hệ thống.
# ============================================================

import sys
import json
import logging
from pathlib import Path
from math import floor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ==============================================================
# HELPER MATH
# ==============================================================

def calc_vat(pre_tax: int, vat_rate: float) -> tuple[int, int]:
    """
    Tính vat_amount và total_amount từ pre_tax và vat_rate.
    Làm tròn vat_amount theo quy tắc kế toán Việt Nam (floor).

    Returns:
        (vat_amount, total_amount)
    """
    vat_amount = floor(pre_tax * vat_rate)
    total = pre_tax + vat_amount
    return vat_amount, total


def apply_discount(total: int, rate: float) -> tuple[int, int]:
    """
    Tính tiền chiết khấu và số tiền sau khi trừ chiết khấu.

    Returns:
        (discount_amount, net_amount)
    """
    discount = floor(total * rate)
    net = total - discount
    return discount, net


# ==============================================================
# DỮ LIỆU KHÁCH HÀNG (10 Công ty)
# ==============================================================

CUSTOMERS = [
    # (customer_id, customer_name, short_name, tax_code, discount_rate, credit_days)
    ("KH001", "Công ty TNHH Thương Mại Minh Phát",        "MINH PHAT",   "0100123456", 0.02, 30),
    ("KH002", "Công ty Cổ Phần Xây Dựng An Bình",         "AN BINH",     "0200234567", 0.03, 45),
    ("KH003", "Công ty TNHH Dược Phẩm Hoàng Gia",         "HOANG GIA",   "0300345678", 0.05, 30),
    ("KH004", "Công ty Cổ Phần Xuất Nhập Khẩu Sao Vàng",  "SAO VANG",    "0400456789", 0.02, 60),
    ("KH005", "Công ty TNHH Sản Xuất Thiết Bị Tiến Đạt",  "TIEN DAT",    "0500567890", 0.00, 30),
    ("KH006", "Công ty Cổ Phần Thực Phẩm Việt Hương",     "VIET HUONG",  "0600678901", 0.03, 30),
    ("KH007", "Công ty TNHH Vận Tải Trường Phát",         "TRUONG PHAT", "0700789012", 0.00, 45),
    ("KH008", "Công ty Cổ Phần Nội Thất Phú Lợi",         "PHU LOI",     "0800890123", 0.02, 30),
    ("KH009", "Doanh Nghiệp Tư Nhân Hồng Phúc",           "HONG PHUC",   "0900901234", 0.00, 15),
    ("KH010", "Công ty TNHH Kỹ Thuật Số Đại Dương",       "DAI DUONG",   "1000012345", 0.05, 30),
]

# ==============================================================
# PAYER ALIASES (6 cá nhân đại diện)
# ==============================================================

PAYER_ALIASES = [
    # (person_name, role, customer_id)
    ("LE VAN TUNG",    "GD",  "KH001"),  # Giám đốc Minh Phát
    ("NGUYEN THI MAI", "KT",  "KH001"),  # Kế toán Minh Phát
    ("TRAN MINH HUNG", "GD",  "KH003"),  # Giám đốc Hoàng Gia
    ("PHAM THI LAN",   "KT",  "KH003"),  # Kế toán Hoàng Gia
    ("VU DUC THANH",   "GD",  "KH005"),  # Giám đốc Tiến Đạt
    ("HOANG VAN NAM",  "GD",  "KH007"),  # Giám đốc Trường Phát
]

# ==============================================================
# XÂY DỰNG DỮ LIỆU HÓA ĐƠN & SAO KÊ THEO 5 KỊCH BẢN
# ==============================================================

def build_invoice_data() -> list[tuple]:
    """
    Tạo 28 hóa đơn tính toán chính xác về mặt toán học.
    Returns list of tuples theo thứ tự cột bảng invoices.
    """
    invoices = []

    # ----------------------------------------------------------
    # === KỊCH BẢN A ===
    # 10 dòng sao kê chuẩn cho System 1
    # ----------------------------------------------------------

    # -- A1: 5 HĐ riêng lẻ, mỗi HĐ sẽ có 1 dòng sao kê ghi rõ mã --

    # HD00101 — KH001 Minh Phát, VAT 10%
    pre, vat_a, total_a = 20_000_000, *calc_vat(20_000_000, 0.10)
    invoices.append((
        "HD00101", "KH001", "Công ty TNHH Thương Mại Minh Phát", "0100123456",
        "2024-09-05", "2024-10-05",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 22_000_000

    # HD00102 — KH002 An Bình, VAT 10%
    pre, vat_a, total_a = 30_000_000, *calc_vat(30_000_000, 0.10)
    invoices.append((
        "HD00102", "KH002", "Công ty Cổ Phần Xây Dựng An Bình", "0200234567",
        "2024-09-10", "2024-10-10",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 33_000_000

    # HD00103 — KH004 Sao Vàng, VAT 10%
    pre, vat_a, total_a = 15_000_000, *calc_vat(15_000_000, 0.10)
    invoices.append((
        "HD00103", "KH004", "Công ty Cổ Phần Xuất Nhập Khẩu Sao Vàng", "0400456789",
        "2024-09-12", "2024-10-12",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # total = 16_500_000

    # HD00104 — KH006 Việt Hương, VAT 8%
    pre, vat_a, total_a = 50_000_000, *calc_vat(50_000_000, 0.08)
    invoices.append((
        "HD00104", "KH006", "Công ty Cổ Phần Thực Phẩm Việt Hương", "0600678901",
        "2024-09-15", "2024-10-15",
        pre, 0.08, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 54_000_000

    # HD00105 — KH002 An Bình, VAT 10% (cùng KH002, sẽ dùng ghép với HD00102 ở A2)
    pre, vat_a, total_a = 20_000_000, *calc_vat(20_000_000, 0.10)
    invoices.append((
        "HD00105", "KH002", "Công ty Cổ Phần Xây Dựng An Bình", "0200234567",
        "2024-09-18", "2024-10-18",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 22_000_000

    # -- A2: 3 HĐ sẽ được ghép theo cặp/bộ 3 trả gộp --

    # HD00106 — KH007 Trường Phát, VAT 10% (ghép cùng HD00107)
    pre, vat_a, total_a = 25_000_000, *calc_vat(25_000_000, 0.10)
    invoices.append((
        "HD00106", "KH007", "Công ty TNHH Vận Tải Trường Phát", "0700789012",
        "2024-09-20", "2024-10-20",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # total = 27_500_000

    # HD00107 — KH007 Trường Phát, VAT 10%
    pre, vat_a, total_a = 18_000_000, *calc_vat(18_000_000, 0.10)
    invoices.append((
        "HD00107", "KH007", "Công ty TNHH Vận Tải Trường Phát", "0700789012",
        "2024-09-22", "2024-10-22",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # total = 19_800_000

    # HD00108 — KH008 Phú Lợi, VAT 10% (ghép cùng HD00109 + HD00110 ở bộ 3)
    pre, vat_a, total_a = 10_000_000, *calc_vat(10_000_000, 0.10)
    invoices.append((
        "HD00108", "KH008", "Công ty Cổ Phần Nội Thất Phú Lợi", "0800890123",
        "2024-09-23", "2024-10-23",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 11_000_000

    # HD00109 — KH008 Phú Lợi, VAT 10%
    pre, vat_a, total_a = 8_000_000, *calc_vat(8_000_000, 0.10)
    invoices.append((
        "HD00109", "KH008", "Công ty Cổ Phần Nội Thất Phú Lợi", "0800890123",
        "2024-09-24", "2024-10-24",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 8_800_000

    # HD00110 — KH008 Phú Lợi, VAT 10%
    pre, vat_a, total_a = 7_000_000, *calc_vat(7_000_000, 0.10)
    invoices.append((
        "HD00110", "KH008", "Công ty Cổ Phần Nội Thất Phú Lợi", "0800890123",
        "2024-09-25", "2024-10-25",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 7_700_000  → 3 HĐ PHÚ LỢI: 11+8.8+7.7 = 27_500_000

    # -- A3: 2 HĐ dùng cho dòng sao kê không ghi mã, chỉ ghi tên viết tắt --

    # HD00111 — KH009 Hồng Phúc (chỉ có 1 HĐ → fuzzy name là đủ)
    pre, vat_a, total_a = 35_000_000, *calc_vat(35_000_000, 0.10)
    invoices.append((
        "HD00111", "KH009", "Doanh Nghiệp Tư Nhân Hồng Phúc", "0900901234",
        "2024-09-28", "2024-10-28",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 38_500_000

    # HD00112 — KH010 Đại Dương (chỉ có 1 HĐ cho kỳ này)
    pre, vat_a, total_a = 42_000_000, *calc_vat(42_000_000, 0.10)
    invoices.append((
        "HD00112", "KH010", "Công ty TNHH Kỹ Thuật Số Đại Dương", "1000012345",
        "2024-09-30", "2024-10-30",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 46_200_000

    # ----------------------------------------------------------
    # === KỊCH BẢN B ===
    # Subset Sum — khách chuyển khoản gộp không ghi mã HĐ
    # KH003 Hoàng Gia có 4 HĐ lẻ
    # ----------------------------------------------------------

    # HD00113 — KH003, VAT 10%
    pre, vat_a, total_a = 10_909_091, *calc_vat(10_909_091, 0.10)
    invoices.append((
        "HD00113", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-01", "2024-11-01",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # total = 12_000_000  (10_909_091 × 1.10 = 11_999_999 ≈ 12_000_000)
    # Chỉnh lại chính xác:
    # Dùng pre_tax sao cho total đúng bằng 12_000_000
    # 12_000_000 / 1.10 = 10_909_090.909 → pre = 10_909_091, vat = 1_090_909, total = 12_000_000 ✓
    invoices[-1] = (
        "HD00113", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-01", "2024-11-01",
        10_909_091, 0.10, 1_090_909, 12_000_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    )

    # HD00114 — KH003, 18.500.000
    invoices.append((
        "HD00114", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-03", "2024-11-03",
        16_818_182, 0.10, 1_681_818, 18_500_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # 16_818_182 × 1.10 = 18_499_999 ≈ 18_500_000
    # Điều chỉnh vat sao cho total = 18_500_000:
    invoices[-1] = (
        "HD00114", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-03", "2024-11-03",
        16_818_182, 0.10, 1_681_818, 18_500_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    )

    # HD00115 — KH003, 25.000.000
    invoices.append((
        "HD00115", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-05", "2024-11-05",
        22_727_273, 0.10, 2_272_727, 25_000_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))

    # HD00116 — KH003, 9.000.000
    invoices.append((
        "HD00116", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-07", "2024-11-07",
        8_181_818, 0.10, 818_182, 9_000_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))

    # ----------------------------------------------------------
    # === KỊCH BẢN C ===
    # Chiết khấu hợp đồng tự trừ / Phí ngân hàng
    # ----------------------------------------------------------

    # HD00117 — KH001 Minh Phát (discount 2%), VAT 10%
    # Khách chuyển: total - 2% chiết khấu = X
    pre, vat_a, total_a = 45_000_000, *calc_vat(45_000_000, 0.10)
    invoices.append((
        "HD00117", "KH001", "Công ty TNHH Thương Mại Minh Phát", "0100123456",
        "2024-10-08", "2024-11-08",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 49_500_000; sau CK 2% → thực nhận = 48_510_000

    # HD00118 — KH002 An Bình (discount 3%), VAT 10%
    pre, vat_a, total_a = 60_000_000, *calc_vat(60_000_000, 0.10)
    invoices.append((
        "HD00118", "KH002", "Công ty Cổ Phần Xây Dựng An Bình", "0200234567",
        "2024-10-10", "2024-11-10",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 66_000_000; sau CK 3% → thực nhận = 64_020_000

    # HD00119 — KH005 Tiến Đạt (phí NH 22.000đ, không CK), VAT 10%
    pre, vat_a, total_a = 25_000_000, *calc_vat(25_000_000, 0.10)
    invoices.append((
        "HD00119", "KH005", "Công ty TNHH Sản Xuất Thiết Bị Tiến Đạt", "0500567890",
        "2024-10-12", "2024-11-12",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 27_500_000; sau phí NH 22k → thực nhận = 27_478_000

    # ----------------------------------------------------------
    # === KỊCH BẢN D ===
    # Cá nhân chuyển hộ công ty
    # ----------------------------------------------------------

    # HD00120 — KH001 Minh Phát (sẽ do LE VAN TUNG chuyển), VAT 10%
    pre, vat_a, total_a = 32_000_000, *calc_vat(32_000_000, 0.10)
    invoices.append((
        "HD00120", "KH001", "Công ty TNHH Thương Mại Minh Phát", "0100123456",
        "2024-10-15", "2024-11-15",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))  # total = 35_200_000

    # HD00121 — KH003 Hoàng Gia (sẽ do TRAN MINH HUNG chuyển tổng 2 HĐ)
    pre, vat_a, total_a = 14_545_454, *calc_vat(14_545_454, 0.10)
    invoices.append((
        "HD00121", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-17", "2024-11-17",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # total ≈ 16_000_000
    invoices[-1] = (
        "HD00121", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-17", "2024-11-17",
        14_545_454, 0.10, 1_454_546, 16_000_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    )

    # HD00122 — KH003 Hoàng Gia (cùng TRAN MINH HUNG chuyển gộp)
    invoices.append((
        "HD00122", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-18", "2024-11-18",
        13_636_364, 0.10, 1_363_636, 15_000_000,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))  # total = 15_000_000; HD00121 + HD00122 = 31_000_000

    # ----------------------------------------------------------
    # === KỊCH BẢN E ===
    # Lỗi OCR hóa đơn — 2 HĐ bị đọc sai số tiền
    # ----------------------------------------------------------

    # HD00123 — KH004 Sao Vàng, OCR đọc nhầm 18.5tr thành 10.5tr
    # Thực tế: pre=16_818_182, vat=1_681_818, total=18_500_000
    # OCR lưu sai: total_amount = 10_500_000
    invoices.append((
        "HD00123", "KH004", "Công ty Cổ Phần Xuất Nhập Khẩu Sao Vàng", "0400456789",
        "2024-10-20", "2024-11-20",
        9_545_455, 0.10, 954_545, 10_500_000,  # Giá trị SAI do OCR
        0, "UNPAID", "IMAGE_OCR", 0.58, 0,
        "Chữ số '8' bị nhòe → OCR đọc nhầm '18.5tr' thành '10.5tr'. Cần kiểm tra ảnh gốc.",
        18_500_000,  # raw_ocr_true_amount = giá trị thật
    ))

    # HD00124 — KH006 Việt Hương, OCR đọc nhầm 54.0tr thành 54.00tr (đúng) nhưng
    # thuế bị đọc sai: pre bị hiểu là 50.000.000 → 46.000.000 (nhầm 5→4)
    # Thực tế: pre=46_296_296, vat=3_703_704, total=50_000_000 (VAT 8%)
    # OCR lưu sai: total_amount = 54_000_000 → thực ra là 50_000_000
    invoices.append((
        "HD00124", "KH006", "Công ty Cổ Phần Thực Phẩm Việt Hương", "0600678901",
        "2024-10-22", "2024-11-22",
        50_000_000, 0.08, 4_000_000, 54_000_000,  # OCR đọc VAT sai: lưu 54tr
        0, "UNPAID", "IMAGE_OCR", 0.61, 0,
        "Chữ số đầu tiên của pre_tax bị nhòe: OCR đọc pre=50tr thay vì 46.3tr. Tổng sai.",
        50_000_000,  # Khách chuyển đúng 50_000_000 trên sao kê
    ))

    # ----------------------------------------------------------
    # === HÓA ĐƠN BỔ SUNG (để đủ 28 HĐ) ===
    # ----------------------------------------------------------

    # HD00125 — KH003 Hoàng Gia (ghép cùng HD00128 cho GD013)
    invoices.append((
        "HD00125", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-25", "2024-11-25",
        10_000_000, 0.10, 1_000_000, 11_000_000,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))

    # HD00126 — KH007 Trường Phát
    pre, vat_a, total_a = 22_000_000, *calc_vat(22_000_000, 0.10)
    invoices.append((
        "HD00126", "KH007", "Công ty TNHH Vận Tải Trường Phát", "0700789012",
        "2024-10-27", "2024-11-27",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "PDF", 1.0, 1, None, None,
    ))

    # HD00127 — KH010 Đại Dương
    pre, vat_a, total_a = 55_000_000, *calc_vat(55_000_000, 0.10)
    invoices.append((
        "HD00127", "KH010", "Công ty TNHH Kỹ Thuật Số Đại Dương", "1000012345",
        "2024-10-28", "2024-11-28",
        pre, 0.10, vat_a, total_a,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))

    # HD00128 — KH003 Hoàng Gia (ghép cùng HD00125 cho GD013)
    invoices.append((
        "HD00128", "KH003", "Công ty TNHH Dược Phẩm Hoàng Gia", "0300345678",
        "2024-10-30", "2024-11-30",
        25_909_091, 0.10, 2_590_909, 28_500_000,
        0, "UNPAID", "XML", 1.0, 1, None, None,
    ))

    return invoices


def build_bank_statement_data() -> list[tuple]:
    """
    Tạo 20 dòng sao kê ngân hàng được tính toán chính xác,
    tương ứng với 5 Kịch bản A–E.

    Returns list of tuples theo thứ tự cột bảng bank_statements.
    Thứ tự: (trans_id, trans_date, raw_description, credit_amount, debit_amount, bank_ref, w2ner_json, reconcile_status)
    """
    stmts = []

    # ===========================================================
    # KỊCH BẢN A — 10 dòng sao kê chuẩn cho System 1
    # ===========================================================

    # --- A1: 5 dòng ghi rõ 1 mã hóa đơn ---

    # A1-1: HD00101 — Minh Phát TT tiền hàng → đúng 22_000_000
    stmts.append((
        "GD001", "2024-09-20",
        "CTY TNHH TM MINH PHAT TT TIEN HANG HD 00101",
        22_000_000, 0, "NAPAS20240920001", None, "PENDING",
    ))

    # A1-2: HD00103 — Sao Vàng → đúng 16_500_000
    stmts.append((
        "GD002", "2024-09-25",
        "CTY CP XNK SAO VANG THANH TOAN HD00103 TIEN HANG THANG 9",
        16_500_000, 0, "NAPAS20240925002", None, "PENDING",
    ))

    # A1-3: HD00104 — Việt Hương → đúng 54_000_000
    stmts.append((
        "GD003", "2024-09-28",
        "CONG TY VIET HUONG TT HD SO 00104 HANG HOA DOT 3",
        54_000_000, 0, "NAPAS20240928003", None, "PENDING",
    ))

    # A1-4: HD00126 — Trường Phát → đúng 24_200_000
    stmts.append((
        "GD004", "2024-10-02",
        "CTY TNHH VAN TAI TRUONG PHAT TT HD00126 CUOC VAN CHUYEN T9",
        24_200_000, 0, "NAPAS20241002004", None, "PENDING",
    ))

    # A1-5: HD00112 — Đại Dương → đúng 46_200_000
    stmts.append((
        "GD005", "2024-10-05",
        "CONG TY DAI DUONG THANH TOAN HOA DON 00112 DICH VU THANG 9",
        46_200_000, 0, "NAPAS20241005005", None, "PENDING",
    ))

    # --- A2: 3 dòng thực thể đứt đoạn trả gộp 2–3 hóa đơn ---

    # A2-1: KH002 An Bình trả HD00102 + HD00105 trừ CK 3%
    # Tổng = 33_000_000 + 22_000_000 = 55_000_000
    # CK 3% = 1_650_000 → thực chuyển = 53_350_000
    # (Nhưng ở kịch bản A, trả gộp có ghi mã → System 1 xử lý)
    stmts.append((
        "GD006", "2024-10-05",
        "CTY CP XD AN BINH THANH TOAN HD SO 00102 THANG 9 VA HD 00105 TRU CK 2PC",
        53_900_000,  # 55_000_000 × (1 - 0.02) = 53_900_000
        0, "NAPAS20241005006", None, "PENDING",
    ))

    # A2-2: KH007 Trường Phát gộp HD00106 + HD00107 (Trường Phát không CK)
    # HD00106 = 27_500_000, HD00107 = 19_800_000 → tổng = 47_300_000
    stmts.append((
        "GD007", "2024-10-08",
        "CONG TY TRUONG PHAT THANH TOAN HD 00106 00107 CUOC VAN CHUYEN QUY 3",
        47_300_000, 0, "NAPAS20241008007", None, "PENDING",
    ))

    # A2-3: KH008 Phú Lợi gộp HD00108 + HD00109 + HD00110 (trả cả bộ 3)
    # 11_000_000 + 8_800_000 + 7_700_000 = 27_500_000
    stmts.append((
        "GD008", "2024-10-10",
        "CTY CP NOI THAT PHU LOI TT HD 00108 00109 00110 TIEN HANG T9/2024",
        27_500_000, 0, "NAPAS20241010008", None, "PENDING",
    ))

    # --- A3: 2 dòng không ghi mã HĐ, chỉ ghi tên viết tắt ---

    # A3-1: KH009 Hồng Phúc (1 HĐ duy nhất = HD00111 = 38_500_000)
    stmts.append((
        "GD009", "2024-10-12",
        "DN TN HONG PHUC CHUYEN TIEN HANG THANG 9",
        38_500_000, 0, "NAPAS20241012009", None, "PENDING",
    ))

    # A3-2: KH010 Đại Dương (HD00127 = 60_500_000)
    _pre, _vat, _total = 55_000_000, *calc_vat(55_000_000, 0.10)
    stmts.append((
        "GD010", "2024-10-15",
        "CTY KY THUAT SO DAI DUONG TT CONG NO THANG 10",
        60_500_000, 0, "NAPAS20241015010", None, "PENDING",
    ))

    # ===========================================================
    # KỊCH BẢN B — 3 dòng Subset Sum (không ghi mã HĐ)
    # ===========================================================

    # B-1: KH003 Hoàng Gia chuyển HD00113 + HD00114 = 12_000_000 + 18_500_000 = 30_500_000
    stmts.append((
        "GD011", "2024-10-20",
        "CTY HOANG GIA TT TIEN HANG DOT 1 THANG 10",
        30_500_000, 0, "NAPAS20241020011", None, "PENDING",
    ))

    # B-2: KH003 Hoàng Gia chuyển HD00115 + HD00116 = 25_000_000 + 9_000_000 = 34_000_000
    stmts.append((
        "GD012", "2024-10-22",
        "DUOC PHAM HOANG GIA THANH TOAN TIEN HANG DOT 2 T10",
        34_000_000, 0, "NAPAS20241022012", None, "PENDING",
    ))

    # B-3: KH003 Hoàng Gia chuyển HD00125 + HD00128 = 11tr + 28.5tr = 39_500_000
    stmts.append((
        "GD013", "2024-10-24",
        "HOANG GIA PHARMA TT CONG NO T9 T10",
        39_500_000, 0, "NAPAS20241024013", None, "PENDING",
    ))

    # ===========================================================
    # KỊCH BẢN C — 3 dòng Chiết khấu hợp đồng / Phí ngân hàng
    # ===========================================================

    # C-1: HD00117 — KH001 Minh Phát CK 2%
    # total = 49_500_000; CK 2% = 990_000 → thực chuyển = 48_510_000
    stmts.append((
        "GD014", "2024-10-28",
        "CTY MINH PHAT TT HD00117 TIEN HANG THANG 10",
        48_510_000, 0, "NAPAS20241028014", None, "PENDING",
    ))

    # C-2: HD00118 — KH002 An Bình CK 3%
    # total = 66_000_000; CK 3% = 1_980_000 → thực chuyển = 64_020_000
    stmts.append((
        "GD015", "2024-10-30",
        "AN BINH CONSTRUCTION TT HD00118 T10",
        64_020_000, 0, "NAPAS20241030015", None, "PENDING",
    ))

    # C-3: HD00119 — KH005 Tiến Đạt bị trừ phí NH 22.000
    # total = 27_500_000 → thực nhận = 27_478_000 (mất 22k phí liên NH)
    stmts.append((
        "GD016", "2024-11-01",
        "CTY TIEN DAT TT HD00119 THIET BI THANG 10",
        27_478_000, 0, "NAPAS20241101016", None, "PENDING",
    ))

    # ===========================================================
    # KỊCH BẢN D — 2 dòng Cá nhân chuyển hộ công ty
    # ===========================================================

    # D-1: LE VAN TUNG (GĐ Minh Phát) chuyển HD00120 = 35_200_000
    stmts.append((
        "GD017", "2024-11-05",
        "LE VAN TUNG CHUYEN KHOAN TIEN HANG THANG 11",
        35_200_000, 0, "NAPAS20241105017", None, "PENDING",
    ))

    # D-2: TRAN MINH HUNG (GĐ Hoàng Gia) chuyển HD00121 + HD00122 = 16_000_000 + 15_000_000 = 31_000_000
    stmts.append((
        "GD018", "2024-11-07",
        "TRAN MINH HUNG TT CONG NO THANG 10 THANG 11",
        31_000_000, 0, "NAPAS20241107018", None, "PENDING",
    ))

    # ===========================================================
    # KỊCH BẢN E — 2 dòng OCR sai số tiền
    # ===========================================================

    # E-1: Khách KH004 Sao Vàng chuyển đúng 18_500_000 (giá trị thật)
    # Nhưng DB đang lưu total_amount sai là 10_500_000
    stmts.append((
        "GD019", "2024-11-10",
        "CTY SAO VANG THANH TOAN HD00123 THANG 10",
        18_500_000, 0, "NAPAS20241110019", None, "PENDING",
    ))

    # E-2: Khách KH006 Việt Hương chuyển đúng 50_000_000 (giá trị thật trên ảnh)
    # DB đang lưu sai tổng là 54_000_000 (OCR đọc nhầm vat_rate)
    stmts.append((
        "GD020", "2024-11-12",
        "CONG TY VIET HUONG THANH TOAN HD00124 HANG HOA T10",
        50_000_000, 0, "NAPAS20241112020", None, "PENDING",
    ))

    return stmts


# ==============================================================
# HÀM CHÍNH: seed_all_data
# ==============================================================

def seed_all_data(db_manager: DatabaseManager) -> None:
    """
    Xóa toàn bộ dữ liệu cũ và nạp bộ dữ liệu giả lập kiểm thử
    bao gồm 5 Kịch bản A–E.

    Args:
        db_manager: Instance DatabaseManager đã kết nối.
    """
    logger.info("=" * 60)
    logger.info("BAT DAU SEED DU LIEU KIEM THU PHASE 1")
    logger.info("=" * 60)

    # -- 1. Xóa dữ liệu cũ --
    db_manager.clear_all_data()
    logger.info("[Seed] Da xoa du lieu cu.")

    # -- 2. Chèn 10 khách hàng --
    sql_cust = """
        INSERT OR IGNORE INTO customers
            (customer_id, customer_name, short_name, tax_code,
             contract_discount_rate, credit_days)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    db_manager.execute_query(sql_cust, CUSTOMERS, many=True)
    logger.info(f"[Seed] Da chen {len(CUSTOMERS)} khach hang.")

    # -- 3. Chèn 6 payer aliases --
    sql_alias = """
        INSERT OR IGNORE INTO payer_aliases (person_name, role, customer_id)
        VALUES (?, ?, ?)
    """
    db_manager.execute_query(sql_alias, PAYER_ALIASES, many=True)
    logger.info(f"[Seed] Da chen {len(PAYER_ALIASES)} payer aliases.")

    # -- 4. Chen 28 hoa don --
    invoices = build_invoice_data()
    sql_inv = """
        INSERT OR IGNORE INTO invoices
            (invoice_id, customer_id, customer_name, tax_code,
             issue_date, due_date,
             pre_tax_amount, vat_rate, vat_amount, total_amount,
             paid_amount, status, source_type,
             ocr_confidence, ocr_audit_passed, ocr_suspect_note,
             raw_ocr_true_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    db_manager.execute_query(sql_inv, invoices, many=True)
    logger.info(f"[Seed] Da chen {len(invoices)} hoa don.")

    # -- 5. Chen 20 dong sao ke ngan hang --
    statements = build_bank_statement_data()
    sql_stmt = """
        INSERT OR IGNORE INTO bank_statements
            (trans_id, trans_date, raw_description,
             credit_amount, debit_amount, bank_ref,
             w2ner_json, reconcile_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    db_manager.execute_query(sql_stmt, statements, many=True)
    logger.info(f"[Seed] Da chen {len(statements)} dong sao ke ngan hang.")

    # -- 6. Verification --
    _verify_seed(db_manager)

    logger.info("=" * 60)
    logger.info("SEED DU LIEU HOAN THANH THANH CONG!")
    logger.info("=" * 60)


def _verify_seed(db_manager: DatabaseManager) -> None:
    """
    Chay kiem tra co ban de xac nhan tinh nhat quan du lieu.
    In ra bang tom tat kich ban.
    """
    df_cust  = db_manager.fetch_df("SELECT COUNT(*) AS n FROM customers")
    df_alias = db_manager.fetch_df("SELECT COUNT(*) AS n FROM payer_aliases")
    df_inv   = db_manager.fetch_df("SELECT COUNT(*) AS n FROM invoices")
    df_stmt  = db_manager.fetch_df("SELECT COUNT(*) AS n FROM bank_statements")

    sep = "=" * 60
    print(f"\n{sep}")
    print("  BANG KIEM TRA DU LIEU SEED (Phase 1)")
    print(sep)
    print(f"  Khach hang   (customers):            {df_cust.iloc[0]['n']:>4} / 10")
    print(f"  Nguoi dai dien (payer_aliases):      {df_alias.iloc[0]['n']:>4} / 6")
    print(f"  Hoa don      (invoices):             {df_inv.iloc[0]['n']:>4} / 28")
    print(f"  Sao ke NH    (bank_statements):      {df_stmt.iloc[0]['n']:>4} / 20")
    print(sep)

    # Trang thai sao ke
    df_scen = db_manager.fetch_df("""
        SELECT
            reconcile_status,
            COUNT(*)           AS so_dong,
            SUM(credit_amount) / 1e6 AS tong_tien_trieu_vnd
        FROM bank_statements
        GROUP BY reconcile_status
    """)
    print("\n  TRANG THAI SAO KE:")
    print(df_scen.to_string(index=False))

    # Hoa don OCR suspect
    df_ocr = db_manager.fetch_df("""
        SELECT invoice_id, total_amount, raw_ocr_true_amount,
               ocr_confidence, ocr_audit_passed
        FROM invoices
        WHERE ocr_audit_passed = 0
    """)
    print("\n  HOA DON CO LOI OCR (can kiem tra lai):")
    print(df_ocr.to_string(index=False))

    # Payer aliases
    df_alias_check = db_manager.fetch_df("""
        SELECT a.person_name, a.role, c.customer_id, c.short_name
        FROM payer_aliases a
        JOIN customers c ON a.customer_id = c.customer_id
    """)
    print("\n  DANH SACH NGUOI DAI DIEN THANH TOAN:")
    print(df_alias_check.to_string(index=False))
    print(sep + "\n")


# ==============================================================
# RUN DIRECTLY
# ==============================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    db = DatabaseManager()
    seed_all_data(db)

    # Confirmation
    print("\n[OK] Seed hoan tat! Database tai:", db.db_path)
    stats = db.get_summary_stats()
    print(f"   Total receivable : {stats['total_receivable']:>15,} VND")
    print(f"   Pending stmts    : {stats['pending_statements']:>4} rows")
    print(f"   Unapproved match : {stats['unapproved_matches']:>4} records")
