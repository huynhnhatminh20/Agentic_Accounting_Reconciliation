# ============================================================
# database/db_manager.py — Quản lý SQLite Database
# Hệ thống Đối soát & Định khoản Kế toán (TT 200/133)
# ============================================================

import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Optional, Union

import pandas as pd

# Thêm thư mục gốc vào sys.path để import config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, ReconcileStatus, InvoiceStatus

logger = logging.getLogger(__name__)


# ==============================================================
# DDL — Định nghĩa 5 bảng SQL chuẩn
# ==============================================================

DDL_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id             TEXT PRIMARY KEY,
    customer_name           TEXT NOT NULL,
    short_name              TEXT,
    tax_code                TEXT,
    contract_discount_rate  REAL DEFAULT 0.0,
    credit_days             INTEGER DEFAULT 30
);
"""

# Danh bạ cá nhân (Giám đốc / Kế toán) chuyển khoản thay công ty
DDL_PAYER_ALIASES = """
CREATE TABLE IF NOT EXISTS payer_aliases (
    alias_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    role        TEXT,                           -- GD (Giám đốc), KT (Kế toán), NV (Nhân viên)
    customer_id TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

# Hóa đơn bán ra — Phải thu khách hàng
DDL_INVOICES = """
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id          TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL,
    customer_name       TEXT NOT NULL,
    tax_code            TEXT,
    issue_date          TEXT NOT NULL,          -- ISO 8601: YYYY-MM-DD
    due_date            TEXT NOT NULL,
    pre_tax_amount      INTEGER NOT NULL,       -- Tiền chưa thuế (VNĐ, làm tròn)
    vat_rate            REAL NOT NULL,          -- Thuế suất: 0.08, 0.10, 0.0
    vat_amount          INTEGER NOT NULL,       -- Tiền thuế GTGT
    total_amount        INTEGER NOT NULL,       -- Tổng tiền thanh toán (pre_tax + vat)
    paid_amount         INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'UNPAID',  -- UNPAID | PAID | PARTIAL
    source_type         TEXT,                   -- XML | PDF | IMAGE_OCR
    ocr_confidence      REAL DEFAULT 1.0,       -- Độ tin cậy OCR (0.0–1.0); 1.0 = không qua OCR
    ocr_audit_passed    INTEGER DEFAULT 1,      -- 1 = OK, 0 = cần kiểm tra lại
    ocr_suspect_note    TEXT,                   -- Ghi chú nghi ngờ từ OCR
    raw_ocr_true_amount INTEGER,               -- Giá trị thật trên ảnh (để cross-check khi OCR sai)
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

# Sao kê ngân hàng — dữ liệu thô nhập từ file ngân hàng
DDL_BANK_STATEMENTS = """
CREATE TABLE IF NOT EXISTS bank_statements (
    trans_id            TEXT PRIMARY KEY,
    trans_date          TEXT NOT NULL,          -- YYYY-MM-DD
    raw_description     TEXT NOT NULL,          -- Nội dung chuyển khoản nguyên bản
    credit_amount       INTEGER NOT NULL,       -- Số tiền nhận vào (phải thu)
    debit_amount        INTEGER DEFAULT 0,      -- Số tiền chuyển đi (phí, trả lại)
    bank_ref            TEXT,                   -- Mã tham chiếu giao dịch của ngân hàng
    w2ner_json          TEXT,                   -- JSON kết quả NER: {entities: [...]}
    reconcile_status    TEXT DEFAULT 'PENDING'  -- PENDING | AUTO_MATCHED_SYS1 | AGENT_RESOLVED_SYS2 | NEED_HUMAN_REVIEW
);
"""

# Sổ kết quả đối soát & Bút toán định khoản
DDL_RECONCILIATION_LEDGER = """
CREATE TABLE IF NOT EXISTS reconciliation_ledger (
    match_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trans_id                TEXT NOT NULL,
    matched_invoice_ids     TEXT,               -- JSON list: ["HD00101", "HD00102"]
    customer_id             TEXT,
    customer_name           TEXT,
    bank_amount             INTEGER NOT NULL,   -- Số tiền thực nhận trên sao kê
    invoiced_total          INTEGER,            -- Tổng hóa đơn được ghép
    discount_amount         INTEGER DEFAULT 0,  -- Tiền chiết khấu (nếu có)
    bank_fee_amount         INTEGER DEFAULT 0,  -- Phí ngân hàng (nếu có)
    discrepancy_amount      INTEGER DEFAULT 0,  -- Chênh lệch còn lại sau giải thích
    resolution_source       TEXT,               -- SYSTEM_1_RULE | SYSTEM_2_REACT_AGENT | MANUAL
    match_type              TEXT,               -- Xem MatchType trong config.py
    journal_entries_json    TEXT,               -- JSON danh sách bút toán Nợ/Có chi tiết
    agent_reasoning_trace   TEXT,               -- Toàn bộ Thought→Action→Observation của Agent
    explanation             TEXT,               -- Giải thích ngắn gọn bằng tiếng Việt
    approved                INTEGER DEFAULT 0,  -- 0 = chờ duyệt, 1 = kế toán đã xác nhận
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trans_id) REFERENCES bank_statements(trans_id)
);
"""

ALL_DDL = [
    DDL_CUSTOMERS,
    DDL_PAYER_ALIASES,
    DDL_INVOICES,
    DDL_BANK_STATEMENTS,
    DDL_RECONCILIATION_LEDGER,
]

# Các index tăng tốc truy vấn thường dùng
DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_invoices_customer   ON invoices(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_status     ON invoices(status);",
    "CREATE INDEX IF NOT EXISTS idx_bank_reconcile      ON bank_statements(reconcile_status);",
    "CREATE INDEX IF NOT EXISTS idx_ledger_trans        ON reconciliation_ledger(trans_id);",
    "CREATE INDEX IF NOT EXISTS idx_ledger_customer     ON reconciliation_ledger(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_alias_customer      ON payer_aliases(customer_id);",
]


# ==============================================================
# Class DatabaseManager
# ==============================================================

class DatabaseManager:
    """
    Lớp quản lý toàn bộ thao tác với SQLite database kế toán.

    Sử dụng:
        db = DatabaseManager()
        df = db.fetch_df("SELECT * FROM invoices WHERE status='UNPAID'")
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        logger.info(f"[DB] Kết nối SQLite: {self.db_path}")

    # ----------------------------------------------------------
    # Context manager — tự động commit/rollback & đóng connection
    # ----------------------------------------------------------

    @contextmanager
    def _get_connection(self):
        """Trả về sqlite3 Connection với row_factory và timeout an toàn."""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")   # Tốt hơn cho concurrent read
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"[DB] Lỗi transaction, đã rollback: {exc}")
            raise
        finally:
            conn.close()

    # ----------------------------------------------------------
    # Khởi tạo schema
    # ----------------------------------------------------------

    def init_db(self) -> None:
        """Tạo tất cả bảng và index nếu chưa tồn tại."""
        with self._get_connection() as conn:
            for ddl in ALL_DDL:
                conn.execute(ddl)
            for idx_sql in DDL_INDEXES:
                conn.execute(idx_sql)
        logger.info("[DB] Schema khởi tạo thành công.")

    # ----------------------------------------------------------
    # Truy vấn → DataFrame
    # ----------------------------------------------------------

    def fetch_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """
        Thực thi câu SELECT và trả về pandas DataFrame.

        Args:
            sql:    Câu SQL SELECT.
            params: Tuple tham số (dùng ? placeholder để tránh SQL injection).

        Returns:
            pd.DataFrame — rỗng nếu không có kết quả.
        """
        with self._get_connection() as conn:
            try:
                df = pd.read_sql_query(sql, conn, params=params)
                return df
            except Exception as exc:
                logger.error(f"[DB] fetch_df lỗi: {exc}\nSQL: {sql}")
                return pd.DataFrame()

    # ----------------------------------------------------------
    # Thực thi lệnh DML / DDL
    # ----------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        params: Union[tuple, list] = (),
        many: bool = False,
    ) -> int:
        """
        Thực thi INSERT / UPDATE / DELETE hoặc DDL.

        Args:
            sql:    Câu SQL.
            params: Tuple tham số, hoặc list of tuples khi many=True.
            many:   True → dùng executemany (batch insert).

        Returns:
            rowcount — số dòng bị ảnh hưởng.
        """
        with self._get_connection() as conn:
            try:
                if many:
                    cur = conn.executemany(sql, params)
                else:
                    cur = conn.execute(sql, params)
                return cur.rowcount
            except Exception as exc:
                logger.error(f"[DB] execute_query lỗi: {exc}\nSQL: {sql}")
                raise

    # ----------------------------------------------------------
    # Cập nhật trạng thái đối soát (cập nhật bank_statements)
    # ----------------------------------------------------------

    def update_reconcile_status(
        self,
        trans_id: str,
        new_status: str,
    ) -> bool:
        """
        Cập nhật reconcile_status của một dòng sao kê.

        Args:
            trans_id:   Mã giao dịch ngân hàng.
            new_status: Một trong ReconcileStatus.*

        Returns:
            True nếu cập nhật thành công.
        """
        valid_statuses = {
            ReconcileStatus.PENDING,
            ReconcileStatus.AUTO_MATCHED_SYS1,
            ReconcileStatus.AGENT_RESOLVED_SYS2,
            ReconcileStatus.NEED_HUMAN_REVIEW,
        }
        if new_status not in valid_statuses:
            raise ValueError(f"Status '{new_status}' không hợp lệ. Chọn: {valid_statuses}")

        rows = self.execute_query(
            "UPDATE bank_statements SET reconcile_status=? WHERE trans_id=?",
            (new_status, trans_id),
        )
        logger.info(f"[DB] trans_id={trans_id} → status={new_status} ({rows} dòng cập nhật)")
        return rows > 0

    # ----------------------------------------------------------
    # Cập nhật trạng thái hóa đơn & số tiền đã thanh toán
    # ----------------------------------------------------------

    def update_invoice_payment(
        self,
        invoice_id: str,
        paid_amount: int,
        status: Optional[str] = None,
    ) -> bool:
        """
        Cập nhật paid_amount và status cho hóa đơn sau khi khớp.

        Args:
            invoice_id:  Mã hóa đơn.
            paid_amount: Tổng tiền đã nhận (cộng dồn).
            status:      Tự động tính nếu để None.

        Returns:
            True nếu cập nhật thành công.
        """
        if status is None:
            # Lấy total_amount hiện tại để tự xác định trạng thái
            df = self.fetch_df(
                "SELECT total_amount FROM invoices WHERE invoice_id=?",
                (invoice_id,),
            )
            if df.empty:
                logger.warning(f"[DB] Không tìm thấy invoice_id={invoice_id}")
                return False
            total = int(df.iloc[0]["total_amount"])
            if paid_amount >= total:
                status = InvoiceStatus.PAID
            elif paid_amount > 0:
                status = InvoiceStatus.PARTIAL
            else:
                status = InvoiceStatus.UNPAID

        rows = self.execute_query(
            "UPDATE invoices SET paid_amount=?, status=? WHERE invoice_id=?",
            (paid_amount, status, invoice_id),
        )
        logger.info(
            f"[DB] invoice_id={invoice_id} paid={paid_amount:,}đ status={status} ({rows} dòng)"
        )
        return rows > 0

    # ----------------------------------------------------------
    # Ghi kết quả đối soát vào reconciliation_ledger
    # ----------------------------------------------------------

    def insert_reconciliation_result(self, record: dict) -> int:
        """
        Chèn một bản ghi kết quả đối soát vào reconciliation_ledger.

        Args:
            record: Dict với các key tương ứng cột của bảng.
                    matched_invoice_ids và journal_entries_json
                    sẽ được tự động serialize thành JSON string nếu truyền list/dict.

        Returns:
            match_id mới được tạo (AUTOINCREMENT).
        """
        # Serialize list/dict → JSON string
        for json_field in ("matched_invoice_ids", "journal_entries_json"):
            if json_field in record and not isinstance(record[json_field], str):
                record[json_field] = json.dumps(record[json_field], ensure_ascii=False)

        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        sql = f"INSERT INTO reconciliation_ledger ({columns}) VALUES ({placeholders})"

        with self._get_connection() as conn:
            cur = conn.execute(sql, tuple(record.values()))
            return cur.lastrowid

    # ----------------------------------------------------------
    # Helper: Lấy danh sách hóa đơn chưa thanh toán của khách hàng
    # ----------------------------------------------------------

    def get_unpaid_invoices(self, customer_id: str) -> pd.DataFrame:
        """Trả về DataFrame hóa đơn UNPAID hoặc PARTIAL của một khách."""
        return self.fetch_df(
            """
            SELECT invoice_id, total_amount, paid_amount,
                   (total_amount - paid_amount) AS remaining_amount,
                   issue_date, due_date, status
            FROM invoices
            WHERE customer_id = ?
              AND status IN ('UNPAID', 'PARTIAL')
            ORDER BY issue_date ASC
            """,
            (customer_id,),
        )

    # ----------------------------------------------------------
    # Helper: Lấy sao kê chưa đối soát
    # ----------------------------------------------------------

    def get_pending_statements(self) -> pd.DataFrame:
        """Trả về DataFrame tất cả dòng sao kê chưa xử lý."""
        return self.fetch_df(
            "SELECT * FROM bank_statements WHERE reconcile_status='PENDING' ORDER BY trans_date ASC"
        )

    # ----------------------------------------------------------
    # Helper: Thống kê nhanh
    # ----------------------------------------------------------

    def get_summary_stats(self) -> dict:
        """Trả về dict thống kê tổng quan cho dashboard."""
        stats = {}

        # Tổng hóa đơn chưa thu
        df_inv = self.fetch_df(
            "SELECT SUM(total_amount - paid_amount) AS total_receivable FROM invoices WHERE status!='PAID'"
        )
        stats["total_receivable"] = int(df_inv.iloc[0]["total_receivable"] or 0)

        # Số dòng sao kê chờ xử lý
        df_pending = self.fetch_df(
            "SELECT COUNT(*) AS cnt FROM bank_statements WHERE reconcile_status='PENDING'"
        )
        stats["pending_statements"] = int(df_pending.iloc[0]["cnt"] or 0)

        # Số kết quả đối soát chờ duyệt
        df_unapproved = self.fetch_df(
            "SELECT COUNT(*) AS cnt FROM reconciliation_ledger WHERE approved=0"
        )
        stats["unapproved_matches"] = int(df_unapproved.iloc[0]["cnt"] or 0)

        return stats

    # ----------------------------------------------------------
    # Xóa toàn bộ dữ liệu (dùng cho seed/test)
    # ----------------------------------------------------------

    def clear_all_data(self) -> None:
        """Xóa sạch dữ liệu tất cả bảng (giữ schema)."""
        tables = [
            "reconciliation_ledger",
            "bank_statements",
            "invoices",
            "payer_aliases",
            "customers",
        ]
        with self._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys=OFF;")
            for t in tables:
                conn.execute(f"DELETE FROM {t};")
            conn.execute("PRAGMA foreign_keys=ON;")
        logger.info("[DB] Đã xóa toàn bộ dữ liệu.")

    # ----------------------------------------------------------
    # Repr
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return f"DatabaseManager(db_path='{self.db_path}')"

