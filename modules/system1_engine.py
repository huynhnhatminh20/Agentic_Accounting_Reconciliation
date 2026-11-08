# ============================================================
# modules/system1_engine.py
# System 1: Fast Symbolic Reconciliation Engine
#
# High-speed deterministic reconciliation engine (0% mathematical error)
# Processes standard transactions using 2 deterministic rules:
#   - Rule 1A & 1B: Exact 1-1 or 1-N matching by INVOICE_ID (continuous/discontinuous)
#   - Rule 2: Fuzzy matching 1-1 by company name (when invoice ID is missing)
#
# Generates TT 200/133 compliant accounting journal entries:
#   Nợ TK 1121: credit_amount
#   Nợ TK 521: discount_amount (if applicable)
#   Có TK 131: invoiced_total
#
# Failed transactions are pushed to EXCEPTION_QUEUE for System 2 ReAct Agent.
# ============================================================

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    TOLERANCE_VND,
    FUZZY_MATCH_THRESHOLD,
    InvoiceStatus,
    ReconcileStatus,
    MatchType,
    ResolutionSource,
)
from database.db_manager import DatabaseManager
from modules.w2ner_adapter import W2NERStatementExtractor, W2NEROutput

logger = logging.getLogger(__name__)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def remove_accents(text: str) -> str:
    """Strip Vietnamese diacritics and convert to uppercase for normalization."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_text).strip().upper()


def format_currency(amount: Union[int, float]) -> str:
    """Format integer/float amount to Vietnamese currency string."""
    return f"{int(amount):,}đ"


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class ReconciliationLedgerRecord:
    """Data transfer object for reconciliation_ledger entry."""
    trans_id: str
    matched_invoice_ids: List[str]
    customer_id: str
    customer_name: str
    bank_amount: int
    invoiced_total: int
    discount_amount: int = 0
    bank_fee_amount: int = 0
    discrepancy_amount: int = 0
    resolution_source: str = ResolutionSource.SYSTEM_1_RULE
    match_type: str = MatchType.EXACT_1_TO_1
    journal_entries: List[Dict[str, Any]] = field(default_factory=list)
    agent_reasoning_trace: str = ""
    explanation: str = ""
    approved: int = 0

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dict with serialized JSON fields for DB insertion."""
        return {
            "trans_id": self.trans_id,
            "matched_invoice_ids": json.dumps(self.matched_invoice_ids, ensure_ascii=False),
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "bank_amount": self.bank_amount,
            "invoiced_total": self.invoiced_total,
            "discount_amount": self.discount_amount,
            "bank_fee_amount": self.bank_fee_amount,
            "discrepancy_amount": self.discrepancy_amount,
            "resolution_source": self.resolution_source,
            "match_type": self.match_type,
            "journal_entries_json": json.dumps(self.journal_entries, ensure_ascii=False),
            "agent_reasoning_trace": self.agent_reasoning_trace,
            "explanation": self.explanation,
            "approved": self.approved,
        }


@dataclass
class ExceptionItem:
    """Represents a transaction rejected by System 1 to be investigated by System 2."""
    trans_id: str
    trans_date: str
    raw_description: str
    credit_amount: int
    bank_ref: Optional[str]
    w2ner_data: Dict[str, Any]
    failure_reason: str
    trans_row: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# MAIN SYSTEM 1 ENGINE
# ===========================================================================

class System1ReconciliationEngine:
    """
    System 1: Fast Symbolic Reconciliation Engine.

    Executes 2 deterministic rules:
      1. Rule 1A & 1B: Exact 1-1 or 1-N matching by INVOICE_ID with discount calculation.
      2. Rule 2: Fuzzy matching 1-1 by company name when INVOICE_ID is absent.

    Unmatched transactions are gathered into an EXCEPTION_QUEUE for System 2.
    """

    def __init__(
        self,
        tolerance_vnd: int = TOLERANCE_VND,
        fuzzy_threshold: int = FUZZY_MATCH_THRESHOLD,
    ):
        self.tolerance_vnd = tolerance_vnd
        self.fuzzy_threshold = fuzzy_threshold
        logger.info(
            f"[System1] Engine initialized (Tolerance={tolerance_vnd:,}đ, "
            f"FuzzyThreshold={fuzzy_threshold})"
        )

    # -----------------------------------------------------------------------
    # PRIMARY PIPELINE
    # -----------------------------------------------------------------------

    def run_fast_reconciliation(
        self,
        db_manager: DatabaseManager,
        w2ner_extractor: W2NERStatementExtractor,
    ) -> Dict[str, Any]:
        """
        Execute fast reconciliation on all PENDING bank statements.

        Args:
            db_manager: DatabaseManager instance.
            w2ner_extractor: W2NERStatementExtractor instance.

        Returns:
            Dict containing:
              - total_processed: Total statements inspected
              - matched_count: Statements auto-matched by System 1
              - exception_count: Statements pushed to Exception Queue
              - matched_records: List of matched ledger records
              - exception_queue: List of ExceptionItem dicts for System 2
        """
        pending_df = db_manager.fetch_df(
            "SELECT * FROM bank_statements WHERE reconcile_status = 'PENDING' ORDER BY trans_date ASC"
        )

        total_processed = len(pending_df)
        matched_records: List[ReconciliationLedgerRecord] = []
        exception_queue: List[ExceptionItem] = []

        logger.info(f"[System1] Starting fast reconciliation for {total_processed} pending statements.")

        if pending_df.empty:
            return {
                "total_processed": 0,
                "matched_count": 0,
                "exception_count": 0,
                "matched_records": [],
                "exception_queue": [],
            }

        # Cache customers for fuzzy matching
        customers_df = db_manager.fetch_df(
            "SELECT customer_id, customer_name, short_name, contract_discount_rate FROM customers"
        )

        for _, row in pending_df.iterrows():
            trans_id = str(row["trans_id"])
            trans_date = str(row["trans_date"])
            raw_desc = str(row["raw_description"])
            credit_amount = int(row["credit_amount"])
            bank_ref = row["bank_ref"]

            # Step 1: W2NER Entity Extraction
            w2ner_output: W2NEROutput = w2ner_extractor.extract_entities(raw_desc)
            w2ner_dict = w2ner_output.to_dict()
            w2ner_json_str = json.dumps(w2ner_dict, ensure_ascii=False)

            # Persist w2ner_json in bank_statements
            db_manager.execute_query(
                "UPDATE bank_statements SET w2ner_json = ? WHERE trans_id = ?",
                (w2ner_json_str, trans_id),
            )

            # Step 2: Evaluate Rule 1 (Invoice ID Matching)
            rule1_result, rule1_reason = self._try_rule_1_invoice_matching(
                row=row,
                w2ner_output=w2ner_output,
                db_manager=db_manager,
            )

            if rule1_result is not None:
                self._apply_successful_match(db_manager, rule1_result)
                matched_records.append(rule1_result)
                continue

            # Step 3: Evaluate Rule 2 (Fuzzy Company Name Matching)
            rule2_result, rule2_reason = self._try_rule_2_fuzzy_company_matching(
                row=row,
                w2ner_output=w2ner_output,
                customers_df=customers_df,
                db_manager=db_manager,
            )

            if rule2_result is not None:
                self._apply_successful_match(db_manager, rule2_result)
                matched_records.append(rule2_result)
                continue

            # Step 4: If both rules fail, push to Exception Queue
            failure_reason = rule1_reason or rule2_reason or "Không khớp được theo bất kỳ quy tắc tất định nào của System 1."
            exception_item = ExceptionItem(
                trans_id=trans_id,
                trans_date=trans_date,
                raw_description=raw_desc,
                credit_amount=credit_amount,
                bank_ref=bank_ref,
                w2ner_data=w2ner_dict,
                failure_reason=failure_reason,
                trans_row=row.to_dict(),
            )
            exception_queue.append(exception_item)
            logger.info(f"[System1] Exception -> trans_id={trans_id}: {failure_reason}")

        logger.info(
            f"[System1] Completed fast reconciliation: {len(matched_records)} matched, "
            f"{len(exception_queue)} exceptions queued."
        )

        return {
            "total_processed": total_processed,
            "matched_count": len(matched_records),
            "exception_count": len(exception_queue),
            "matched_records": [asdict(r) for r in matched_records],
            "exception_queue": [e.to_dict() for e in exception_queue],
        }

    # -----------------------------------------------------------------------
    # RULE 1: INVOICE ID MATCHING (1-1 or 1-N)
    # -----------------------------------------------------------------------

    def _try_rule_1_invoice_matching(
        self,
        row: pd.Series,
        w2ner_output: W2NEROutput,
        db_manager: DatabaseManager,
    ) -> Tuple[Optional[ReconciliationLedgerRecord], Optional[str]]:
        """
        Check Rule 1:
          - Extract invoice IDs.
          - Query invoices table.
          - Calculate invoiced_total, discount_rate (if explicitly stated in statement).
          - Check math within tolerance and verify ocr_audit_passed == 1.
        """
        extracted_ids = w2ner_output.invoice_ids
        if not extracted_ids:
            return None, None  # Rule 1 does not apply, proceed to Rule 2

        trans_id = str(row["trans_id"])
        credit_amount = int(row["credit_amount"])

        # Normalize query IDs to check both raw digits and HD prefix (e.g. '00101' and 'HD00101')
        candidate_ids = []
        for inv_id in extracted_ids:
            clean_digits = re.sub(r"\D", "", inv_id).zfill(5)
            candidate_ids.extend([clean_digits, f"HD{clean_digits}"])

        placeholders = ", ".join(["?"] * len(candidate_ids))
        invoices_df = db_manager.fetch_df(
            f"""
            SELECT * FROM invoices
            WHERE invoice_id IN ({placeholders})
              AND status IN ('{InvoiceStatus.UNPAID}', '{InvoiceStatus.PARTIAL}')
            """,
            tuple(candidate_ids),
        )

        if invoices_df.empty:
            return None, f"Tìm thấy mã HĐ {extracted_ids} trong nội dung nhưng không có hóa đơn tương ứng ở trạng thái chưa thanh toán trong DB."

        # Check if all extracted invoice IDs were resolved in DB
        found_ids = set(invoices_df["invoice_id"].tolist())
        unfound = [
            inv for inv in extracted_ids
            if inv not in found_ids and f"HD{inv.zfill(5)}" not in found_ids
        ]
        if unfound:
            return None, f"Có {len(unfound)} mã HĐ không tồn tại trong DB hoặc đã thanh toán: {unfound}."

        # Check OCR audit flag
        ocr_failed_rows = invoices_df[invoices_df["ocr_audit_passed"] == 0]
        if not ocr_failed_rows.empty:
            failed_id = ocr_failed_rows.iloc[0]["invoice_id"]
            return None, (
                f"Hóa đơn {failed_id} có cờ nghi ngờ lỗi OCR (ocr_audit_passed=0, "
                f"conf={ocr_failed_rows.iloc[0]['ocr_confidence']}). Cần System 2 điều tra lại ảnh gốc."
            )

        # Calculate invoiced total (remaining unpaid amount)
        invoices_df["remaining"] = invoices_df["total_amount"] - invoices_df["paid_amount"]
        invoiced_total = int(invoices_df["remaining"].sum())

        # Check explicit discount rate parsed from statement description
        discount_rate = 0.0
        if w2ner_output.discount_rates:
            discount_rate = float(w2ner_output.discount_rates[0])

        if discount_rate > 0.0:
            expected_payment = round(invoiced_total * (1.0 - discount_rate))
            discount_amount = invoiced_total - expected_payment
        else:
            expected_payment = invoiced_total
            discount_amount = 0

        # Mathematical verification
        diff = abs(credit_amount - expected_payment)
        if diff > self.tolerance_vnd:
            return None, (
                f"Số tiền thực nhận {credit_amount:,}đ lệch so với tổng hóa đơn "
                f"{expected_payment:,}đ (chênh lệch {credit_amount - expected_payment:,}đ). "
                f"Có thể do khách tự trừ chiết khấu hoặc phí ngân hàng chưa ghi rõ."
            )

        # Match succeeded!
        matched_inv_ids = invoices_df["invoice_id"].tolist()
        cust_id = str(invoices_df.iloc[0]["customer_id"])
        cust_name = str(invoices_df.iloc[0]["customer_name"])

        match_type = (
            MatchType.EXACT_1_TO_1
            if len(matched_inv_ids) == 1
            else MatchType.MULTI_INVOICE_1_TO_N
        )

        journal_entries = self._generate_journal_entries(
            credit_amount=credit_amount,
            invoiced_total=invoiced_total,
            discount_amount=discount_amount,
            bank_fee_amount=0,
            matched_inv_ids=matched_inv_ids,
            customer_name=cust_name,
        )

        discount_str = f" (đã trừ chiết khấu {discount_rate:.0%}: {discount_amount:,}đ)" if discount_amount > 0 else ""
        explanation = (
            f"[System 1] Khớp tự động quy tắc mã HĐ: {', '.join(matched_inv_ids)}. "
            f"Tổng tiền HĐ: {invoiced_total:,}đ{discount_str}, thực nhận: {credit_amount:,}đ."
        )

        record = ReconciliationLedgerRecord(
            trans_id=trans_id,
            matched_invoice_ids=matched_inv_ids,
            customer_id=cust_id,
            customer_name=cust_name,
            bank_amount=credit_amount,
            invoiced_total=invoiced_total,
            discount_amount=discount_amount,
            bank_fee_amount=0,
            discrepancy_amount=0,
            resolution_source=ResolutionSource.SYSTEM_1_RULE,
            match_type=match_type,
            journal_entries=journal_entries,
            agent_reasoning_trace="[System 1 Deterministic Rule 1] Khớp chính xác 100% theo mã hóa đơn từ sao kê.",
            explanation=explanation,
            approved=0,
        )

        return record, None

    # -----------------------------------------------------------------------
    # RULE 2: FUZZY COMPANY NAME MATCHING (1-1)
    # -----------------------------------------------------------------------

    def _try_rule_2_fuzzy_company_matching(
        self,
        row: pd.Series,
        w2ner_output: W2NEROutput,
        customers_df: pd.DataFrame,
        db_manager: DatabaseManager,
    ) -> Tuple[Optional[ReconciliationLedgerRecord], Optional[str]]:
        """
        Check Rule 2:
          - INVOICE_ID is empty.
          - PAYER_ORG is non-empty.
          - RapidFuzz matches PAYER_ORG to customer_name or short_name >= FUZZY_MATCH_THRESHOLD.
          - Customer has EXACTLY 1 unpaid invoice where total_amount == credit_amount.
          - Invoice has ocr_audit_passed == 1.
        """
        payer_orgs = w2ner_output.payer_orgs
        if not payer_orgs:
            # Check if there is only SENDER_PER
            if w2ner_output.sender_persons:
                return None, (
                    f"Nội dung không có mã HĐ hay tên công ty, phát hiện người đại diện cá nhân: "
                    f"{w2ner_output.sender_persons}. Cần System 2 tra cứu danh bạ (Alias Lookup)."
                )
            return None, "Không tìm thấy mã hóa đơn, tên công ty hay thông tin người gửi hợp lệ."

        trans_id = str(row["trans_id"])
        credit_amount = int(row["credit_amount"])
        payer_text = payer_orgs[0]
        payer_norm = remove_accents(payer_text)

        # Match against customers table
        best_match = None
        best_score = -1.0

        for _, cust in customers_df.iterrows():
            c_name_norm = remove_accents(cust["customer_name"])
            c_short_norm = remove_accents(cust["short_name"])

            # Compute score against both full name and short name
            score_name = fuzz.token_set_ratio(payer_norm, c_name_norm)
            score_short = fuzz.token_set_ratio(payer_norm, c_short_norm)
            max_score = max(score_name, score_short)

            if max_score > best_score:
                best_score = max_score
                best_match = cust

        if best_match is None or best_score < self.fuzzy_threshold:
            return None, (
                f"Tên công ty '{payer_text}' không đạt ngưỡng so khớp mờ tối thiểu "
                f"({best_score:.1f} < {self.fuzzy_threshold})."
            )

        matched_cust_id = str(best_match["customer_id"])
        matched_cust_name = str(best_match["customer_name"])

        # Fetch unpaid invoices of matched customer
        unpaid_invoices = db_manager.fetch_df(
            f"""
            SELECT * FROM invoices
            WHERE customer_id = ?
              AND status IN ('{InvoiceStatus.UNPAID}', '{InvoiceStatus.PARTIAL}')
            """,
            (matched_cust_id,),
        )

        if unpaid_invoices.empty:
            return None, f"Công ty '{matched_cust_name}' (khớp {best_score:.0f}%) không có hóa đơn nào chưa thanh toán."

        # Filter invoices matching credit_amount
        unpaid_invoices["remaining"] = unpaid_invoices["total_amount"] - unpaid_invoices["paid_amount"]
        amount_matches = unpaid_invoices[
            (unpaid_invoices["remaining"] - credit_amount).abs() <= self.tolerance_vnd
        ]

        if len(amount_matches) == 1:
            inv_row = amount_matches.iloc[0]

            # Check OCR audit flag
            if inv_row["ocr_audit_passed"] == 0:
                return None, (
                    f"Hóa đơn {inv_row['invoice_id']} có cờ nghi ngờ lỗi OCR (ocr_audit_passed=0). "
                    f"Cần System 2 điều tra lại ảnh gốc."
                )

            inv_id = str(inv_row["invoice_id"])
            invoiced_total = int(inv_row["remaining"])

            journal_entries = self._generate_journal_entries(
                credit_amount=credit_amount,
                invoiced_total=invoiced_total,
                discount_amount=0,
                bank_fee_amount=0,
                matched_inv_ids=[inv_id],
                customer_name=matched_cust_name,
            )

            explanation = (
                f"[System 1] Khớp tự động quy tắc tên công ty viết tắt ({payer_text} -> {matched_cust_name}, {best_score:.0f}%). "
                f"Khách hàng chỉ có duy nhất 1 hóa đơn {inv_id} khớp số tiền: {credit_amount:,}đ."
            )

            record = ReconciliationLedgerRecord(
                trans_id=trans_id,
                matched_invoice_ids=[inv_id],
                customer_id=matched_cust_id,
                customer_name=matched_cust_name,
                bank_amount=credit_amount,
                invoiced_total=invoiced_total,
                discount_amount=0,
                bank_fee_amount=0,
                discrepancy_amount=0,
                resolution_source=ResolutionSource.SYSTEM_1_RULE,
                match_type=MatchType.FUZZY_NAME_1_TO_1,
                journal_entries=journal_entries,
                agent_reasoning_trace=f"[System 1 Deterministic Rule 2] Khớp mờ tên công ty {best_score:.0f}%, tìm thấy đúng 1 hóa đơn khớp tiền.",
                explanation=explanation,
                approved=0,
            )

            return record, None

        elif len(amount_matches) > 1:
            return None, (
                f"Công ty '{matched_cust_name}' có nhiều hơn 1 hóa đơn ({len(amount_matches)} HĐ) "
                f"có cùng số tiền {credit_amount:,}đ. Cần System 2 thẩm tra ngày đến hạn và ngữ cảnh."
            )
        else:
            return None, (
                f"Công ty '{matched_cust_name}' có {len(unpaid_invoices)} hóa đơn chưa thanh toán, "
                f"nhưng không có hóa đơn đơn lẻ nào khớp số tiền {credit_amount:,}đ. "
                f"Cần System 2 kiểm tra chuyển gộp (Subset Sum) hoặc chiết khấu."
            )

    # -----------------------------------------------------------------------
    # JOURNAL ENTRY GENERATION (Thông tư 200/133)
    # -----------------------------------------------------------------------

    def _generate_journal_entries(
        self,
        credit_amount: int,
        invoiced_total: int,
        discount_amount: int,
        bank_fee_amount: int,
        matched_inv_ids: List[str],
        customer_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate double-entry bookkeeping records compliant with TT 200/133:
          Nợ TK 1121: Tiền gửi ngân hàng VNĐ (credit_amount)
          Nợ TK 521:  Chiết khấu thương mại (discount_amount, if any)
          Nợ TK 642:  Chi phí quản lý DN / Phí ngân hàng (bank_fee_amount, if any)
          Có TK 131:  Phải thu khách hàng (invoiced_total)
        """
        inv_desc = ", ".join(matched_inv_ids)
        entries: List[Dict[str, Any]] = [
            {
                "entry_type": "DEBIT",
                "account_code": "1121",
                "account_name": "Tiền gửi ngân hàng VNĐ",
                "amount": credit_amount,
                "description": f"Thu tiền gửi ngân hàng thanh toán HĐ {inv_desc} ({customer_name})",
            }
        ]

        if discount_amount > 0:
            entries.append({
                "entry_type": "DEBIT",
                "account_code": "521",
                "account_name": "Chiết khấu thương mại",
                "amount": discount_amount,
                "description": f"Chiết khấu thương mại theo hợp đồng HĐ {inv_desc}",
            })

        if bank_fee_amount > 0:
            entries.append({
                "entry_type": "DEBIT",
                "account_code": "642",
                "account_name": "Chi phí quản lý DN / Phí ngân hàng",
                "amount": bank_fee_amount,
                "description": f"Phí chuyển tiền ngân hàng HĐ {inv_desc}",
            })

        entries.append({
            "entry_type": "CREDIT",
            "account_code": "131",
            "account_name": "Phải thu khách hàng",
            "amount": invoiced_total,
            "description": f"Giảm trừ công nợ phải thu khách hàng {customer_name} HĐ {inv_desc}",
        })

        return entries

    # -----------------------------------------------------------------------
    # DB PERSISTENCE
    # -----------------------------------------------------------------------

    def _apply_successful_match(
        self,
        db_manager: DatabaseManager,
        record: ReconciliationLedgerRecord,
    ) -> None:
        """Persist match in ledger, update invoices status, and update statement status."""
        # 1. Insert into reconciliation_ledger
        match_id = db_manager.insert_reconciliation_result(record.to_db_dict())
        logger.info(
            f"[System1] Match saved: match_id={match_id}, trans_id={record.trans_id}, "
            f"type={record.match_type}"
        )

        # 2. Update status of matched invoices
        for inv_id in record.matched_invoice_ids:
            # Mark invoice as paid
            inv_df = db_manager.fetch_df(
                "SELECT total_amount FROM invoices WHERE invoice_id = ?",
                (inv_id,),
            )
            if not inv_df.empty:
                total_val = int(inv_df.iloc[0]["total_amount"])
                db_manager.update_invoice_payment(
                    invoice_id=inv_id,
                    paid_amount=total_val,
                    status=InvoiceStatus.PAID,
                )

        # 3. Update bank statement status
        db_manager.update_reconcile_status(
            trans_id=record.trans_id,
            new_status=ReconcileStatus.AUTO_MATCHED_SYS1,
        )


# ===========================================================================
# ENTRYPOINT / VERIFICATION HELPER
# ===========================================================================

def run_system1_test() -> None:
    """Run System 1 against existing database."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    db = DatabaseManager()
    w2ner = W2NERStatementExtractor()
    engine = System1ReconciliationEngine()

    results = engine.run_fast_reconciliation(db, w2ner)
    print("\n" + "=" * 60)
    print("  KẾT QUẢ CHẠY SYSTEM 1 RECONCILIATION")
    print("=" * 60)
    print(f"  Tổng số dòng sao kê xử lý: {results['total_processed']}")
    print(f"  Số dòng khớp thành công:    {results['matched_count']}")
    print(f"  Số dòng chuyển System 2:   {results['exception_count']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_system1_test()

