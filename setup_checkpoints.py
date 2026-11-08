#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SETUP CHECKPOINTS & PRETRAINED MODELS FOR AGENTIC ACCOUNTING RECONCILIATION
=============================================================================
Script độc lập tự động kiểm tra môi trường, khởi tạo thư mục và tải về 
toàn bộ các model weights / checkpoints công khai:
  1. PhoBERT Backbone (vinai/phobert-base-v2) -> weights/phobert-base-v2/
  2. OCR Models:
     - VietOCR (vgg_transformer.pth)          -> weights/ocr/vgg_transformer.pth
     - PaddleOCR (PP-OCRv4 vi)                 -> cache hệ thống / ép tải về
  3. LLM Agent qua Ollama (qwen2.5:7b)        -> ollama pull
  4. Cấu trúc thư mục W2NER                   -> weights/w2ner/ (thư mục trống)

Chạy độc lập trên Windows PowerShell / Python 3.10+:
    python setup_checkpoints.py
    python setup_checkpoints.py --llm-model qwen2.5:7b
    python setup_checkpoints.py --skip-llm
=============================================================================
"""

import os
import sys
import time
import shutil
import urllib.request
import urllib.error
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Đường dẫn thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

DIR_PHOBERT = WEIGHTS_DIR / "phobert-base-v2"
DIR_OCR = WEIGHTS_DIR / "ocr"
DIR_W2NER = WEIGHTS_DIR / "w2ner"

VIETOCR_WEIGHT_FILE = DIR_OCR / "vgg_transformer.pth"

# Danh sách URLs tải VietOCR vgg_transformer.pth (Primary + Backup mirrors)
VIETOCR_URLS = [
    "https://vocr.vn/data/vietocr/vgg_transformer.pth",
    "https://huggingface.co/brian100/vietocr-vgg-transformer/resolve/main/vgg_transformer.pth",
]

# Định dạng ANSI màu hiển thị trên terminal Windows
class TermColor:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"


def log_step(step_num: int, title: str):
    print(f"\n{TermColor.BOLD}{TermColor.CYAN}=== [{step_num}/5] {title} ==={TermColor.RESET}")


def log_success(msg: str):
    print(f"{TermColor.GREEN}[OK] {msg}{TermColor.RESET}")


def log_warn(msg: str):
    print(f"{TermColor.YELLOW}[WARN] {msg}{TermColor.RESET}")


def log_error(msg: str):
    print(f"{TermColor.RED}[ERROR] {msg}{TermColor.RESET}")


def log_info(msg: str):
    print(f"{TermColor.MAGENTA}[INFO] {msg}{TermColor.RESET}")


def get_dir_size_mb(path: Path) -> float:
    """Tính tổng dung lượng thư mục theo MB."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def download_with_progress(url: str, dest_path: Path, min_expected_mb: float = 50.0) -> bool:
    """
    Tải file từ URL về dest_path kèm thanh hiển thị tiến trình (progress bar),
    hỗ trợ resume hoặc tải lại nếu file lỗi / chưa đủ dung lượng.
    """
    temp_path = dest_path.with_suffix(".tmp")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  -> Bắt đầu kết nối: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
            total_size = response.getheader("Content-Length")
            total_bytes = int(total_size) if total_size else 0

            downloaded = 0
            block_size = 1024 * 64  # 64 KB per chunk
            start_time = time.time()

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)

                # Hiển thị thanh tiến trình
                elapsed = max(time.time() - start_time, 0.001)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed

                if total_bytes > 0:
                    percent = downloaded * 100 / total_bytes
                    curr_mb = downloaded / (1024 * 1024)
                    tot_mb = total_bytes / (1024 * 1024)
                    bar_len = 30
                    filled = int(bar_len * downloaded // total_bytes)
                    bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
                    sys.stdout.write(
                        f"\r  [{bar}] {percent:5.1f}% | {curr_mb:6.1f}/{tot_mb:6.1f} MB | {speed_mb:5.2f} MB/s"
                    )
                else:
                    curr_mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r  -> Đã tải: {curr_mb:6.1f} MB | {speed_mb:5.2f} MB/s")
                sys.stdout.flush()

        print()  # Xuống dòng sau khi hoàn tất

        # Kiểm tra tính toàn vẹn dung lượng tối thiểu
        file_size_mb = temp_path.stat().st_size / (1024 * 1024)
        if file_size_mb < min_expected_mb:
            log_error(f"File tải về quá nhỏ ({file_size_mb:.2f} MB < {min_expected_mb} MB). Đang xóa file tạm.")
            if temp_path.exists():
                temp_path.unlink()
            return False

        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        return True

    except Exception as exc:
        print()
        log_warn(f"Lỗi khi tải từ {url}: {exc}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return False


# =============================================================================
# BƯỚC 1: KHỞI TẠO CẤU TRÚC THƯ MỤC LƯU TRỮ
# =============================================================================
def init_directory_structure() -> Dict[str, Path]:
    """Tạo cấu trúc thư mục weights/ phục vụ lưu trữ mô hình."""
    log_step(1, "Khởi tạo Cấu trúc Thư mục weights/")
    
    dirs = {
        "phobert": DIR_PHOBERT,
        "ocr": DIR_OCR,
        "w2ner": DIR_W2NER,
    }

    for name, p in dirs.items():
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            log_info(f"Đã tạo mới thư mục: {p.relative_to(BASE_DIR)}")
        else:
            log_success(f"Thư mục đã sẵn sàng: {p.relative_to(BASE_DIR)}")

    log_info("Thư mục weights/w2ner/ được giữ nguyên trống (chờ người dùng chép file .pt tùy biến vào sau).")
    return dirs


# =============================================================================
# BƯỚC 2: TẢI BACKBONE PHOBERT (vinai/phobert-base-v2)
# =============================================================================
def setup_phobert_backbone(model_name: str = "vinai/phobert-base-v2") -> Tuple[bool, str]:
    """
    Kiểm tra và tải model vinai/phobert-base-v2 từ Hugging Face
    vào weights/phobert-base-v2/ để phục vụ chạy offline hoàn toàn.
    """
    log_step(2, f"Kiểm tra & Tải PhoBERT Backbone: {model_name}")

    # Kiểm tra các file tối thiểu bắt buộc
    config_file = DIR_PHOBERT / "config.json"
    weights_bin = DIR_PHOBERT / "pytorch_model.bin"
    weights_safe = DIR_PHOBERT / "model.safetensors"

    has_weights = weights_bin.exists() or weights_safe.exists()
    has_config = config_file.exists()

    if has_config and has_weights:
        size_mb = get_dir_size_mb(DIR_PHOBERT)
        if size_mb > 300.0:
            log_success(f"PhoBERT backbone đã tồn tại đầy đủ ({size_mb:.1f} MB) tại {DIR_PHOBERT.relative_to(BASE_DIR)}")
            return True, f"Có sẵn ({size_mb:.1f} MB)"

    log_info(f"Đang tải {model_name} từ Hugging Face về {DIR_PHOBERT.relative_to(BASE_DIR)}...")

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        log_error("Chưa cài đặt thư viện 'transformers'. Hãy chạy: pip install transformers")
        return False, "Thiếu thư viện transformers"

    try:
        print("  -> Tải Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(DIR_PHOBERT))

        print("  -> Tải Model Architecture & Weights...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(str(DIR_PHOBERT))

        size_mb = get_dir_size_mb(DIR_PHOBERT)
        log_success(f"Đã lưu thành công PhoBERT vào {DIR_PHOBERT.relative_to(BASE_DIR)} ({size_mb:.1f} MB)")
        return True, f"Đã tải ({size_mb:.1f} MB)"

    except Exception as exc:
        log_error(f"Tải PhoBERT thất bại: {exc}")
        return False, f"Thất bại ({exc})"


# =============================================================================
# BƯỚC 3: TẢI CHECKPOINT NHÁNH OCR (VietOCR & PaddleOCR)
# =============================================================================
def setup_vietocr() -> Tuple[bool, str]:
    """Kiểm tra và tải vgg_transformer.pth cho VietOCR."""
    print(f"\n--- [3.1] VietOCR (vgg_transformer.pth) ---")
    
    if VIETOCR_WEIGHT_FILE.exists():
        size_mb = VIETOCR_WEIGHT_FILE.stat().st_size / (1024 * 1024)
        if size_mb >= 50.0:
            log_success(f"VietOCR weight đã sẵn sàng: {VIETOCR_WEIGHT_FILE.relative_to(BASE_DIR)} ({size_mb:.1f} MB)")
            return True, f"Có sẵn ({size_mb:.1f} MB)"
        else:
            log_warn(f"File hiện tại bị hỏng hoặc chưa tải hết ({size_mb:.1f} MB). Đang tiến hành tải lại...")
            VIETOCR_WEIGHT_FILE.unlink()

    # Thử tải lần lượt qua các URLs
    downloaded = False
    for url in VIETOCR_URLS:
        log_info(f"Đang thử tải từ: {url}")
        if download_with_progress(url, VIETOCR_WEIGHT_FILE, min_expected_mb=50.0):
            downloaded = True
            break

    if downloaded:
        size_mb = VIETOCR_WEIGHT_FILE.stat().st_size / (1024 * 1024)
        log_success(f"Tải thành công VietOCR weight ({size_mb:.1f} MB) tại {VIETOCR_WEIGHT_FILE.relative_to(BASE_DIR)}")
        return True, f"Đã tải ({size_mb:.1f} MB)"
    else:
        log_error(f"Không thể tự động tải VietOCR weights từ các mirrors. Bạn có thể tải thủ công file 'vgg_transformer.pth' bỏ vào {DIR_OCR.relative_to(BASE_DIR)}")
        return False, "Không thể tải file"


def setup_paddleocr() -> Tuple[bool, str]:
    """Kiểm tra và kích hoạt tải model Text Detection PaddleOCR (ch_PP-OCRv4_det)."""
    print(f"\n--- [3.2] PaddleOCR (Text Detection ch_PP-OCRv4) ---")

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        log_warn("Thư viện 'paddleocr' chưa được cài đặt trong môi trường Python.")
        log_info("Bạn có thể cài đặt thêm bằng lệnh: pip install paddlepaddle paddleocr")
        return False, "Chưa cài paddleocr"

    try:
        log_info("Đang khởi tạo PaddleOCR(lang='vi') để kiểm tra / tải tự động weights phát hiện chữ...")
        # Khởi tạo sẽ tự động tải các models PP-OCRv4 về cache hệ thống (~/.paddleocr/)
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
        log_success("PaddleOCR đã được nạp và tải đầy đủ model weights (Detection & Recognition).")
        return True, "Sẵn sàng (Đã nạp)"
    except Exception as exc:
        log_warn(f"Khởi tạo PaddleOCR gặp lỗi: {exc}")
        return False, f"Lỗi ({exc})"


# =============================================================================
# BƯỚC 4: KIỂM TRA & TẢI LLM MODEL QUA OLLAMA
# =============================================================================
def check_ollama_server() -> bool:
    """Kiểm tra xem Ollama server có đang chạy tại http://localhost:11434 không."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def setup_ollama_llm(model_name: str = "qwen2.5:7b") -> Tuple[bool, str]:
    """
    Kiểm tra dịch vụ Ollama và tự động kéo model LLM (mặc định qwen2.5:7b)
    về máy nếu chưa có.
    """
    log_step(4, f"Kiểm tra & Tải LLM qua Ollama: {model_name}")

    # 1. Kiểm tra lệnh 'ollama' có trong PATH không
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        log_error("Không tìm thấy lệnh 'ollama' trong biến môi trường PATH.")
        log_info("Vui lòng cài đặt Ollama từ trang chủ https://ollama.com/")
        return False, "Chưa cài Ollama CLI"

    # 2. Kiểm tra Ollama Daemon/Server
    is_server_up = check_ollama_server()
    if not is_server_up:
        log_warn("Ollama server chưa khởi chạy tại http://localhost:11434.")
        log_info("Đang kiểm tra danh sách model qua lệnh CLI 'ollama list'...")

    # 3. Kiểm tra xem model đã tồn tại trong Ollama chưa
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            model_base = model_name.split(":")[0]
            already_pulled = any(model_name in line or model_base in line for line in lines[1:])
            if already_pulled:
                log_success(f"Model Ollama '{model_name}' đã được tải sẵn trên máy.")
                return True, "Đã có sẵn trên Ollama"
        else:
            if "could not connect" in result.stderr.lower():
                log_warn("Không thể kết nối đến Ollama daemon.")
                log_info("Gợi ý: Mở ứng dụng Ollama trên Windows hoặc chạy lệnh 'ollama serve' trong một terminal riêng.")
                return False, "Ollama daemon chưa bật"
    except Exception as exc:
        log_warn(f"Không thể kiểm tra 'ollama list': {exc}")

    # 4. Kích hoạt 'ollama pull' hiển thị trực tiếp tiến trình
    log_info(f"Bắt đầu tải model '{model_name}' từ Ollama Library (tiến trình sẽ hiển thị bên dưới)...")
    try:
        cmd = ["ollama", "pull", model_name]
        ret = subprocess.run(cmd, check=True)
        if ret.returncode == 0:
            log_success(f"Tải thành công model Ollama '{model_name}'!")
            return True, "Đã tải xong"
        else:
            log_error(f"Lệnh 'ollama pull {model_name}' kết thúc với mã lỗi {ret.returncode}")
            return False, f"Exit code {ret.returncode}"
    except subprocess.CalledProcessError as exc:
        log_error(f"Lỗi khi thực hiện 'ollama pull': {exc}")
        return False, "Pull thất bại"
    except Exception as exc:
        log_error(f"Lỗi không xác định: {exc}")
        return False, str(exc)


# =============================================================================
# BƯỚC 5: TỔNG KẾT HỆ THỐNG (SYSTEM SUMMARY TABLE)
# =============================================================================
def print_summary_table(results: Dict[str, Tuple[bool, str]]):
    """In bảng tổng kết trạng thái tất cả các thành phần ra terminal."""
    log_step(5, "Bảng Tổng kết Trạng thái Checkpoints Hệ thống")

    col_w_name = 38
    col_w_status = 12
    col_w_detail = 36

    border = "+" + "-" * (col_w_name + 2) + "+" + "-" * (col_w_status + 2) + "+" + "-" * (col_w_detail + 2) + "+"
    header = f"| {'Thành phần':<{col_w_name}} | {'Trạng thái':<{col_w_status}} | {'Chi tiết / Đường dẫn':<{col_w_detail}} |"

    print(border)
    print(header)
    print(border)

    for item_name, (ok, detail) in results.items():
        if ok:
            status_text = f"{TermColor.GREEN}[OK]{TermColor.RESET}"
            display_status = "[OK]"
        else:
            if "INFO" in item_name or "W2NER" in item_name:
                status_text = f"{TermColor.MAGENTA}[INFO]{TermColor.RESET}"
                display_status = "[INFO]"
            else:
                status_text = f"{TermColor.YELLOW}[CHƯA SẴN SÀNG]{TermColor.RESET}"
                display_status = "[CHƯA CÓ]"

        # Format dòng
        line = f"| {item_name:<{col_w_name}} | {display_status:<{col_w_status}} | {detail:<{col_w_detail}} |"
        print(line)

    print(border)
    print(f"\n{TermColor.BOLD}Ghi chú quan trọng:{TermColor.RESET}")
    print(f"  • Thư mục weights/w2ner/ đã được tạo sẵn để nhận file .pt tùy biến của bạn.")
    print(f"  • Nếu chưa cài đặt PaddleOCR hoặc Ollama daemon chưa bật, bạn có thể chạy lại script sau khi cấu hình.")
    print(f"{TermColor.GREEN}Toàn bộ quy trình kiểm tra và chuẩn bị checkpoints hoàn tất!{TermColor.RESET}\n")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================
def main():
    # Hỗ trợ bật hiển thị màu ANSI trên Windows terminal cũ
    if os.name == "nt":
        os.system("")

    parser = argparse.ArgumentParser(
        description="Auto-download and setup all public checkpoints for Agentic Accounting project."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen2.5:7b",
        help="Tên model LLM trên Ollama (mặc định: qwen2.5:7b)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Bỏ qua bước kiểm tra và pull model LLM trên Ollama",
    )
    parser.add_argument(
        "--skip-phobert",
        action="store_true",
        help="Bỏ qua bước tải PhoBERT backbone",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Bỏ qua bước tải OCR checkpoints",
    )
    args = parser.parse_args()

    print(f"{TermColor.BOLD}{TermColor.MAGENTA}" + "=" * 70)
    print("  AGENTIC ACCOUNTING SYSTEM — SETUP CHECKPOINTS & PRETRAINED WEIGHTS")
    print("=" * 70 + f"{TermColor.RESET}")
    print(f"Thư mục gốc: {BASE_DIR}")
    print(f"Thư mục lưu trữ: {WEIGHTS_DIR}")

    summary_results: Dict[str, Tuple[bool, str]] = {}

    # Bước 1: Khởi tạo thư mục
    init_directory_structure()

    # Bước 2: PhoBERT Backbone
    if not args.skip_phobert:
        ok_phobert, detail_phobert = setup_phobert_backbone()
        summary_results["Backbone PhoBERT (base-v2)"] = (ok_phobert, detail_phobert)
    else:
        summary_results["Backbone PhoBERT (base-v2)"] = (False, "Bỏ qua qua cờ --skip-phobert")

    # Bước 3: OCR Checkpoints
    if not args.skip_ocr:
        log_step(3, "Kiểm tra & Tải Checkpoint Nhánh OCR")
        ok_vietocr, detail_vietocr = setup_vietocr()
        summary_results["VietOCR (vgg_transformer.pth)"] = (ok_vietocr, detail_vietocr)

        ok_paddle, detail_paddle = setup_paddleocr()
        summary_results["PaddleOCR (PP-OCRv4 detection)"] = (ok_paddle, detail_paddle)
    else:
        summary_results["VietOCR (vgg_transformer.pth)"] = (False, "Bỏ qua qua cờ --skip-ocr")
        summary_results["PaddleOCR (PP-OCRv4 detection)"] = (False, "Bỏ qua qua cờ --skip-ocr")

    # Bước 4: LLM Checkpoint qua Ollama
    if not args.skip_llm:
        ok_llm, detail_llm = setup_ollama_llm(model_name=args.llm_model)
        summary_results[f"Agent LLM (Ollama {args.llm_model})"] = (ok_llm, detail_llm)
    else:
        summary_results[f"Agent LLM (Ollama {args.llm_model})"] = (False, "Bỏ qua qua cờ --skip-llm")

    # W2NER Custom Folder Info
    w2ner_files = list(DIR_W2NER.glob("*.pt"))
    if w2ner_files:
        w2ner_detail = f"Đã có file: {w2ner_files[0].name}"
        summary_results["W2NER Custom Model (.pt)"] = (True, w2ner_detail)
    else:
        summary_results["W2NER Custom Model (.pt)"] = (True, "Thư mục trống (chờ bạn bỏ file vào)")

    # Bước 5: Bảng tổng kết
    print_summary_table(summary_results)


if __name__ == "__main__":
    main()

