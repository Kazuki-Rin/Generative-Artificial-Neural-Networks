#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ASUNA_LAUNCHER")

def setup_environment():
    """
    [BẢN VÁ P0 - CHỐNG TREO MÁY]: Tối ưu hóa CPU cho 11 tiến trình PyTorch song song.
    Nếu không giới hạn, 11 mô hình PyTorch sẽ sinh ra hàng nghìn luồng OpenMP gây treo cứng hệ thống.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def verify_runtime_environment():
    """
    [BẢN VÁ P0]: Kiểm tra interpreter đang chạy đúng môi trường chính (kazuki_env) và đủ thư viện cốt lõi.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.normcase(os.path.abspath(os.path.join(base_dir, ".venv")))
    curr_prefix = os.path.normcase(os.path.abspath(sys.prefix))
    env_name = os.path.basename(curr_prefix).lower()

    is_kazuki_env = (env_name == "kazuki_env")
    is_project_venv = os.path.exists(venv_dir) and (curr_prefix == venv_dir)

    if not (is_kazuki_env or is_project_venv):
        logger.critical(
            f"💀 [FAIL-CLOSED] Sai môi trường thực thi: Interpreter hiện tại ({sys.executable}) "
            f"không thuộc môi trường chính 'kazuki_env'. Vui lòng kích hoạt 'kazuki_env' trước khi chạy!"
        )
        raise SystemExit(1)

    for required_mod in ("torch", "neat", "zmq"):
        try:
            __import__(required_mod)
        except ImportError as e:
            logger.critical(
                f"💀 [FAIL-CLOSED] Môi trường 'kazuki_env' thiếu thư viện bắt buộc '{required_mod}' ({e}). "
                f"Vui lòng cài đặt đầy đủ thư viện trước khi khởi động!"
            )
            raise SystemExit(1)

def run_pre_flight_audit():
    """
    [BẢN VÁ P0]: Chạy kiểm tra toàn vẹn Checkpoint ở chế độ chỉ đọc (Fail-closed).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audit_path = os.path.join(base_dir, "migration_and_repair.py")
    if not os.path.exists(audit_path):
        logger.critical("💀 [FAIL-CLOSED] Không tìm thấy migration_and_repair.py. Từ chối khởi động khi chưa xác minh Checkpoint!")
        raise SystemExit(1)

    logger.info("🔍 Đang thực hiện Audit Checkpoints (Dry-run Read-only Mode)...")
    try:
        subprocess.run([sys.executable, audit_path], cwd=base_dir, check=True)
    except subprocess.CalledProcessError:
        logger.critical(
            "💀 [FAIL-CLOSED] Pre-flight audit THẤT BẠI. Hệ thống phát hiện sai lệch Checkpoint hoặc Manifest.\n"
            "👉 HƯỚNG DẪN AN TOÀN: Không ghi đè checkpoint mù quáng! Hãy sao lưu toàn bộ thư mục 'models/', "
            "'brain_manifest.json' và 'models/M8_Memory/chroma_db/', kiểm tra kỹ báo cáo chẩn đoán rồi mới chạy "
            "quy trình phục hồi đã xác minh."
        )
        raise SystemExit(1)

def boot_asuna():
    logger.info("======================================================")
    logger.info("🌸 KHỞI ĐỘNG HỆ ĐIỀU HÀNH NHẬN THỨC ASUNA (AGI V1.0) 🌸")
    logger.info("======================================================")
    
    # 1. Khởi tạo môi trường an toàn cho đa tiến trình và xác minh .venv
    setup_environment()
    verify_runtime_environment()
    
    # 2. Kiểm tra tính toàn vẹn của Dữ liệu (Checkpoints - Fail-closed)
    run_pre_flight_audit()

    logger.info("⚠️ Đã giao toàn quyền quản lý vòng đời cho MasterSupervisor!")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_loop_path = os.path.join(base_dir, "main_life_loop.py")

    if not os.path.exists(main_loop_path):
        logger.critical("❌ CẢNH BÁO TỬ THẦN: Không tìm thấy main_life_loop.py! Hủy khởi động.")
        raise SystemExit(1)

    process = None
    ret_code = 0
    try:
        process = subprocess.Popen([sys.executable, main_loop_path], cwd=base_dir)
        ret_code = process.wait()
    except KeyboardInterrupt:
        logger.info("\n🛑 Nhận lệnh ngắt khẩn cấp. Đang tắt toàn bộ hệ thống Asuna...")
        if process is not None and process.poll() is None:
            process.terminate() 
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Tiến trình từ chối tắt. Đang ép buộc tiêu diệt (Kill)...")
                process.kill()
                process.wait()
    finally:
        if ret_code not in (0, None):
            logger.critical(f"💀 Hệ thống Asuna dừng bất thường với mã thoát: {ret_code}")
            raise SystemExit(ret_code)
        logger.info("✅ Asuna đã đi vào giấc ngủ an toàn.")

if __name__ == "__main__":
    boot_asuna()