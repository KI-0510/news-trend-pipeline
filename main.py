import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


KST = timezone(timedelta(hours=9))

def log(msg: str):
    print(f"[MAIN] {msg}", flush=True)

def run_step(name: str, cmd: list[str], env: dict):
    log(f"=== [{name}] 실행: {' '.join(cmd)} ===")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"[중단] 단계 실패: {name}")
    log(f"[완료] {name}")

def load_env_file():
    if load_dotenv is None:
        return
    candidates = [Path(".env"), Path(__file__).resolve().parent / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p)
            log(f".env 로드: {p}")
            break

def merge_config_env(env: dict):
    cfg_path = Path("config.json")
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def pick(key_env: str, key_cfg: str, default=None):
        if env.get(key_env) is not None:
            return env[key_env]
        if key_cfg in cfg and cfg[key_cfg] is not None:
            return str(cfg[key_cfg]).lower() if isinstance(cfg[key_cfg], bool) else str(cfg[key_cfg])
        return default

    env.setdefault("DRY_RUN", pick("DRY_RUN", "dry_run", "false"))
    env.setdefault("USE_PRO", pick("USE_PRO", "use_pro", "false"))
    env.setdefault("TZ", "Asia/Seoul")
    return env

def require_key(env: dict, name: str):
    v = env.get(name) or os.getenv(name)
    if not v:
        raise SystemExit(f"[중단] 필수 키 누락: {name}")
    return v

def do_archive_daily(out_base="outputs"):
    date_kst = datetime.now(KST).strftime("%Y-%m-%d")
    time_kst = datetime.now(KST).strftime("%H%M-KST")
    outdir = os.path.join(out_base, "daily", date_kst, time_kst)
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "export"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "debug"), exist_ok=True)

    def cp_glob(src_glob, dst_dir):
        import glob
        for p in glob.glob(src_glob):
            try:
                shutil.copy(p, dst_dir)
            except Exception:
                pass

    for pat in ["outputs/*.json", "outputs/*.html", "outputs/*.md"]:
        cp_glob(pat, outdir)

    for sub in ["export", "fig", "debug"]:
        src = os.path.join("outputs", sub)
        if os.path.isdir(src):
            for root, _, files in os.walk(src):
                for f in files:
                    srcp = os.path.join(root, f)
                    rel = os.path.relpath(srcp, src)
                    dstp = os.path.join(outdir, sub, rel)
                    os.makedirs(os.path.dirname(dstp), exist_ok=True)
                    try:
                        shutil.copy(srcp, dstp)
                    except Exception:
                        pass

    log(f"[아카이브 완료] {outdir}")

def build_steps():
    PY = sys.executable
    return [
        ("a",        [PY, "-m", "src.module_a"]),
        ("check_a",  [PY, "-m", "src.check_a"]),
        ("wh",       [PY, "-m", "src.warehouse_append"]),
        ("body",     [PY, "-m", "scripts.fetch_article_bodies"]),
        ("b",        [PY, "-m", "src.module_b"]),
        ("check_b",  [PY, "-m", "src.check_b"]),
        ("c",        [PY, "-m", "src.module_c"]),
        ("check_c",  [PY, "-m", "src.check_c"]),
        ("export",   [PY, "-m", "scripts.signal_export"]),
        ("future",   [PY, "-m", "scripts.future_insights"]),
        ("d",        [PY, "-m", "src.module_d"]),
        ("check_d",  [PY, "-m", "src.check_d"]),
        ("e",        [PY, "-m", "src.module_e"]),
        ("check_e",  [PY, "-m", "src.check_e"]),
        ("gen_visual",[PY, "-m", "scripts.generate_visuals"]),
        ("preflight",[PY, "-m", "scripts.preflight"]),
        ("f",        [PY, "-m", "src.module_f"]),
        ("check_f",  [PY, "-m", "src.check_f"]),
    ]

def main():
    parser = argparse.ArgumentParser(description="Local runner (GitHub Actions-like)")
    parser.add_argument("--dry-run", choices=["true", "false"], help="드라이런 실행 여부")
    parser.add_argument("--pro-mode", choices=["true", "false"], help="Pro 모드")
    parser.add_argument("--body-min-len", type=int, default=None, help="본문 최소 길이")
    parser.add_argument("--only", nargs="*", help="선택 실행 단계: a check_a wh body b check_b c check_c export d check_d e check_e gen_visual preflight f check_f")
    parser.add_argument("--archive-daily", action="store_true", help="outputs를 날짜/시간 폴더로 아카이브")
    args = parser.parse_args()

    load_env_file()

    env = os.environ.copy()
    if args.dry_run is not None:
        env["DRY_RUN"] = args.dry_run
    if args.pro_mode is not None:
        env["USE_PRO"] = args.pro_mode
    if args.body_min_len is not None:
        env["BODY_MIN_LEN"] = str(args.body_min_len)

    env = merge_config_env(env)

    steps = build_steps()
    if args.only:
        wanted = {s.lower() for s in args.only}
        steps = [s for s in steps if s[0] in wanted]
        if not steps:
            raise SystemExit("[중단] --only에 해당하는 단계가 없습니다.")

    will_run = {name for name, _ in steps}
    if "c" in will_run or "d" in will_run or "e" in will_run:
        require_key(env, "GEMINI_API_KEY")

    last_ok = None
    for name, cmd in steps:
        run_step(name, cmd, env)
        last_ok = name

    log(f"[성공] 전체 완료. 마지막 성공 단계: {last_ok}")

    if args.archive_daily:
        do_archive_daily()

if __name__ == "__main__":
    main()