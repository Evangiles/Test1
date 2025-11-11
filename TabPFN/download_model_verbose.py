"""Download TabPFN v2.5 with verbose output."""
import os
from pathlib import Path

# 캐시 디렉토리
cache_dir = Path("C:/Users/jrjin/Desktop/TabPFN2.5/tabpfn_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['TABPFN_MODEL_CACHE_DIR'] = str(cache_dir)

print(f"캐시 디렉토리: {cache_dir}")
print("HuggingFace 로그인 확인...")

# HuggingFace 로그인 확인
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f"✓ 로그인됨: {user['name']}")
except Exception as e:
    print(f"✗ 로그인 실패: {e}")
    print("먼저 'huggingface-cli login' 실행 필요")
    exit(1)

# 라이선스 동의 확인
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info("Prior-Labs/tabpfn_2_5")
    print(f"✓ 모델 접근 가능: {info.modelId}")
except Exception as e:
    print(f"✗ 모델 접근 불가: {e}")
    print("https://huggingface.co/Prior-Labs/tabpfn_2_5 에서 라이선스 동의 필요!")
    exit(1)

# TabPFN 다운로드 (verbose)
print("\n모델 다운로드 시작...")
from tabpfn import TabPFNRegressor

try:
    model = TabPFNRegressor(
        model_path="auto",  # 자동 다운로드
        device="cpu",
    )
    print("✓ TabPFNRegressor 생성 완료!")
except Exception as e:
    print(f"✗ 다운로드 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 캐시 확인
print(f"\n캐시 내용:")
if cache_dir.exists():
    for item in cache_dir.rglob('*'):
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.relative_to(cache_dir)}: {size_mb:.2f} MB")
    
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    print(f"\n총 크기: {total_size / 1e6:.2f} MB")
    
    if total_size < 1e6:
        print("⚠️ 파일이 너무 작음 - 다운로드 실패 가능성!")
else:
    print("✗ 캐시 디렉토리가 생성되지 않음")

print("\n완료!")

