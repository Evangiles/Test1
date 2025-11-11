"""Force TabPFN v2.5 model download by actually using it."""
import os
import numpy as np
from pathlib import Path
from sklearn.datasets import make_regression

# 캐시 디렉토리
cache_dir = Path("C:/Users/jrjin/Desktop/TabPFN2.5/tabpfn_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['TABPFN_MODEL_CACHE_DIR'] = str(cache_dir)

print(f"캐시 디렉토리: {cache_dir}")

# TabPFN 생성
from tabpfn import TabPFNRegressor
print("TabPFNRegressor 생성 중...")
model = TabPFNRegressor(device="cpu", n_estimators=1)

# 더미 데이터로 fit 호출 (강제 다운로드)
print("더미 데이터로 fit 호출 (모델 다운로드 강제)...")
X, y = make_regression(n_samples=100, n_features=10, random_state=42)
model.fit(X[:50], y[:50])

print("✓ fit 완료!")

# predict 호출
predictions = model.predict(X[50:])
print(f"✓ predict 완료 (샘플 예측값: {predictions[:3]})")

# 캐시 확인
print(f"\n캐시 내용 확인:")
if cache_dir.exists():
    files = list(cache_dir.rglob('*'))
    file_list = [f for f in files if f.is_file()]
    
    for f in file_list:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(cache_dir)}: {size_mb:.2f} MB")
    
    total_size = sum(f.stat().st_size for f in file_list)
    print(f"\n총 크기: {total_size / 1e9:.2f} GB")
    print(f"파일 수: {len(file_list)}")
    
    if total_size > 1e6:
        print("\n✓ 모델 다운로드 성공!")
    else:
        print("\n⚠️ 파일이 너무 작음 - 다운로드 실패 가능성")
else:
    print("✗ 캐시 디렉토리가 없음")

