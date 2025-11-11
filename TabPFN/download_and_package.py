"""Download TabPFN v2.5 and create Kaggle package."""
import os
import shutil
from pathlib import Path

# 로컬 캐시 디렉토리 지정
cache_dir = Path("C:/Users/jrjin/Desktop/TabPFN2.5/tabpfn_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['TABPFN_MODEL_CACHE_DIR'] = str(cache_dir)

print(f"캐시 디렉토리: {cache_dir}")

# TabPFN 다운로드
from tabpfn import TabPFNRegressor
print("다운로드 시작...")
model = TabPFNRegressor()
print("✓ 다운로드 완료!")

# Kaggle 패키지 생성
pkg_dir = Path("C:/Users/jrjin/Desktop/tabpfn_kaggle_package")
if pkg_dir.exists():
    shutil.rmtree(pkg_dir)
pkg_dir.mkdir()

# 1) 가중치
shutil.copytree(cache_dir, pkg_dir / "model_weights")
print(f"✓ 가중치 복사: {pkg_dir / 'model_weights'}")

# 2) 소스 코드
src_base = Path("TabPFN/src")
for pkg in ["tabpfn", "tabpfn_common_utils"]:
    shutil.copytree(src_base / pkg, pkg_dir / pkg)
    print(f"✓ {pkg} 복사 완료")

# 3) 유틸 파일
utils = ["eval.py", "metric.py", "cv.py", "preprocessing_utils.py",
         "models_baselines.py", "models_tabpfn.py", "position_mapping.py"]
for util in utils:
    shutil.copy(f"TabPFN/{util}", pkg_dir / util)
print(f"✓ 유틸 {len(utils)}개 복사 완료")

# 4) README
readme = pkg_dir / "README.md"
readme.write_text("""# TabPFN v2.5 Kaggle Package

## Usage in Kaggle Notebook

```python
import sys, os, shutil
from pathlib import Path

pkg = Path('/kaggle/input/tabpfn-v25-package')
work = Path('/kaggle/working')

# Copy all files
shutil.copytree(pkg / 'tabpfn', work / 'tabpfn')
shutil.copytree(pkg / 'tabpfn_common_utils', work / 'tabpfn_common_utils')
shutil.copytree(pkg / 'model_weights', work / '.cache/tabpfn')

for f in pkg.glob('*.py'):
    shutil.copy(f, work / f.name)

# Setup environment
os.environ['TABPFN_MODEL_CACHE_DIR'] = str(work / '.cache/tabpfn')
sys.path.insert(0, str(work))

# Use TabPFN
from tabpfn import TabPFNRegressor
import pandas as pd
import eval as tp_eval

df = pd.read_csv('path/to/train.csv')
results = tp_eval.evaluate_models_with_cv(df, include_tabpfn=True)
```

## License
TabPFN v2.5: Non-commercial license (research/evaluation only)
""", encoding='utf-8')

# 크기 계산
total_size = sum(f.stat().st_size for f in pkg_dir.rglob('*') if f.is_file())
print(f"\n{'='*50}")
print(f"패키지 생성 완료!")
print(f"위치: {pkg_dir}")
print(f"크기: {total_size / 1e9:.2f} GB")
print(f"파일 수: {len(list(pkg_dir.rglob('*')))}")
print(f"{'='*50}")
print("\n다음 단계:")
print("1. https://www.kaggle.com/datasets 접속")
print("2. 'New Dataset' 클릭")
print("3. 제목: tabpfn-v25-package")
print(f"4. 폴더 업로드: {pkg_dir}")
print("5. Private 설정 → Create")
print("\n업로드 완료 후 Kaggle 노트북에서 위 README 코드 실행!")

