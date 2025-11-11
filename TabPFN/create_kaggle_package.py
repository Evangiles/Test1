"""Create Kaggle Dataset package with TabPFN v2.5 weights."""
import shutil
from pathlib import Path
from tabpfn.model_loading import get_cache_dir

# 캐시 위치
cache_dir = Path(get_cache_dir())
print(f"모델 캐시: {cache_dir}")

# Kaggle 업로드용 패키지 폴더
pkg_dir = Path("C:/Users/jrjin/Desktop/tabpfn_kaggle_package")
pkg_dir.mkdir(exist_ok=True)

# 1) 가중치 복사
weights_dir = pkg_dir / "model_weights"
if weights_dir.exists():
    shutil.rmtree(weights_dir)
shutil.copytree(cache_dir, weights_dir)
print(f"✓ 가중치 복사 완료: {weights_dir}")

# 2) 소스 코드 복사
src_base = Path("TabPFN/src")
for pkg in ["tabpfn", "tabpfn_common_utils"]:
    dst = pkg_dir / pkg
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_base / pkg, dst)
    print(f"✓ {pkg} 복사 완료")

# 3) 유틸 파일 복사
utils = ["eval.py", "metric.py", "cv.py", "preprocessing_utils.py",
         "models_baselines.py", "models_tabpfn.py", "position_mapping.py"]
for util in utils:
    shutil.copy(f"TabPFN/{util}", pkg_dir / util)
print(f"✓ 유틸 파일 {len(utils)}개 복사 완료")

# 4) README 생성
readme = pkg_dir / "README.md"
readme.write_text("""# TabPFN v2.5 Kaggle Package

## Contents
- `model_weights/`: Pre-downloaded TabPFN v2.5 weights
- `tabpfn/`: TabPFN source code
- `tabpfn_common_utils/`: Telemetry stubs
- `*.py`: Evaluation utilities

## Usage in Kaggle
```python
import sys, os, shutil
from pathlib import Path

# Copy package
pkg = Path('/kaggle/input/tabpfn-v25-package')
work = Path('/kaggle/working')

shutil.copytree(pkg / 'tabpfn', work / 'tabpfn')
shutil.copytree(pkg / 'tabpfn_common_utils', work / 'tabpfn_common_utils')
shutil.copytree(pkg / 'model_weights', work / '.cache/tabpfn')

for f in ['eval.py', 'metric.py', 'cv.py', 'preprocessing_utils.py',
          'models_baselines.py', 'models_tabpfn.py', 'position_mapping.py']:
    shutil.copy(pkg / f, work / f)

# Set cache path
os.environ['TABPFN_MODEL_CACHE_DIR'] = str(work / '.cache/tabpfn')
sys.path.insert(0, str(work))

# Use TabPFN
from tabpfn import TabPFNRegressor
model = TabPFNRegressor(device='cpu')
```

## License
TabPFN v2.5: tabpfn-2.5-license-v1.0 (non-commercial)
""", encoding='utf-8')
print(f"✓ README 생성 완료")

# 크기 확인
total_size = sum(f.stat().st_size for f in pkg_dir.rglob('*') if f.is_file())
print(f"\n패키지 생성 완료!")
print(f"위치: {pkg_dir}")
print(f"크기: {total_size / 1e9:.2f} GB")
print(f"파일 수: {len(list(pkg_dir.rglob('*')))}")
print("\n다음 단계:")
print("1. Kaggle.com → Datasets → New Dataset")
print("2. 제목: tabpfn-v25-package")
print(f"3. 폴더 업로드: {pkg_dir}")
print("4. Private 설정 → Create")

