import os
import importlib.util

# 使用 importlib 查找包路径，避免执行 import PyNvVideoCodec 触发驱动检查
spec_info = importlib.util.find_spec("PyNvVideoCodec")
if spec_info and spec_info.submodule_search_locations:
    pynv_dir = list(spec_info.submodule_search_locations)[0]
else:
    raise ImportError("未找到 PyNvVideoCodec 安装路径")

binaries = []

if os.path.exists(pynv_dir):
    for f in os.listdir(pynv_dir):
        if f.endswith(".pyd") or f.endswith(".dll"):
            binaries.append(
                (
                    os.path.join(pynv_dir, f),
                    "PyNvVideoCodec"
                )
            )

a = Analysis(
    ['videoextract.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=[
        'PyNvVideoCodec',
        '_PyNvVideoCodec'
    ],
)

# 后续的 PYZ, EXE, COLLECT 等保持不变...
