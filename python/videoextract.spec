# -*- mode: python ; coding: utf-8 -*-
import os
import importlib.util

# 1. 获取 PyNvVideoCodec 模块路径（避开 GPU 驱动检测）
spec_info = importlib.util.find_spec("PyNvVideoCodec")
binaries = []

if spec_info and spec_info.submodule_search_locations:
    pynv_dir = list(spec_info.submodule_search_locations)[0]
    if os.path.exists(pynv_dir):
        for f in os.listdir(pynv_dir):
            if f.endswith(".pyd") or f.endswith(".dll"):
                binaries.append((os.path.join(pynv_dir, f), "PyNvVideoCodec"))

# 2. 依赖分析
a = Analysis(
    ['videoextract.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=[
        'PyNvVideoCodec',
        '_PyNvVideoCodec'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 3. 打包 PYZ 资源
pyz = PYZ(a.pure)

# 4. 生成 EXE 文件
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='videoextract',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 5. 收集依赖输出到目录 (onedir 模式)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='videoextract',
)
