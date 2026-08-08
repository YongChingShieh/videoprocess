# -*- mode: python ; coding: utf-8 -*-

import os
import importlib.util

# 动态获取 PyNvVideoCodec 的物理路径
spec = importlib.util.find_spec("PyNvVideoCodec")
pynv_dir = os.path.dirname(spec.origin) if spec else None

datas = []
if pynv_dir:
    # 强制将整个 PyNvVideoCodec 目录打入 exe 内部的 PyNvVideoCodec 文件夹
    datas.append((pynv_dir, 'PyNvVideoCodec'))

a = Analysis(
    ['videoextract.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['PyNvVideoCodec'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='videoextract',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
