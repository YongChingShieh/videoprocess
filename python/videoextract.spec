# -*- mode: python ; coding: utf-8 -*-
import os
import importlib.util
from PyInstaller.utils.hooks import collect_all

# 1. 一次性收集 setuptools、pkg_resources 和 pip 的依赖与静态资源
pkg_res_datas, pkg_res_binaries, pkg_res_hiddenimports = collect_all('setuptools')
pip_datas, pip_binaries, pip_hiddenimports = collect_all('pip')

# 2. 获取 PyNvVideoCodec 模块路径
spec_info = importlib.util.find_spec("PyNvVideoCodec")
binaries = list(pkg_res_binaries) + list(pip_binaries)

if spec_info and spec_info.submodule_search_locations:
    pynv_dir = list(spec_info.submodule_search_locations)[0]
    if os.path.exists(pynv_dir):
        for f in os.listdir(pynv_dir):
            if f.endswith(".pyd") or f.endswith(".dll"):
                binaries.append((os.path.join(pynv_dir, f), "PyNvVideoCodec"))

# 3. 依赖分析
a = Analysis(
    ['videoextract.py'],
    pathex=[],
    binaries=binaries,
    datas=pkg_res_datas + pip_datas,
    hiddenimports=[
        'PyNvVideoCodec',
        '_PyNvVideoCodec',
        'pkg_resources',
        'pkg_resources.extern',
        'setuptools',
        'pip',                     # 明确强制加入 pip 模块
    ] + pkg_res_hiddenimports + pip_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 4. 打包 PYZ 资源
pyz = PYZ(a.pure)

# 5. 生成 EXE 文件
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

# 6. 收集依赖输出到目录 (onedir 模式)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='videoextract',
)
