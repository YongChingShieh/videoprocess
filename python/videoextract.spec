import os
import PyNvVideoCodec

pynv_dir = os.path.dirname(PyNvVideoCodec.__file__)


binaries = []

for f in os.listdir(pynv_dir):
    if f.endswith(".pyd") or f.endswith(".dll"):
        binaries.append(
            (
                os.path.join(pynv_dir,f),
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
