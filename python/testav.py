#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N46Whisper 完全复现Colab版本
确保每个细节都与Colab一致
"""

import os
import sys
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
import time
import pysubs2
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def main():
    """完全复现Colab的转录流程"""
    
    # 使用您提供的EXACT参数
    file_type = "audio"
    model_size = "large-v2" 
    language = "ja"
    sub_style = "default"
    is_split = "No"
    split_method = "Modest"
    is_vad_filter = True  
    set_beam_size = 5
    beam_size_off = False
    
    # 音频文件路径
    audio_file = "HEYZO-2856.wav"  # 替换为您的文件路径
    
    print("=" * 60)
    print("N46Whisper - 完全复现Colab版本")
    print("=" * 60)
    print(f"音频文件: {audio_file}")
    print(f"模型: {model_size}")
    print(f"语言: {language}")
    print(f"VAD过滤: {is_vad_filter} (类型: {type(is_vad_filter)})")
    print(f"Beam大小: {set_beam_size}")
    print("=" * 60)
    
    # 1. 模型加载 - 完全复制Colab方式
    print('加载模型 Loading model...')
    model = WhisperModel("/mnt/c/Users/YongChing/Downloads/_models/faster-whisper-large-v2test")  # 与Colab完全一致，不指定任何额外参数
    
    torch.cuda.empty_cache()
    
    # 2. 转录参数 - 完全复制Colab逻辑
    tic = time.time()
    print('识别中 Transcribe in progress...')
    
    # 关键：使用完全相同的参数传递逻辑
    if beam_size_off:
        segments, info = model.transcribe(
            audio=audio_file,
            language=language,
            vad_filter=is_vad_filter,  # 传递字符串"False"
            vad_parameters=dict(min_silence_duration_ms=1000)
        )
    else:
        segments, info = model.transcribe(
            audio=audio_file,
            beam_size=set_beam_size,
            language=language, 
            vad_filter=is_vad_filter,   
            vad_parameters=dict(min_silence_duration_ms=1000)
        )
    
    # 3. 结果处理 - 完全复制Colab方式
    total_duration = round(info.duration, 2)
    results = []
    
    with tqdm(total=total_duration, unit=" seconds") as pbar:
        for s in segments:
            segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
            results.append(segment_dict)
            segment_duration = s.end - s.start
            pbar.update(segment_duration)
    
    toc = time.time()
    print('识别完毕 Done')
    print(f'Time consumption: {toc-tic}s')
    
    # 4. 保存结果 - 复制Colab方式
    subs = pysubs2.load_from_whisper(results)
    
    # 保存SRT
    srt_filename = Path(audio_file).stem + '.srt'
    subs.save(srt_filename)
    
    # 尝试转换为ASS（如果srt2ass可用）
    try:
        from srt2ass import srt2ass
        ass_filename = srt2ass(srt_filename, sub_style, is_split, split_method)
        print('ASS字幕保存为: ' + ass_filename)
    except ImportError:
        print("srt2ass不可用，仅保存SRT文件")
        ass_filename = None
    
    print("=" * 60)
    print("转录完成！")
    print(f"SRT文件: {srt_filename}")
    if ass_filename:
        print(f"ASS文件: {ass_filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()