 
import os
from typing import List, Optional, Dict, Any, Callable, Union,Tuple
import numpy as np
import torch
import torch.nn.functional as F
 
 
import torchvision.transforms as transforms 
from PIL import Image
import time
import threading
import subprocess
import json
import hashlib
from pathlib import Path
 
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import concurrent.futures
import requests
import PyNvVideoCodec as nvc
import io 
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
 
from flask import Flask, request, jsonify
from fractions import Fraction
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "video_segments"
VECTOR_DIM = 768
 
class MilvusDB:
    def __init__(self, host: str = MILVUS_HOST, port: str = MILVUS_PORT, collection_name: str = COLLECTION_NAME, vector_dim: int = VECTOR_DIM):
                # 控制重启的锁
        self._restart_lock = threading.Lock()
        # 控制插入的锁
        self._insert_lock = threading.Lock()
        # 记录最后一次重启时间
        self._last_restart_time = 0
        # 重启成功后等待的时间（秒）
        self._wait_after_restart = 60
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.collection: Collection = None
        self._connect()
        self.ensure_collection_and_index()

    def _connect(self):
        connections.connect(host=self.host, port=self.port)
    def _check_connection(self) -> bool:
        """使用curl检查Milvus HTTP健康端点"""
        url = f"http://{self.host}:9091/healthz"  # 注意端口改为9091
        try:
            # 执行curl命令，启用失败时返回错误码(-f)，设置超时2秒(--max-time)
            result = subprocess.run(
                ["curl", "-f", "-s", "--max-time", "2", url],
                capture_output=True,
                text=True,
                timeout=2.5  # 总超时略长于curl超时
            )
            # 检查返回内容是否包含预期文本（可选）
            return "OK" in result.stdout
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
 
    def _restart_milvus(self) -> bool:
        """重启Milvus容器，返回是否成功"""
        print("开始重启Milvus容器...")
        try:
            # 执行重启命令
            result = subprocess.run(
                ["docker", "restart", "milvus-standalone"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"重启失败: {result.stderr}")
                return False
            
            print("容器重启成功，等待服务启动...")
            
            # 等待服务启动
            for i in range(10):  # 最多等待30秒
                time.sleep(3)
                if self._check_connection():
                    print("Milvus服务已恢复")
                    return True
                print(f"等待服务启动... ({i+1}/10)")
            
            print("服务启动超时")
            return False
            
        except subprocess.TimeoutExpired:
            print("重启命令执行超时")
            return False
        except Exception as e:
            print(f"重启过程中发生错误: {e}")
            return False
    def _ensure_milvus_available(self):
        """
        确保Milvus可用
        1. 如果连接正常，直接返回
        2. 如果连接失败，执行重启
        3. 重启成功，等待1分钟后才允许插入
        4. 重启失败，终止程序
        """
        # 先快速检查连接
        if self._check_connection():
            return

        # 尝试获取重启锁
        if self._restart_lock.acquire(blocking=False):
            try:
                print("检测到Milvus连接失败，开始重启...")
                
                # 执行重启
                if self._restart_milvus():
                    # 重启成功，记录时间
                    self._last_restart_time = time.time()
                    print(f"重启成功，等待{self._wait_after_restart}秒后才能插入...")
                else:
                    # 重启失败，终止程序
                    print("Milvus重启失败，终止程序")
                    os._exit(1)  # 终止整个程序
            finally:
                self._restart_lock.release()
        else:
            # 已经有其他线程在重启，等待重启完成
            with self._restart_lock:
                pass  # 等待重启完成
                
        # 检查是否在等待期内
        elapsed = time.time() - self._last_restart_time
        if elapsed < self._wait_after_restart:
            wait_time = self._wait_after_restart - elapsed
            print(f"重启等待期内，等待{wait_time:.1f}秒...")
            time.sleep(wait_time)

    def create_video_collection_schema(self) -> Tuple[CollectionSchema, Dict[str, Any]]:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="frame_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="video_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="start_time", dtype=DataType.DOUBLE),
            FieldSchema(name="end_time", dtype=DataType.DOUBLE),
            FieldSchema(name="segment_index", dtype=DataType.INT64),
            FieldSchema(name="duration", dtype=DataType.DOUBLE),
            FieldSchema(name="path_sha256", dtype=DataType.VARCHAR, max_length=64),
        ]

        schema = CollectionSchema(fields, description="视频片段向量数据库")

        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 20,
                "efConstruction": 300,
            }
        }
        
        return schema, index_params

    def ensure_collection_and_index(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            return

        schema, index_params = self.create_video_collection_schema()
        
        self.collection = Collection(
            name=self.collection_name, 
            schema=schema, 
            using='default'
        )
        self.collection.create_index(
            field_name="frame_vector",
            index_params=index_params
        )
        self.collection.create_index(
            field_name="path_sha256",
            index_params={"index_type": "TRIE"}
        )
        self.collection.load()

    def insert_data(self, data: List[Dict[str, Any]]) -> List[int]:
                # 获取插入锁
        with self._insert_lock:
            # 确保Milvus可用
            self._ensure_milvus_available()
            
            # 原始插入逻辑
            if not self.collection:
                raise ValueError("集合未初始化或连接失败。")
            
            field_names = [field.name for field in self.collection.schema.fields 
                          if not field.is_primary or not field.auto_id]
            columns = {name: [d.get(name) for d in data] for name in field_names}
            result = self.collection.insert(list(columns.values()), 
                                           field_names=list(columns.keys()))
            self.collection.flush()
            
            return result.primary_keys

    def search_by_vector(self, query_vector: List[float], top_k: int = 10, filter_expr: str = "") -> List[Dict[str, Any]]:
        if not self.collection:
            raise ValueError("集合未初始化或连接失败。")
        search_params = {
            "data": [query_vector],
            "anns_field": "frame_vector",
            "param": {"ef": 50},
            "limit": top_k,
            "output_fields": ["video_path", "start_time", "end_time", "segment_index", "duration"],
        }
        
        if filter_expr:
            search_params["expr"] = filter_expr
            
        results = self.collection.search(**search_params)
        
        formatted_results = []
        for hit in results[0]:
            result_dict = {
                "id": hit.id,
                "distance": hit.distance,
                "video_path": hit.entity.get("video_path"),
                "start_time": hit.entity.get("start_time"),
                "end_time": hit.entity.get("end_time"),
                "segment_index": hit.entity.get("segment_index"),
                "duration": hit.entity.get("duration"),
            }
            formatted_results.append(result_dict)
            
        return formatted_results

    def query_by_expression(self, expression: str, limit: int, output_fields: List[str] = None) -> List[Dict[str, Any]]:
        if not self.collection:
            raise ValueError("集合未初始化或连接失败。")
            
        if output_fields is None:
            output_fields = [f.name for f in self.collection.schema.fields if f.dtype != DataType.FLOAT_VECTOR]
            
        results = self.collection.query(
            expr=expression,
            output_fields=output_fields,
            limit=limit
        )
        
        return results
    def export_to_json(
    self,
    output_file: Optional[str] = None,
    batch_size: int = 1000,
    expr: str = "",
    output_fields: Optional[List[str]] = None,
    include_vectors: bool = True,
    show_progress: bool = True) :
        """
        将集合数据导出到 JSON 文件
        
        Args:
            output_file: 输出文件路径，如果为 None 则自动生成
            batch_size: 每批次获取的记录数
            expr: 过滤表达式
            output_fields: 要导出的字段列表，None 表示全部字段
            include_vectors: 是否包含向量字段
            show_progress: 是否显示进度条
            
        Returns:
            导出统计信息字典
        """
        if not self.collection:
            raise ValueError("集合未初始化或连接失败")
        
        start_time = time.time()
        
        # 获取集合信息
        self.collection.load()
        total_records = self.collection.num_entities
        
        # 确定输出字段
        if output_fields is None:
            output_fields = ["*"]  # 导出所有字段
        
        # 如果不包含向量，过滤掉向量字段
        if not include_vectors:
            schema_fields = [field.name for field in self.collection.schema.fields 
                            if field.dtype != DataType.FLOAT_VECTOR]
            if output_fields == ["*"]:
                output_fields = schema_fields
            else:
                # 只保留非向量字段
                output_fields = [field for field in output_fields 
                                if field in schema_fields]
        
        # 设置输出文件路径
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "./milvus_exports"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{self.collection_name}_{timestamp}.json")
        else:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始导出集合: {self.collection_name}")
        print(f"总记录数: {total_records:,}")
        print(f"输出文件: {output_file}")
        if expr:
            print(f"过滤条件: {expr}")
        print(f"批次大小: {batch_size}")
        
        # 创建迭代器
        iterator = self.collection.query_iterator(
            batch_size=batch_size,
            expr=expr,
            output_fields=output_fields
        )
        
        all_data = []
        batch_count = 0
        total_exported = 0
        
        # 创建进度条
 
        
        try:
            while True:
                result = iterator.next()
                
                if not result:
                    break
                
                all_data.extend(result)
                batch_count += 1
                total_exported += len(result)
                
      
        
        except Exception as e:
            print(f"导出过程中发生错误: {e}")
            raise
        finally:
            iterator.close()
      
        
        # 保存到 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        

    
    def export_to_jsonl(
        self,
        output_file: Optional[str] = None,
        batch_size: int = 1000,
        expr: str = "",
        output_fields: Optional[List[str]] = None,
        include_vectors: bool = True,
        show_progress: bool = True) :
        """
        将集合数据导出到 JSONL 文件（适合大数据量）
        
        Args:
            output_file: 输出文件路径，如果为 None 则自动生成
            batch_size: 每批次获取的记录数
            expr: 过滤表达式
            output_fields: 要导出的字段列表
            include_vectors: 是否包含向量字段
            show_progress: 是否显示进度条
            
        Returns:
            导出统计信息字典
        """
        if not self.collection:
            raise ValueError("集合未初始化或连接失败")
        
 
        
        # 获取集合信息
        self.collection.load()
        total_records = self.collection.num_entities
        
        # 确定输出字段
        if output_fields is None:
            output_fields = ["*"]  # 导出所有字段
        
        # 如果不包含向量，过滤掉向量字段
        if not include_vectors:
            schema_fields = [field.name for field in self.collection.schema.fields 
                           if field.dtype != DataType.FLOAT_VECTOR]
            if output_fields == ["*"]:
                output_fields = schema_fields
            else:
                # 只保留非向量字段
                output_fields = [field for field in output_fields 
                               if field in schema_fields]
        
        # 设置输出文件路径
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "./milvus_exports"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{self.collection_name}_{timestamp}.jsonl")
        else:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始导出集合: {self.collection_name} (JSONL格式)")
        print(f"总记录数: {total_records:,}")
        print(f"输出文件: {output_file}")
        if expr:
            print(f"过滤条件: {expr}")
        
        # 创建迭代器
        iterator = self.collection.query_iterator(
            batch_size=batch_size,
            expr=expr,
            output_fields=output_fields
        )
        
        total_exported = 0
        batch_count = 0
        

        
        # 分批写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            try:
                while True:
                    result = iterator.next()
                    
                    if not result:
                        break
                    
                    # 逐行写入 JSONL
                    for record in result:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
                    
                    total_exported += len(result)
                    batch_count += 1
                    
         
            
            except Exception as e:
                print(f"导出过程中发生错误: {e}")
                raise
            finally:
                iterator.close()
          
        


def get_video_metadata_json(video_path: str) -> Dict[str, any]:
    """使用JSON格式获取视频元数据"""
    ffprobe_cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        # 查找视频流
        video_stream = None
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("未找到视频流")
        
        # 提取信息
        fps_str = video_stream.get('avg_frame_rate', '30/1')
        duration = float(video_stream.get('duration', 0) or probe_data.get('format', {}).get('duration', 0))
        codec_name=video_stream.get('codec_name')
        # 计算帧率
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(Fraction(int(num), int(den)))
            else:
                fps = float(fps_str)
        except (ValueError, ZeroDivisionError):
            fps = 30.0
            logger.warning(f"无法解析帧率 '{fps_str}'，使用默认值30fps")
        
        if fps <= 0:
            fps = 30.0
        
        # 计算总帧数
        if duration > 0:
            total_frames = int(duration * fps)
        else:
            total_frames = 0  # 未知时长
        
        return {
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames,
            'width': video_stream.get('width', 1920),
            'height': video_stream.get('height', 1080),
            'codec_name':codec_name
        }
        
    except Exception as e:
        logger.error(f"获取视频元数据失败: {e}")
        return {'fps': 30.0, 'duration': 0, 'total_frames': 0, 'width': 1920, 'height': 1080}

socks5='socks5://127.0.0.1:10080'
def download_image_from_url(image_url: str) -> Image.Image:
    """
    从URL下载图片，支持代理
    
    Args:
        image_url: 图片URL
        proxy_host: 代理主机
        proxy_port: 代理端口
        
    Returns:
        PIL Image对象
    """
  
    try:
        response = requests.get(image_url, proxies={
                'http': socks5,
                'https': socks5
            },verify=False, timeout=30)
        response.raise_for_status()
        
        # 从字节流创建图片
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes).convert('RGB')
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"下载图片失败: {str(e)}")
    except Exception as e:
        raise Exception(f"处理图片失败: {str(e)}")
 
def extract_image_features(image_path: str, model) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image=download_image_from_url(image_url=image_path)
    else:
        image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).cuda()
    
    video_format_tensor = image_tensor.unsqueeze(1)
    
    with torch.no_grad():
        features = model.vision_model.get_vid_features(video_format_tensor).float()
        features = F.normalize(features, p=2, dim=-1)
    
    return features.cpu()

def search_similar_videos(image_path: str, db: MilvusDB, model:Any, top_k: int = 5) -> List[Dict]:
    image_features = extract_image_features(image_path, model)
    return db.search_by_vector(image_features.tolist()[0], top_k)
 
Bluray = "bluray:"

def process_folder(path: str, virtual_disk: str, filter_ext: str = ".srt,.ass,.txt,.json,.wav,.text") -> List[Dict[str, Any]]:
    bdmv = "BDMV"
    special_exts = set(ext.strip().lower() for ext in filter_ext.split(","))
    result_list = []
    
    if virtual_disk:
        result_list.append(f"{Bluray}{path}")
    else:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == ".iso":
                result_list.append(f"{Bluray}{path}")
            else:
                result_list.append(path)
        else:
            bluray_roots = set()
            
            for root, dirs, files in os.walk(path):
                if bdmv in dirs:
                    parent_dir = root
                    if parent_dir not in bluray_roots:
                        bluray_roots.add(parent_dir)
                        result_list.append(f"{Bluray}{parent_dir}")
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    
                    if ext in special_exts:
                        continue
                    
                    is_under_bluray = any(
                        file_path.startswith(bluray_root + os.sep) 
                        for bluray_root in bluray_roots
                    )
                    
                    if is_under_bluray:
                        continue
                    
                    if ext == ".iso":
                        result_list.append(f"{Bluray}{file_path}")
                    else:
                        result_list.append(file_path)
    
    results = []
    for file_path in result_list:
        hash_result = sha256(file_path, virtual_disk)
        results.append({
            "sha256": hash_result[0],
            "path": hash_result[1], 
            "virtual_disk": hash_result[2]
        })
    return results

def filter_existing_videos(videopath: List[Tuple], db: MilvusDB) -> List[Tuple]:
    all_existing = set()
    field = "path_sha256"
    for item in videopath:
        try:
            expr = f"{field} == '{item["sha256"]}'"
            query_result = db.query_by_expression(
                expression=expr, 
                output_fields=[f"{field}"],
                limit=1
            )
            
            if query_result:
                for result in query_result:
                    if field in result:
                        all_existing.add(result[f"{field}"])
        except Exception as e:
            print(f"查询sha256时出错: {e}")
            
    return [item for item in videopath 
            if item["sha256"].lower() not in {existing.lower() for existing in all_existing}]

def sha256(path: str, virtual_disk: str) -> Tuple[str, str, str]:
    path_obj = Path(path)
    
    if path_obj.is_absolute():
        drive, tail = os.path.splitdrive(path)
        path_segments = tail.lstrip(os.sep).split(os.sep)
    else:
        path_segments = path.split(os.sep)
    
    path_string = "".join(path_segments)
    sha256_hash = hashlib.sha256(path_string.encode('utf-8')).hexdigest()
    
    return (sha256_hash, path, virtual_disk)

def format_time(seconds):
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"



app = Flask(__name__)

# 全局变量
videoclip_xl = None
#model_lock = threading.Lock() 
db = MilvusDB()
def load_model_once(weights_path="/mnt/c/Users/YongChing/Downloads/_models/VideoCLIP-XL/VideoCLIP-XL.bin"):
    """一次性加载模型（全局单例）"""
    global videoclip_xl
    if videoclip_xl is not None:
        return videoclip_xl
    
    from modeling import VideoCLIP_XL
  
    
    logger.info("正在加载模型...")
    videoclip_xl = VideoCLIP_XL()
    state_dict = torch.load(weights_path, map_location="cpu")
    videoclip_xl.load_state_dict(state_dict)
    videoclip_xl.cuda().eval()
 
    logger.info("模型加载完成")
    return videoclip_xl

device = "cuda"
# 改进的代码 - 专注批次推理支持
def improved_process_with_simple_ffmpeg(
    video_path: str, 
    video_info: Dict, 
    model, 
    metadata=None, 
    interval_seconds: float = 5.0,
    max_batch_size: int = 32
) -> bool:
    """改进的FFmpeg处理版本，结合两种方法的优点"""
    
    try:
        logger.info(f"ffmpeg video {video_info}")
        # 1. 获取元数据
        if metadata is None or not isinstance(metadata, dict):
            metadata = get_video_metadata_json(video_path)
        fps = metadata['fps']
        total_frames = metadata['total_frames']
        
        if fps <= 0:
            fps = 30.0
        
        # 2. 计算采样参数
        frames_per_segment = 8
        interval_frames = max(1, int(round(interval_seconds * fps)))
        
        # 3. 计算总采样点数
        if total_frames > 0:
            num_segments = max(1, (total_frames - frames_per_segment) // interval_frames)
            logger.info(f"预计采样点: {num_segments}个")
        else:
            num_segments = 0
            logger.info("未知视频时长，采用流式处理")
        
        # 4. 创建VideoFeatureExtractor实例
        feature_extractor = VideoFeatureExtractor(model, interval_seconds, max_batch_size, gpu_id=0)
        
        # 5. 构建优化的FFmpeg命令
        ffmpeg_cmd = [
            'ffmpeg', 
            '-hwaccel', 'cuda',
            '-i', video_path,
            '-vf', f'select=between(mod(n\\,{interval_frames})\\,0\\,{frames_per_segment-1}),scale=224:224',
            '-vsync', '0',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-loglevel', 'error',
            'pipe:1'
        ]
        
        frame_size = 224 * 224 * 3
        all_data = []
        batch_tensors = []
        batch_metadata = []
        
        segment_index = 0
        current_segment_frames = []
        logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")
        logger.info(f"参数: fps={fps}, interval_frames={interval_frames}, frames_per_segment={frames_per_segment}")
        
        # 6. 启动FFmpeg进程
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        try:
            while True:
                # 读取一帧
                frame_data = process.stdout.read(frame_size)
                if not frame_data:
                    break
                
                # 转换为tensor
                frame_np = np.frombuffer(frame_data, dtype=np.uint8).reshape(224, 224, 3).copy()
                tensor = torch.from_numpy(frame_np).to(f'cuda:{feature_extractor.gpu_id}')
                tensor = tensor.float().div(255.0)
                tensor = tensor.permute(2, 0, 1)  # HWC to CHW
                
                # 添加到当前segment
                current_segment_frames.append(tensor)
                
                # 当收集到8帧时处理一个segment
                if len(current_segment_frames) == frames_per_segment:
                    segment_tensor = torch.stack(current_segment_frames)  # (8, 3, 224, 224)
                    batch_tensors.append(segment_tensor)
                    
                    # 计算起始帧
                    start_frame = segment_index * interval_frames
                    batch_metadata.append({
                        'start_frame': start_frame,
                        'segment_index': segment_index
                    })
                    
                    segment_index += 1
                    current_segment_frames = []
                    
                    # 批次处理
                    if len(batch_tensors) >= max_batch_size:
                        batch_results = _process_batch_improved(
                            batch_tensors, batch_metadata, feature_extractor, fps,
                            video_path, video_info, frames_per_segment, interval_frames
                        )
                        all_data.extend(batch_results)
                        batch_tensors = []
                        batch_metadata = []
        
        finally:
            # 终止进程
            if process.poll() is None:
                process.terminate()
                process.wait()
        
        # 7. 处理剩余的批次
        if batch_tensors:
            batch_results = _process_batch_improved(
                batch_tensors, batch_metadata, feature_extractor, fps,
                video_path, video_info, frames_per_segment, interval_frames
            )
            all_data.extend(batch_results)
        
        # 8. 处理最后不足8帧的情况
        if current_segment_frames:
            logger.warning(f"丢弃不足{frames_per_segment}帧的片段")
        db.insert_data(all_data)
        logger.info(f"成功处理 {len(all_data)} 个视频片段")
        return True
        
    except Exception as e:
        logger.error(f"改进版FFmpeg处理失败: {e}")
        return False


def _process_batch_improved(
    batch_tensors: List[torch.Tensor], 
    batch_metadata: List[Dict], 
    feature_extractor, 
    fps, 
    video_path, 
    video_info, 
    frames_per_segment, 
    interval_frames
) -> List[Dict]:
    """处理批次的完整实现"""
    
    if not batch_tensors:
        return []
    
    try:
        # 堆叠批次
        batch_tensor = torch.stack(batch_tensors)  # (B, 8, 3, 224, 224)
        
        # 批次推理
        features = feature_extractor.infer_batch(batch_tensor)
        
        if features is None:
            return []
        
        results = []
        inv_fps = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # 处理每个结果
        for i, (feature, meta) in enumerate(zip(features, batch_metadata)):
            # 计算时间信息
            actual_start_frame_index = meta['start_frame']
            start_time = actual_start_frame_index * inv_fps
            duration = float(frames_per_segment * inv_fps)
            end_time = start_time + duration
            
            results.append({
                'frame_vector': feature.float().cpu().numpy().flatten().tolist(),
                'video_path': video_path,
                'start_time': float(start_time), 
                'end_time': float(end_time),     
                'segment_index': meta['segment_index'],
                'start_frame': actual_start_frame_index,
                'duration': duration,
                'path_sha256': video_info.get('sha256', '')
            })
        
        return results
    except Exception as e:
        logger.error(f"批次处理失败: {e}")
        return []


class VideoDemuxDecoder:
    """使用标准解码器管道的视频解码器"""
    
    def __init__(self, video_path: str, gpu_id: int = 0, use_device_memory: bool = True):
        self.video_path = video_path
        self.gpu_id = gpu_id
        self.use_device_memory = use_device_memory
        self.demuxer = None
        self.decoder = None
        self.frames = []
        self.current_frame_index = 0
        
    def initialize(self) -> bool:
        """初始化解码器"""
        try:
            torch.cuda.set_device(self.gpu_id)
            # 1. 创建解复用器
            self.demuxer = nvc.CreateDemuxer(filename=self.video_path)
            
            # 2. 获取视频编码信息并创建解码器
            codec = self.demuxer.GetNvCodecId()
            
            self.decoder = nvc.CreateDecoder(
                gpuid=self.gpu_id,
                codec=codec,
                cudacontext=0,
                cudastream=0,
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.RGBP
            )
            
            return True
            
        except Exception as e:
            logger.error(f"解码器初始化失败 {self.video_path}: {e}")
            return False
    
    def get_next_frame(self):
        """获取下一帧"""
        if self.demuxer is None or self.decoder is None:
            raise RuntimeError("解码器未初始化")
        
        # 1. 优先检查缓存
        if len(self.frames) > 0:
            return self.frames.pop(0)
        
        # 2. 缓存为空，从解码器获取更多帧
        try:
            for packet in self.demuxer:
                decoded_frames = self.decoder.Decode(packet)
                
                # 将新解码的帧加入缓存
                for frame in decoded_frames:
                    self.frames.append(frame)
                
                # 检查刚加入后是否有数据
                if len(self.frames) > 0:
                    return self.frames.pop(0)
            
            # 3. 处理流结束
            empty_packet = nvc.PacketData()
            flushed_frames = self.decoder.Decode(empty_packet)
            
            for frame in flushed_frames:
                self.frames.append(frame)
            
            # 再次检查缓存
            if len(self.frames) > 0:
                return self.frames.pop(0)
            
            return None
            
        except Exception as e:
            logger.error(f"解码帧失败: {e}")
            return None
    
    def skip_frames(self, count: int) -> int:
        """跳过指定数量的帧"""
        skipped = 0
        while skipped < count:
            frame = self.get_next_frame()
            if frame is None:
                break
            skipped += 1
        return skipped
    
    def close(self) -> None:
        """关闭解码器并释放资源"""
        try:
            self.frames.clear()
            self.current_frame_index = 0
            
            if self.decoder is not None:
                del self.decoder
                self.decoder = None
            
            if self.demuxer is not None:
                del self.demuxer
                self.demuxer = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"关闭解码器失败: {e}")

CODEC_MAPPING = {
    "av1": "av1_cuvid",
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "mpeg1video": "mpeg1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid", 
    "vc1": "vc1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid"
}
class VideoFeatureExtractor:
    """使用标准解码器的视频特征提取器，支持批次推理"""
    
    def __init__(self, shared_model, segment_seconds=5, buffer_size=32, gpu_id=0):
        self.shared_model = shared_model
        self.segment_seconds = segment_seconds
        self.gpu_id = gpu_id
        self.buffer_size = buffer_size
        # 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=f'cuda:{gpu_id}').view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=f'cuda:{gpu_id}').view(1, 3, 1, 1)
    
    def extract_features_for_db(self, video_info: Dict) -> bool:
        """提取视频特征，支持批次推理"""
        video_path = video_info.get('path')
        
        try:
            virtual_disk = video_info.get('virtual_disk')
            if virtual_disk:
                video_path = virtual_disk
            if Bluray in video_path: 
                return improved_process_with_simple_ffmpeg(video_path, video_info, self.shared_model, 
                                                          None, self.segment_seconds, self.buffer_size)   
            
            # 1. 获取视频元数据
            metadata = get_video_metadata_json(video_path)
            codec_name=metadata.get("codec_name")
            if codec_name not in  CODEC_MAPPING:
                return improved_process_with_simple_ffmpeg(video_path, video_info, self.shared_model, 
                                                          None, self.segment_seconds, self.buffer_size)   
            fps = metadata['fps']
            total_frames = metadata['total_frames']
            video_duration = metadata['duration']
            
            # 计算或使用已有的SHA256
            path_sha256 = video_info.get('sha256')
            
            logger.info(f"处理视频: {os.path.basename(video_path)}")
            
            # 2. 计算采样参数
            frames_per_segment = int(self.segment_seconds * fps)
            sample_interval = max(1, frames_per_segment // 8)
            
            logger.info(f"  批次大小: {self.buffer_size}个片段")
            logger.info(f"  每段帧数: {frames_per_segment}, 采样间隔: {sample_interval}帧")
            
            # 3. 初始化解码器
            decoder = VideoDemuxDecoder(
                video_path=video_path,
                gpu_id=self.gpu_id,
                use_device_memory=True
            )
            
            if not decoder.initialize():
                logger.error(f"解码器初始化失败: {video_path}")
                return improved_process_with_simple_ffmpeg(
                    video_path, video_info, self.shared_model, metadata, 
                    self.segment_seconds, self.buffer_size
                )
            
            # 4. 批次处理
            db_records = []
            segment_index = 0
            batch_segments = []  # 存储片段张量
            batch_metadata = []  # 存储片段元数据
            
            while True:
                sampled_frames = []
                
                frames_consumed_total = 0
                # 采样当前时间段的8帧
                for sample_idx in range(8):
                    # 获取采样帧
                    frame = decoder.get_next_frame()
                    if frame is None:
                        break
                    frames_consumed_total += 1
                    try:
                        # 预处理帧
                        if torch.cuda.current_device() != self.gpu_id:
                            torch.cuda.set_device(self.gpu_id)
                        
                        frame_tensor = torch.from_dlpack(frame)
                        processed_frame = self._preprocess_frame(frame_tensor)
                        sampled_frames.append(processed_frame)
                       
                        
                    except Exception as e:
                        logger.warning(f"帧预处理失败: {e}")
                       
                        continue
                    
                    # 跳过非采样帧
                    if sample_idx < 7:
                        skipped_in_interval = 0
                        for _ in range(sample_interval - 1):
                            skipped_frame = decoder.get_next_frame()
                            if skipped_frame is None:
                                break
                            skipped_in_interval += 1
                            frames_consumed_total += 1
                        if skipped_in_interval < (sample_interval - 1):
                            break
                
                # 检查是否成功采集到一个完整的时间段
                if len(sampled_frames) == 8:
                    # 堆叠片段帧
                    segment_tensor = torch.stack(sampled_frames)  # (8, 3, 224, 224)
                    batch_segments.append(segment_tensor)
                    
                    # 计算元数据
                    start_frame = segment_index * frames_per_segment
                    batch_metadata.append({
                        'segment_index': segment_index,
                        'start_frame': start_frame,
                        'frames_per_segment': frames_per_segment,
                        'fps': fps
                    })
                    
                    segment_index += 1
                    
                    # 跳过当前时间段剩余的帧
                    remaining_frames = frames_per_segment - frames_consumed_total
                    remaining_frames = max(0, remaining_frames)
                    for _ in range(remaining_frames):
                        skipped_frame = decoder.get_next_frame()
                        if skipped_frame is None:
                            break
                    
                    # 检查批次是否已满
                    if len(batch_segments) >= self.buffer_size:
                        batch_records = self._process_batch(batch_segments, batch_metadata, video_path, video_info)
                        db_records.extend(batch_records)
                        batch_segments = []
                        batch_metadata = []
                
                elif len(sampled_frames) > 0:
                    logger.warning(f"时间段 {segment_index} 不完整，仅采样到 {len(sampled_frames)} 帧，跳过该段")
                    break
                else:
                    break
            
            # 5. 处理剩余的批次
            if batch_segments:
                batch_records = self._process_batch(batch_segments, batch_metadata, video_path, video_info)
                db_records.extend(batch_records)
            
            # 6. 关闭解码器
            decoder.close()
            
            # 7. 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not db_records:
                return improved_process_with_simple_ffmpeg(
                    video_path, video_info, self.shared_model, metadata, 
                    self.segment_seconds, self.buffer_size
                )
            db.insert_data(db_records)
            logger.info(f"成功处理 {len(db_records)} 个片段")
            return True
            
        except Exception as e:
            logger.error(f"处理视频 {video_path} 时出错: {e}", exc_info=True)
            return False
    
    def _process_batch(
        self, 
        batch_segments: List[torch.Tensor], 
        batch_metadata: List[Dict], 
        video_path: str, 
        video_info: Dict
    ) -> List[Dict]:
        """处理批次并生成数据库记录"""
        if not batch_segments:
            return []
        
        try:
            # 堆叠批次
            batch_tensor = torch.stack(batch_segments)  # (B, 8, 3, 224, 224)
            
            # 批次推理
            features = self.infer_batch(batch_tensor)
            
            if features is None:
                return []
            
            db_records = []
            
            for i, (feature, meta) in enumerate(zip(features, batch_metadata)):
                # 计算时间信息
                fps = meta['fps']
                start_frame = meta['start_frame']
                frames_per_segment = meta['frames_per_segment']
                segment_index = meta['segment_index']
                
                start_time_sec = start_frame / fps
                end_time_sec = (start_frame + frames_per_segment) / fps
                
                # 创建数据库记录
                record = {
                    'frame_vector': feature.float().cpu().numpy().flatten().tolist(),
                    'video_path': video_path,
                    'start_time': float(start_time_sec),
                    'end_time': float(end_time_sec),
                    'segment_index': segment_index,
                    'duration': float(frames_per_segment / fps),
                    'path_sha256': video_info.get('sha256', '')
                }
                
                db_records.append(record)
            
            return db_records
        except Exception as e:
            logger.error(f"批次处理失败: {e}")
            return []
    
    def infer_batch(self, batch_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """批次推理函数"""
        try:
            if torch.cuda.current_device() != self.gpu_id:
                torch.cuda.set_device(self.gpu_id)
            
            # 归一化
            batch_tensor = (batch_tensor - self.mean) / self.std
            
            with torch.amp.autocast('cuda'), torch.no_grad():
                features = self.shared_model.vision_model.get_vid_features(batch_tensor)
                features = F.normalize(features, p=2, dim=-1)
            
            return features
        
        except Exception as e:
            logger.error(f"批次推理失败: {e}")
            return None
    
    def _preprocess_frame(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """预处理单帧"""
        frame_tensor = frame_tensor.cuda(device=self.gpu_id).float() / 255.0
        
        if frame_tensor.shape[-2:] != (224, 224):
            frame_tensor = F.interpolate(
                frame_tensor.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return frame_tensor 

# 初始化函数
def initialize_system():
    """初始化模型和数据库"""
    global videoclip_xl 
    
    try:
        # 初始化数据库
      
        
        # 加载模型
        videoclip_xl =  load_model_once()
        
        return True, "系统初始化成功"
    except Exception as e:
        return False, f"系统初始化失败: {str(e)}"

@app.route('/api/features/exportjsonl', methods=['POST']) 
def export_to_jsonl():
    try:
        data = request.get_json()
        batch_size=data.get("batch_size",1000)
        if batch_size>16384:
            batch_size=16384
        db.export_to_jsonl(batch_size=batch_size)
        return jsonify("ok")
    except Exception as e:

        return jsonify({"error": f"{str(e)}"}), 500

@app.route('/api/features/extract', methods=['POST'])

def extract_features():

    """

    提取视频特征API

    接收视频列表，提取特征并存储到向量数据库

    返回处理结果和异常文件列表

    """

    try:

        if videoclip_xl is None or db is None:

            return jsonify({"error": "系统未初始化，请先调用 /api/initialize"}), 400

        

        data = request.get_json()

        if not data or 'videos' not in data:

            return jsonify({"error": "请提供视频列表"}), 400

        

        video_list = data['videos']

        buffer_size = data.get('buffer_size', 32)

        max_workers = data.get('max_workers', 10)

        

        # 验证视频列表格式

        if not isinstance(video_list, list):

            return jsonify({"error": "videos参数必须是列表"}), 400

        

        # 处理视频文件
        all_results = []

        for path_item in video_list:
            folder_results = process_folder(
                path_item["path"], 
                path_item["virtualDisk"], 
            )
            all_results.extend(folder_results)  # 缩进修正，放在循环内

        # 检查是否有处理结果
        if not all_results:
            return jsonify({"error": "未找到可处理的视频文件"}), 400

        # 过滤已存在的视频
        filtered_results = filter_existing_videos(all_results, db)

        # 检查过滤后是否有结果
        if not filtered_results:
            return jsonify({"error": "所有视频已存在"}), 400  # 或者根据需求返回不同信息
        # 提取视频特征
        error_videos =[]
        total_segments_extracted = 0
        successful_count=0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个视频创建提取器
            extractors = {}
            for video_info in filtered_results:
                extractor = VideoFeatureExtractor(
                    shared_model=videoclip_xl,
                    segment_seconds=5,
                    buffer_size=buffer_size
                )
                extractors[id(video_info)] = extractor
            
            # 提交所有任务
            future_to_video = {
                executor.submit(extractors[id(video_info)].extract_features_for_db, video_info): video_info
                for video_info in filtered_results
            }
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_video):
                video_info = future_to_video[future]
                video_path = video_info.get('path', '未知路径')
                
                try:
                    # 获取结果
                    if future.result():
                        
                        logger.info(f"✅ 视频处理完成: {os.path.basename(video_path)}")
                    else:
                        error_videos.append(video_info)
                        logger.warning(f"⚠️ 未提取到特征: {os.path.basename(video_path)}")
                        
                except Exception as e:
                    error_videos.append(video_info)
                    logger.error(f"❌ 处理视频 {os.path.basename(video_path)} 时发生错误: {e}", exc_info=True)
        response = {

            "status": "success",

            "total_videos": len(filtered_results),

            "successful_count": len(filtered_results) - len(error_videos),

            "error_count": len(error_videos),

            "processed_count": len(filtered_results) - len(error_videos)  # 成功处理的数量

        }

        

        # 添加错误视频列表

        if error_videos:

            response["error_videos"] = [

                {

                    "path": video_info.get("path", "未知路径"),

                    "error_type": "处理失败"

                } for video_info in error_videos

            ]

        else:

            response["error_videos"] = []

        

        return jsonify(response)

        

    except Exception as e:

        return jsonify({"error": f"提取特征时发生错误: {str(e)}"}), 500
 
@app.route('/api/search/similar', methods=['POST'])
def search_similar():
    """
    搜索相似视频API
    根据图片搜索相似的视频片段
    """
    try:
        if videoclip_xl is None or db is None:
            return jsonify({"error": "系统未初始化，请先调用 /api/initialize"}), 400
        
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({"error": "请提供图片路径"}), 400
        
        image_path = data['image_path']
        top_k = data.get('top_k', 5)
        filter_expr = data.get('filter_expr', "")
        
 
        # 搜索相似视频
        results = search_similar_videos(
            image_path=image_path,
            db=db,
            model=videoclip_xl,
            top_k=top_k
        )
        
        # 格式化结果
        formatted_results = []
        for res in results:
            formatted_results.append({
                "video_path": res["video_path"],
                "start_time":format_time( res["start_time"]),
                "end_time":format_time( res["end_time"]),
                "distance": float(res["distance"]),
                "similarity_score": float(1.0 - res["distance"]),  # 转换为相似度分数(0-1)
                "segment_index": res["segment_index"],
                "duration": res["duration"]
            })
        
        return jsonify({
            "query_image": image_path,
            "top_k": top_k,
            "total_results": len(formatted_results),
            "results": formatted_results
        })
        
    except Exception as e:
        return jsonify({"error": f"搜索相似视频时发生错误: {str(e)}"}), 500

if __name__ == '__main__':
    # 自动初始化系统
    initialize_system()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True, threaded=True)
         
 
    

