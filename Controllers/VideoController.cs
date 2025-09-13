using System.Collections.Concurrent;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using MihaZupan;
using Microsoft.AspNetCore.Mvc.Filters;
using Milvus.Client;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static WebApiHelper;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Hosting.Server;

namespace videoprocess.Controllers;

[TypeFilter(typeof(WebApiExceptionFilter))]
[Route("api/[controller]/[action]")]
[Produces("application/json")]
[ApiController]

public class VideoController(MilvusImageService milvusService, ILogger<VideoController> logger, IServer server) : ControllerBase
{
    private readonly MilvusImageService _milvusService = milvusService;
    private readonly ILogger<VideoController> _logger = logger;
    private readonly IServer _server = server;

    static string Bluray => "bluray:";
    readonly Dictionary<string, string> dic = new()
    {
        ["av1"] = "av1_cuvid",
        ["h264"] = "h264_cuvid",
        ["hevc"] = "hevc_cuvid",
        ["mjpeg"] = "mjpeg_cuvid",
        ["mpeg1video"] = "mpeg1_cuvid",
        ["mpeg2video"] = "mpeg2_cuvid",
        ["mpeg4"] = "mpeg4_cuvid",
        ["vc1"] = "vc1_cuvid",
        ["vp8"] = "vp8_cuvid",
        ["vp9"] = "vp9_cuvid"
    };
    public async Task<IActionResult> SubittleProcessAsync()
    {
        //todo 
        return Ok();
  }
    public async Task<IActionResult> VideoProcessAsync(List<Dictionary<string , string>> pathList, CancellationToken cancellationToken)
    {
        var videopath = (await Task.WhenAll(pathList.Select(x => ProcessFolder(x["path"],x["virtualDisk"], cancellationToken)))).SelectMany(x => x);
        bool exists = await _milvusService.MilvusClient.HasCollectionAsync(_milvusService.CollectionName, cancellationToken: cancellationToken);
      
        if (!exists)
        {
            await _milvusService.MilvusClient.CreateCollectionAsync(_milvusService.CollectionName, new CollectionSchema
            {
                Fields = {
            FieldSchema.Create("id", MilvusDataType.Int64,true,true),
            FieldSchema.CreateFloatVector("embedding", 512),
            FieldSchema.CreateVarchar("file_path", 1024),
            FieldSchema.CreateVarchar("path_sha256", 64),
        FieldSchema.Create("timestamp", MilvusDataType.Float, description: "帧时间戳（秒）")
        }
            }, cancellationToken: cancellationToken);
            await _milvusService.MilvusCollection.CreateIndexAsync(fieldName: "embedding", indexType: IndexType.IvfFlat, metricType: SimilarityMetricType.L2, indexName: "idx_embedding", extraParams: new Dictionary<string, string>
                   {
            { "nlist", "1024" } 
                   }
, cancellationToken: cancellationToken);
        }
        else
        {
            var allExisting = new ConcurrentBag<string>();
            var parameters = new QueryParameters
            {
                Limit = 1
            };
            parameters.OutputFields.Add("path_sha256");
            await Parallel.ForEachAsync(videopath, new ParallelOptions()
            {
                MaxDegreeOfParallelism = 10,
                CancellationToken=cancellationToken
            }, async (item, ct) =>
            {
                var expr = $"path_sha256 == '{item.sha256}'";
                var query = await _milvusService.MilvusCollection.QueryAsync(expr, parameters, ct);
                if (query?.Count > 0 && query[0] is FieldData<string> filePathField)
                {
                    foreach (var x in filePathField.Data)
                    {
                        allExisting.Add(x);
                    }
                 
                }
            });
            var existingSet = new HashSet<string>(allExisting, StringComparer.OrdinalIgnoreCase);
            videopath = [.. videopath.Where(x => !existingSet.Contains(x.sha256))];
        }
        var special = "mov,m2ts".Split(",");
        if (!videopath.Any())
        {

            return Ok("not found any video file, skipping...");
        }
        
        await Parallel.ForEachAsync(videopath, new ParallelOptions() { MaxDegreeOfParallelism = _milvusService.MaxDegreeOfParallelism,CancellationToken=cancellationToken }, async (file, ct) =>
        {
            try
            {

                _logger.LogInformation($"{file} start");
                await LoadVideo(file, ct);
                _logger.LogInformation($"{file} end");
            }
            catch (Exception ex)
            {
                _logger.LogError(exception: ex, $"LogError: {file}");
                //return;
            }
        });
        return Ok();
    }
   public  async Task<IActionResult> SearchSimilarFrameAsync(string imagePath, CancellationToken cancellationToken,int limit = 5)
    {
        _logger.LogInformation("req");
        Image<Rgb24> image;
        if (imagePath.StartsWith("http") || imagePath.StartsWith("https"))
        {
            var sockslist = _milvusService.Socks.Split(":");
            var proxy = new HttpToSocks5Proxy(sockslist[0],Convert.ToInt32( sockslist[1]));
            var handler = new HttpClientHandler
            {
                Proxy = proxy,
                UseProxy = true
            };
            using var client = new HttpClient(handler);
            var res = await client.GetAsync(imagePath);
            if (!res.IsSuccessStatusCode)
            {
                return BadRequest(res)  ;
            }
            using var ms = new MemoryStream();
            await res.Content.CopyToAsync(ms, cancellationToken);
            ms.Position = 0; // 回到流的起始位置

            // 用 ImageSharp 加载图片
            image = await Image.LoadAsync<Rgb24>(ms, cancellationToken);
        }
        else
        {
            image = await Image.LoadAsync<Rgb24>(imagePath, cancellationToken);
        }
        byte[] imageBuffer = new byte[image.Width * image.Height * 3];
        image.CopyPixelDataTo(imageBuffer);
        var tensor = PreprocessBatchFrames([imageBuffer], image.Width, image.Height);
        var embeddingTensor =   Encode_image( tensor);

        float[] queryVector = [.. embeddingTensor[0].data<float>()];
        var addressesFeature = _server.Features.Get<IServerAddressesFeature>();
    
        return Ok(JsonSerializer.Deserialize<JsonElement>(await PostJsonAsync<string>($"{addressesFeature.Addresses.FirstOrDefault()}/milvus/v2/vectordb/entities/search", new
        {
            collectionName = _milvusService.CollectionName,
            data = new float[][] { queryVector },
            annsField = "embedding",
            limit,
            outputFields = "timestamp,file_path".Split(","),
        }, cancellationToken)).GetProperty("data").EnumerateArray()
        .Select(item => new
        {
            Score = item.GetProperty("distance").GetSingle(),
            FilePath = item.GetProperty("file_path").GetString(),
            Timestamp = TimeSpan.FromSeconds(item.GetProperty("timestamp").GetSingle()).ToString(@"hh\:mm\:ss\.fff")
        }));
    }

  

      Tensor Encode_image( Tensor tensor)
    {
        using (no_grad())
        {
            return _milvusService.ScriptModule.invoke<Tensor>("encode_image", tensor).to(CPU).to(float32);
        }
    }
    // 只负责把 tensor 的 embedding 写入 currentMeta
    static void AddEmbeddingToMeta(List<FrameVector> currentMeta, Tensor tensor)
    {
        long dim = tensor.shape[1];

        if (currentMeta.Count != tensor.shape[0])
            throw new InvalidOperationException("currentMeta.Count 与 tensor 第一维度不匹配");

        for (int i = 0; i < currentMeta.Count; i++)
        {
            var singleTensor = tensor[i]; // shape [dim]
            float[] embeddingArray = new float[dim];
            singleTensor.data<float>().CopyTo(embeddingArray);

            // 关键修改：转成 List<float>，确保 Milvus SDK 支持 FloatVector
            currentMeta[i].Embedding = embeddingArray;
        }
    }




    async Task LoadVideo( (string sha256, string realpath,string virtualDisk) videopath,CancellationToken cancellationToken )
    {
        var tmppath = $"\"{videopath.realpath}\"";
        if (!string.IsNullOrEmpty(videopath.virtualDisk))
        {
            var virtualDisk = $"{Bluray}{videopath.virtualDisk}";
            tmppath =$"\"{virtualDisk}\"";
        }
       var ffprobeparam = @$"-v error -show_entries format=duration -show_entries stream=codec_name,codec_type,width,height,r_frame_rate -of json {tmppath}";
        var json = await RunCmdWithOutput(_milvusService.  FFprobe, ffprobeparam, cancellationToken);
        using var jsonDoc = JsonDocument.Parse(json);
        var root = jsonDoc.RootElement;
        var stream = root.GetProperty("streams").EnumerateArray().FirstOrDefault(x => x.GetProperty("codec_type").GetString() == "video");
        var codec_name = stream.GetProperty("codec_name").GetString();
        var width = stream.GetProperty("width").GetInt32();
        var height = stream.GetProperty("height").GetInt32();
        var duration = Convert.ToDouble(root.GetProperty("format").GetProperty("duration").GetString());
        var secondsPerFrame = 5;
        int totalFrames = (int)(duration / secondsPerFrame);
        var ffmpegparam = $"-hwaccel cuda";
        if (dic.TryGetValue(codec_name, out var codec))
        {
            ffmpegparam = $"{ffmpegparam} -c:v {codec}";
        }
        
        ffmpegparam = $"{ffmpegparam} -i {tmppath} -vf fps=1/{secondsPerFrame} -f rawvideo -pix_fmt rgb24 pipe:1";

        var process = RunCmd(_milvusService.FFmpeg, ffmpegparam);
        var frameSize = width * height * 3; // bgr24
        int frameIndex = 0;

        var currentBatch = new List<byte[]>();
        var allMeta = new List<FrameVector>();
        var currentMeta = new List<FrameVector>();
        //var allEmbeddingsList = new List<float[]>();
        var BATCH_SIZE = 128;
        while (true)
        {
            byte[] buffer = new byte[frameSize];
            int totalRead = 0;
            while (totalRead < frameSize)
            {
                int read = await process.StandardOutput.BaseStream.ReadAsync(buffer.AsMemory(totalRead, frameSize - totalRead), cancellationToken);
                if (read == 0) break;
                totalRead += read;
            }
            if (totalRead < frameSize) break;
            currentBatch.Add(buffer);
            currentMeta.Add(new FrameVector
            {
                Sha256 = videopath.sha256,
                VideoPath = videopath.realpath,
                FrameIndex = frameIndex,
                Timestamp = frameIndex * secondsPerFrame
            });

            if (currentBatch.Count >= BATCH_SIZE)
            {

                VideoProcess(videopath.realpath, frameIndex, totalFrames);
                using var batchTensor = PreprocessBatchFrames(currentBatch, width, height);
                using var embeddingsTensor = Encode_image(batchTensor);
                AddEmbeddingToMeta(currentMeta, embeddingsTensor);
                allMeta.AddRange(currentMeta);
                currentBatch.Clear();
                currentMeta.Clear();

            }

            frameIndex++;
        }
        if (currentBatch.Count > 0)
        {
            VideoProcess(videopath.realpath, frameIndex, totalFrames);
            using var obj = PreprocessBatchFrames(currentBatch, width, height);
            using var lastembedding = Encode_image( obj);
            AddEmbeddingToMeta(currentMeta, lastembedding);
            allMeta.AddRange(currentMeta);
            currentBatch.Clear();
            currentMeta.Clear();
        }
        if (allMeta?.Count > 0)
        {
            var file_path = new List<string>();
            var embeddings = new List<ReadOnlyMemory<float>>();
            var timestamp = new List<float>();
            var sha256 = new List<string>();

            foreach (var meta in allMeta)
            {
                sha256.Add(meta.Sha256);
                file_path.Add(meta.VideoPath);

                embeddings.Add(new ReadOnlyMemory<float>(meta.Embedding));
                timestamp.Add(meta.Timestamp);
            }

           await _milvusService.MilvusCollection.InsertAsync([
                FieldData.Create("path_sha256", sha256),
  FieldData.Create("file_path", file_path),
        FieldData.Create("timestamp", timestamp),
        FieldData.CreateFloatVector("embedding", embeddings),

        ], cancellationToken: cancellationToken);
        }
        await process.WaitForExitAsync(cancellationToken);


    }
    void VideoProcess(string videopath, int frameIndex, int totalFrames) =>
       _logger.LogInformation($"Video {videopath}: processed {frameIndex}/{totalFrames} frames");

    static Tensor PreprocessBatchFrames(List<byte[]> buffers, int width, int height)
    {
        // 每帧 GPU 预处理
        var tensors = buffers.Select(buf => PreprocessFrameGpu(buf, width, height)).ToArray();

        // 拼接 batch
        var batch = cat(tensors, dim: 0); // shape: (B,3,224,224)
        return batch;
    }

    static Tensor PreprocessFrameGpu(byte[] buffer, int width, int height)
    {
        // CLIP 的 mean/std
        var mean = torch.tensor([0.48145466f, 0.4578275f, 0.40821073f], device: CUDA).reshape(1, 3, 1, 1);
        var std = torch.tensor([0.26862954f, 0.26130258f, 0.27577711f], device: CUDA).reshape(1, 3, 1, 1);

        // 1) 将 byte[] buffer 转 float 并归一化到 [0,1]
        float[] floatBuffer = new float[width * height * 3];
        for (int i = 0; i < buffer.Length; i++)
            floatBuffer[i] = buffer[i] / 255f;

        // 2) 转 Tensor (1,H,W,3) -> (1,3,H,W)
        var tensor = torch.tensor(floatBuffer, dtype: float32)
                        .reshape(1, height, width, 3)
                        .permute(0, 3, 1, 2) // NHWC -> NCHW
                        .to(CUDA);

        // 3) Resize 短边 = 256
        float scale = 256f / Math.Min(height, width);
        int newH = (int)Math.Round(height * scale);
        int newW = (int)Math.Round(width * scale);

        tensor = nn.functional.interpolate(
            tensor,
            size: [newH, newW],
            mode: InterpolationMode.Bilinear,// "bilinear",
            align_corners: false
        );

        // 4) 中心裁剪 224x224
        int cropH = 224, cropW = 224;
        int cropY = Math.Max(0, (newH - cropH) / 2);
        int cropX = Math.Max(0, (newW - cropW) / 2);

        tensor = tensor.index([
            TensorIndex.Slice(), // batch
        TensorIndex.Slice(), // channel
        TensorIndex.Slice(cropY, cropY + cropH),
        TensorIndex.Slice(cropX, cropX + cropW)
        ]);

        // 5) Normalize
        tensor = (tensor - mean) / std;

        return tensor; // shape: (1,3,224,224), GPU 上
    }
    async   Task<string> RunCmdWithOutput(string fileName, string arguments, CancellationToken token)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                UseShellExecute = false,
            },
            EnableRaisingEvents = true
        };

        process.Start();

        // 读取输出
        string output = await process.StandardOutput.ReadToEndAsync(token);
        string error = await process.StandardError.ReadToEndAsync(token);
        if (!string.IsNullOrEmpty(error))
        {
           _logger.LogWarning("ffprobe stderr: " + error);
        }
        await process.WaitForExitAsync(token);
        return output;
    }

    static Process RunCmd(string FileName, string Arguments)
    {
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = FileName,
                Arguments = Arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                UseShellExecute = false,  
                CreateNoWindow = true
            },
            EnableRaisingEvents = true
        };

        process.Start(); 

        // 绑定事件
        process.ErrorDataReceived += (s, e) =>
        {

        };
        process.BeginErrorReadLine();  

        return process;
    }


    static async Task<List<(string sha256, string realpath,string virtualDisk)>> ProcessFolder(string path,string virtualDisk,CancellationToken cancellationToken)
    {
        var bdmv = "BDMV";
        var special = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".srt", ".ass", ".txt", ".json", ".wav" };
        var list = new List<string>();
        if (!string.IsNullOrEmpty(virtualDisk))
        {
            list.Add($"{Bluray}{path}");
        }
        else
        {
            var pathlist = Directory.EnumerateFileSystemEntries(path, "*.*", SearchOption.AllDirectories);

            // 第一次遍历：识别所有蓝光根目录（包含BDMV文件夹的目录）
            var blurayRoots = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var entry in pathlist)
            {
                if (Directory.Exists(entry))
                {
                    string dirName = Path.GetFileName(entry);
                    if (dirName.Equals(bdmv, StringComparison.OrdinalIgnoreCase))
                    {
                        string parentDir = Directory.GetParent(entry)?.FullName;
                        if (parentDir != null && !blurayRoots.Contains(parentDir))
                        {
                            blurayRoots.Add(parentDir);
                            list.Add($"{Bluray}{parentDir}"); // 添加蓝光根目录
                        }
                    }
                }
            }

            // 第二次遍历：处理文件
            foreach (var entry in pathlist)
            {
                if (System.IO.File.Exists(entry)) // 只处理文件
                {
                    string ext = Path.GetExtension(entry).ToLowerInvariant();
                    // 跳过特殊后缀文件
                    if (special.Contains(ext))
                        continue;

                    // 检查文件是否在蓝光根目录下
                    bool isUnderBluray = blurayRoots.Any(root =>
                        entry.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase));

                    if (isUnderBluray)
                        continue; // 跳过蓝光根目录内的所有文件

                    // 处理ISO文件或普通视频文件
                    if (ext == ".iso")
                        list.Add($"{Bluray}{entry}"); // 添加ISO文件路径
                    else
                        list.Add(entry); // 添加普通视频文件路径
                }
            }
        }
        return [.. await Task.WhenAll(list.Select(x => Sha256(x, virtualDisk, cancellationToken)))];
    }
    static async Task<(string sha, string realpath,string virtualDisk)> Sha256(string path,string virtualDisk, CancellationToken cancellationToken)
    {
        var root = Directory.GetDirectoryRoot(path);

        char[] separators = [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar];
        var pathSegments = string.Join("", path.Replace(root, "").Split(separators, StringSplitOptions.RemoveEmptyEntries).Select(x => x));
        return (Convert.ToHexStringLower(await SHA256.Create().ComputeHashAsync(new MemoryStream(Encoding.UTF8.GetBytes(pathSegments)), cancellationToken)), path,virtualDisk);
    }

}
public class WebApiExceptionFilter : IAsyncExceptionFilter
{
    public Task OnExceptionAsync(ExceptionContext context)
    {
        context.Result = new BadRequestObjectResult(new
        {
            success = false,
            message = context.Exception.Message,
            stackTrace = context.Exception.StackTrace // 可选
        });
        return Task.CompletedTask;
    }
}


public class FrameVector
{
    public string Sha256 { get; set; }
    public string VideoPath { get; set; }

    public int FrameIndex { get; set; }

    public float Timestamp { get; set; }

    public float[] Embedding { get; set; }
}
