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
using Tokenizers.DotNet;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace videoprocess.Controllers;

[TypeFilter(typeof(WebApiExceptionFilter))]
[Route("api/[controller]/[action]")]
[Produces("application/json")]
[ApiController]

public class VideoController(MilvusImageService milvusService, ILogger<VideoController> logger, IServer server) : ControllerBase
{
   // Tokenizer Token=
   public Tokenizer Token { get; set; }
    private readonly MilvusImageService _milvusService = milvusService;
    private readonly ILogger<VideoController> _logger = logger;
    private readonly IServer _server = server;
    static JsonDocumentOptions JsonDocumentOptions => new()
    {
        AllowTrailingCommas = true,
        CommentHandling = JsonCommentHandling.Skip,
        MaxDepth = 64,
    };
    static JsonSerializerOptions JsonSerializerOptions=>new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        WriteIndented = false
    };
    //Tokenizer Token => new(Path.Combine(_milvusService.OpenApi.Token, "tokenizer.json"));
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
    public async Task<IActionResult> SubittleProcessAsync(List<Dictionary<string, string>> pathList, CancellationToken cancellationToken)
    {
        Token??= new(Path.Combine(_milvusService.OpenApi.Token, "tokenizer.json"));
        var list = (await Task.WhenAll(pathList.Select(x => ProcessFolder(x["path"], x["virtualDisk"], cancellationToken, true, "")))).SelectMany(x => x);
        if (!list.Any())
        {
            return Ok("not found any video file, skipping...");
        }
        var m3u = new ConcurrentBag<string>();
        var m3upath = Path.Combine(Directory.GetCurrentDirectory(),  "wavfile.m3u");
        if (FileExists(m3upath))
        {
            System.IO.File.Delete(m3upath);
        }
        var wavfile = "wav";
        var jsonfile = "json";
        var srtfile = "srt";
        var assfile = "ass";
        var dic = new List<Dictionary<string, List<(string filepath, string savepath,string virtualDisk)>>>();
        foreach (var (sha256, realpath, virtualDisk) in list)
        {
            var tmppath = realpath.Replace(Bluray, "");
            var wav = "";
            var json = "";
            var srt = "";
            var ass = "";
            if (FileExists(tmppath))
            {
                wav = Path.ChangeExtension(realpath, wavfile);
                json = Path.ChangeExtension(realpath, jsonfile);
                srt = Path.ChangeExtension(realpath, srtfile);
                ass = Path.ChangeExtension(realpath, assfile);
            }
            else
            {
              
                var filename = Path.GetFileName(tmppath);
                wav = Path.Combine(tmppath, $"{filename}.{wavfile}"  );
                json = Path.Combine(tmppath, $"{filename}.{jsonfile}");
                srt = Path.Combine(tmppath, $"{filename}.{srtfile}");
                ass = Path.Combine(tmppath, $"{filename}.{assfile}");
            }
            if (FileExists(srt) || FileExists(ass))
            {
                continue;
            }
            dic.Add(new Dictionary<string, List<(string filepath, string savepath,string virtualDisk)>>
            {
                [wavfile] =
    [
        (realpath, wav,virtualDisk)
    ],
                [jsonfile] =
    [
        (json,srt,"" )
    ]
            });
        }
        var wavlist = dic.Where(x => x.TryGetValue(wavfile, out var list) && list.Count > 0).Select(x => x[wavfile]).SelectMany(x => x).DistinctBy(x => x.savepath).Where(x =>
        {
            var wavexists = FileExists(x.savepath);
            if (wavexists &&  !FileExists(Path.ChangeExtension(x.savepath,jsonfile)))
            {
                m3u.Add(x.savepath);
            }
            return !wavexists;
        });
        var jsonlist = dic.Where(x => x.TryGetValue(jsonfile, out var list) && list.Count > 0).Select(x => x[jsonfile]).SelectMany(x => x).DistinctBy(x => x.savepath).Where(x => !FileExists(x.savepath)).ToList();
        if (wavlist?.Count() > 0)
        {
            await Parallel.ForEachAsync(wavlist, new ParallelOptions() { CancellationToken = cancellationToken }, async (x, _) =>
            {
                try
                {
                  
                    await GetWavFileByVideo(x.filepath, x.savepath, x.virtualDisk, _);
                    m3u.Add(x.savepath);
                }
                catch (System.Exception ex)
                {

                    _logger.LogError(exception: ex, $"LogError: {x}");
                }
            });
        }
        if (!m3u.IsEmpty)
        {

            await System.IO.File.WriteAllLinesAsync(m3upath, m3u, cancellationToken);
            await GetJsonFile(m3upath, cancellationToken);
            // 修正后的新增任务逻辑
            var newjson = m3u
                .Where(x => FileExists(Path.ChangeExtension(x, jsonfile)))
                .Select(x => (
                    filepath: Path.ChangeExtension(x, jsonfile),
                    savepath: Path.ChangeExtension(x, srtfile),
                    virtualDisk: ""
                ))
                .Where(x => !jsonlist.Any(item => item.savepath == x.savepath)) // 去重
                .ToList();

            if (newjson.Count != 0)
            {
                jsonlist.AddRange(newjson);
            }

        }
        if (jsonlist?.Count > 0)
        {
            await Parallel.ForEachAsync(jsonlist, new ParallelOptions() { CancellationToken = cancellationToken }, async (x, _) =>
           {
               try
               {
                   await GetSrtFile(x.filepath, x.savepath, _);

               }
               catch (Exception ex)
               {

                   _logger.LogError(exception: ex, $"LogError: {x}");
               }
           });
        }
        return Ok();
    }
    private static bool FileExists(string path) => System.IO.File.Exists(path);
    async Task GetSrtFile(string jsonpath, string srtpath, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"json file {jsonpath} is begin");
        var RootElement = JsonDocument.Parse(await System.IO.File.ReadAllTextAsync(jsonpath, Encoding.UTF8, cancellationToken), JsonDocumentOptions).RootElement;
        var segments = RootElement.GetProperty("segments").EnumerateArray();

        // 解析原始数据并存储为元组列表 (text, start, end)
        var items = segments
            .Select(s => (
                text: s.GetProperty("text").GetString().Trim(),
                start: SecondsToSrtTime(s.GetProperty("start").GetDouble()),
                end: SecondsToSrtTime(s.GetProperty("end").GetDouble())
            ))
            .ToList();

        // 相邻去重：连续重复文本只保留第一条
        var filteredItems = new List<(string text, string start, string end)>();
        string lastText = null;
        foreach (var item in items)
        {
            if (item.text != lastText)
            {
                filteredItems.Add(item);   // 文本不同时保留
                lastText = item.text;      // 更新最后记录的文本
            }
        }

        // 重新编号并转换为字典
        var origin = filteredItems
            .Select((x, idx) => new
            {
                id = idx + 1,
                x.text,
                time = $"{x.start} --> {x.end}"
            })
            .ToDictionary(k => k.id, v => (v.text, v.time));
        var fulltext = RootElement.GetProperty("text").GetString();
        var systemtoken = GetTokenCounts(_milvusService.SystemPrompt);
        var fulltexttoken = GetTokenCounts(fulltext);
        _logger.LogInformation($"token end");
        
        //return;
        int tokenBudget = _milvusService.OpenApi.MaxTokens - (systemtoken + fulltexttoken);
        var batches = new List<string>();
        var batch = new List<KeyValuePair<int, (string text, string time)>>();
        foreach (var kv in origin)
        {
            batch.Add(kv);

            // 每次加一条后，尝试序列化
            var json = JsonSerializer.Serialize(
    batch.Select(x => new { id = x.Key, x.Value.text }), JsonSerializerOptions);
            int sendtoken = GetTokenCounts(json);
            if (sendtoken > tokenBudget)
            {
                // 回退掉最后一条
                batch.RemoveAt(batch.Count - 1);

                // 保存前一批
                if (batch.Count > 0)
                {
                    var finalized = JsonSerializer.Serialize(
                        batch.Select(x => new { id = x.Key, x.Value.text }), JsonSerializerOptions);
                    batches.Add(finalized);
                    batch.Clear();
                }

                // 重新处理当前 kv（因为它自己可能能单独成一个 batch）
                batch.Add(kv);
            }
        }
        if (batch.Count > 0)
        {
            var finalized = JsonSerializer.Serialize(
    batch.Select(x => new { id = x.Key, x.Value.text }), JsonSerializerOptions);
            batches.Add(finalized);
        }
        var outputList = new List<string>();
        foreach (var json in batches)
        {
            var usercontent = $"翻译字幕 {json}";
            var send = _milvusService.OpenApi.Request.DeepClone();
            var sendmessage = send["messages"].AsArray();
            sendmessage.Add(new JsonObject
            {
                ["role"] = "user",
                ["content"] = $"字幕的全文{Environment.NewLine}{fulltext}"
            });
            sendmessage.Add(new JsonObject
            {
                ["role"] = "user",
                ["content"] = usercontent
            });

            logger.LogInformation($"send ai {usercontent}");
            
            var success = false;
            while (!success)
            {
                try
                {
                    using var response = await PostJsonAsync<Stream>(_milvusService.OpenApi.Chat, send, cancellationToken);
                    if (response == null)
                    {
                        logger.LogWarning($"response is null {jsonpath}");
                        return;
                    }
                    using var doc = await JsonDocument.ParseAsync(response, JsonDocumentOptions, cancellationToken: cancellationToken);
                    var content = doc.RootElement.GetProperty("choices").EnumerateArray().FirstOrDefault().GetProperty("message").GetProperty("content").GetString();
                    System.Console.WriteLine($"result {content}");
                    content = Regex.Replace(content, "<think>.*?</think>", "", RegexOptions.Singleline).Replace("```json", "").Replace("```", "");
                    var airoot = JsonDocument.Parse(content, JsonDocumentOptions).RootElement.EnumerateArray().ToDictionary(key => key.GetProperty("id").GetInt32(), value => value.GetProperty("text").GetString().Trim());
                    outputList.Add(string.Join(Environment.NewLine, origin.Select((x, index) => $"{index + 1}{Environment.NewLine}{x.Value.time}{Environment.NewLine}{airoot[x.Key]}{Environment.NewLine}")));
                    success = true;

                }
                catch (OperationCanceledException) 
                {
                    logger.LogInformation("请求已被取消");
                    return; 
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine(ex);
                    success = false;
                }
            }



        }

        if (outputList?.Count > 0)
        {
            await System.IO.File.WriteAllLinesAsync(srtpath, outputList, Encoding.UTF8, cancellationToken);
        }
        _logger.LogInformation($" json file {jsonpath} is end");
    } 
    int GetTokenCounts(string content)
    {

        var tokens = Token.Encode(content);
        return tokens.Length;
    }
    static string SecondsToSrtTime(double seconds)
    {
        int hours = (int)(seconds / 3600);
        int minutes = (int)(seconds % 3600 / 60);
        int secs = (int)(seconds % 60);
        int millis = (int)((seconds - Math.Floor(seconds)) * 1000);
        return $"{hours:D2}:{minutes:D2}:{secs:D2},{millis:D3}";
    }
    async Task GetJsonFile(string m3upath, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"m3u file {m3upath} is begin");

        var model_dir = $"\"{_milvusService.FasterWhisper.model}\"";
        var cmd = RunCmd(_milvusService.FasterWhisper.path, $" {m3upath} -l Japanese -m large-v2 --model_dir {model_dir}  --no_speech_threshold 0.3  --vad_threshold 0.3 --standard  --output_dir source --output_format json --beep_off --skip");
        await cmd.WaitForExitAsync(cancellationToken);
        int exitCode = cmd.ExitCode;
        _logger.LogInformation($"Process exited with code: {exitCode}");

        if (exitCode != 0)
        {
            _logger.LogWarning($"Third-party program exited with error code: {exitCode}");
        }
        _logger.LogInformation($"m3u file {m3upath} is end");
    }
    async Task GetWavFileByVideo(string filepath, string savepath,string virtualDisk, CancellationToken cancellationToken )
    {
        _logger.LogInformation($"video file {filepath} is begin");
        var tmppath = $"\"{filepath}\"";
        if (!string.IsNullOrEmpty(virtualDisk))
        {
            var tmpvirtualDisk = $"{Bluray}{virtualDisk}";
            tmppath = $"\"{tmpvirtualDisk}\"";
        }
 
        savepath= $"\"{savepath}\"";
        var cmd= RunCmd(_milvusService.FFmpeg, $"-y -i {tmppath}  -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le {savepath}");
        await cmd.WaitForExitAsync(cancellationToken);
        _logger.LogInformation($"video file {filepath} is end");
    }
    public async Task<IActionResult> VideoProcessAsync(List<Dictionary<string, string>> pathList, CancellationToken cancellationToken)
    {
        var videopath = (await Task.WhenAll(pathList.Select(x => ProcessFolder(x["path"], x["virtualDisk"], cancellationToken, true)))).SelectMany(x => x);
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
                CancellationToken = cancellationToken
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

        await Parallel.ForEachAsync(videopath, new ParallelOptions() { MaxDegreeOfParallelism = _milvusService.MaxDegreeOfParallelism, CancellationToken = cancellationToken }, async (file, ct) =>
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
    public async Task<IActionResult> SearchSimilarFrameAsync(string imagePath, CancellationToken cancellationToken, int limit = 5)
    {
        _logger.LogInformation("req");
        Image<Rgb24> image;
        if (imagePath.StartsWith("http") || imagePath.StartsWith("https"))
        {
            var sockslist = _milvusService.Socks.Split(":");
            var proxy = new HttpToSocks5Proxy(sockslist[0], Convert.ToInt32(sockslist[1]));
            var handler = new HttpClientHandler
            {
                Proxy = proxy,
                UseProxy = true
            };
            using var client = new HttpClient(handler);
            var res = await client.GetAsync(imagePath);
            if (!res.IsSuccessStatusCode)
            {
                return BadRequest(res);
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
        var embeddingTensor = Encode_image(tensor);

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



    Tensor Encode_image(Tensor tensor)
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




    async Task LoadVideo((string sha256, string realpath, string virtualDisk) videopath, CancellationToken cancellationToken)
    {
        var tmppath = $"\"{videopath.realpath}\"";
        if (!string.IsNullOrEmpty(videopath.virtualDisk))
        {
            var virtualDisk = $"{Bluray}{videopath.virtualDisk}";
            tmppath = $"\"{virtualDisk}\"";
        }
        var ffprobeparam = @$"-v error -show_entries format=duration -show_entries stream=codec_name,codec_type,width,height,r_frame_rate -of json {tmppath}";
        var json = await RunCmdWithOutput(_milvusService.FFprobe, ffprobeparam, cancellationToken);
        using var jsonDoc = JsonDocument.Parse(json, JsonDocumentOptions);
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

        var process = RunCmd(_milvusService.FFmpeg, ffmpegparam,false);
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
            using var lastembedding = Encode_image(obj);
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
    async Task<string> RunCmdWithOutput(string fileName, string arguments, CancellationToken token)
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

    Process RunCmd(string FileName, string Arguments, bool useEventLog = true)
    {
        _logger.LogInformation($"{FileName} {Arguments}");
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

        if (useEventLog)
        {
            // 事件式输出
            process.OutputDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    _logger.LogInformation($"Output: {e.Data}");
            };
            process.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    _logger.LogInformation($"LogError: {e.Data}");
            };

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

        }
        return process;
    }

        static async Task<List<(string sha256, string realpath, string virtualDisk)>> ProcessFolder(string path, string virtualDisk, CancellationToken cancellationToken, bool ComputeHash, string filter = ".srt,.ass,.txt,.json,.wav,.text")
        {
            var bdmv = "BDMV";
            var special = new HashSet<string>([.. filter.Split(",")], StringComparer.OrdinalIgnoreCase);
            var list = new List<string>();
            if (!string.IsNullOrEmpty(virtualDisk))
            {
                list.Add($"{Bluray}{path}");
            }
            else
            {
            if (System.IO.File.Exists(path))
            {
                string ext = Path.GetExtension(path).ToLowerInvariant();
                if (ext == ".iso")
                {
                    list.Add($"{Bluray}{path}"); // 添加ISO文件路径
                }
                else
                {
                    list.Add(path); // 添加普通视频文件路径
                }
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
               
            }
            return ComputeHash ? [.. await Task.WhenAll(list.Select(x => Sha256(x, virtualDisk, cancellationToken)))] : list.Select(x => ("", x, virtualDisk)).ToList();
        }
    static async Task<(string sha, string realpath, string virtualDisk)> Sha256(string path, string virtualDisk, CancellationToken cancellationToken)
    {
        var root = Directory.GetDirectoryRoot(path);

        char[] separators = [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar];
        var pathSegments = string.Join("", path.Replace(root, "").Split(separators, StringSplitOptions.RemoveEmptyEntries).Select(x => x));
        return (Convert.ToHexStringLower(await SHA256.Create().ComputeHashAsync(new MemoryStream(Encoding.UTF8.GetBytes(pathSegments)), cancellationToken)), path, virtualDisk);
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
