using System.Collections.Concurrent;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;

using Microsoft.AspNetCore.Mvc.Filters;

using static WebApiHelper;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Hosting.Server;
using Tokenizers.DotNet;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace videoprocess.Controllers;

[TypeFilter(typeof(WebApiExceptionFilter))]
[Route("api/[controller]/[action]")]
[Produces("application/json")]
[ApiController]

public class VideoController(VideoProcessService VideoProcessService, ILogger<VideoController> logger ) : ControllerBase
{
   
    public Tokenizer Token { get; set; }
    private readonly VideoProcessService _Service = VideoProcessService;
    private readonly ILogger<VideoController> _logger = logger;

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
    //Tokenizer Token => new(Path.Combine(_Service.OpenApi.Token, "tokenizer.json"));
    static string Bluray => "bluray:";
  
   
  
  
    public async Task<IActionResult> SubittleProcessAsync(List<Dictionary<string, string>> pathList, CancellationToken cancellationToken)
    {
       
        Token??= new(Path.Combine(_Service.OpenApi.Token, "tokenizer.json"));
        var list = pathList.Select(x => ProcessFolder(x["path"], x["virtualDisk"])).SelectMany(x => x);
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
        foreach (var ( realpath, virtualDisk) in list)
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
        var systemtoken = GetTokenCounts(_Service.SystemPrompt);
        var fulltexttoken = GetTokenCounts(fulltext);
        _logger.LogInformation($"token end");

        //return;
        int tokenBudget = _Service.OpenApi.MaxTokens - (systemtoken + fulltexttoken);
        var batches = new List<string>();
        var rawBatches = new List<List<KeyValuePair<int, (string text, string time)>>>();
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
                      var finalized = JsonSerializer.Serialize(batch.Select(x => new { id = x.Key, x.Value.text }), JsonSerializerOptions);
                    batches.Add(finalized);
                    rawBatches.Add([.. batch]);
                      batch.Clear();
                  }

                  // 重新处理当前 kv（因为它自己可能能单独成一个 batch）
                  batch.Add(kv);
              }
          }
          if (batch.Count > 0)
          {
             var finalized = JsonSerializer.Serialize(batch.Select(x => new { id = x.Key, x.Value.text }), JsonSerializerOptions);
            batches.Add(finalized);
            rawBatches.Add([.. batch]);
        }

        var outputList = new List<string>();
 
        for (int i = 0; i < batches.Count; i++)
        {
  
            var usercontent = $"翻译字幕 {batches[i]}";
            var send = _Service.OpenApi.Request.DeepClone();
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

           

            var success = false;
            while (!success)
            {
                try
                {
                    logger.LogInformation($"send ai {usercontent}");
                    using var response = await PostJsonAsync<Stream>(_Service.OpenApi.Chat, send, cancellationToken);
                    if (response == null)
                    {
                        logger.LogWarning($"response is null {jsonpath}");
                        return;
                    }
                    //batch[i]
                    using var doc = await JsonDocument.ParseAsync(response, JsonDocumentOptions, cancellationToken: cancellationToken);
                    var content = doc.RootElement.GetProperty("choices").EnumerateArray().FirstOrDefault().GetProperty("message").GetProperty("content").GetString();
                    System.Console.WriteLine($"result {content}");
                    content = Regex.Replace(content, "<think>.*?</think>", "", RegexOptions.Singleline).Replace("```json", "").Replace("```", "");
                   
                    var airoot = JsonDocument.Parse(content, JsonDocumentOptions).RootElement.EnumerateArray().ToDictionary(key => key.GetProperty("id").GetInt32(), value => value.GetProperty("text").GetString().Trim());
                     outputList.Add(string.Join(Environment.NewLine, rawBatches[i].Select((x, index) => $"{index + 1}{Environment.NewLine}{x.Value.time}{Environment.NewLine}{airoot[x.Key]}{Environment.NewLine}")));
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
    // 在类中添加 P/Invoke 声明
    [DllImport("kernel32.dll")]
    static extern uint SetErrorMode(uint uMode);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool SetProcessShutdownParameters(uint dwLevel, uint dwFlags);

    const uint SEM_NOGPFAULTERRORBOX = 0x0002; // 阻止崩溃对话框
    const uint SEM_FAILCRITICALERRORS = 0x0001; // 阻止磁盘错误对话框
    const uint SHUTDOWN_NORETRY = 0x1; // 防止自动重启
    async Task GetJsonFile(string m3upath, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"m3u file {m3upath} is begin");

        var model_dir = $"\"{_Service.FasterWhisper.model}\"";
        await RunCmdAsync(_Service.FasterWhisper.path, $" {m3upath} -l Japanese -m large-v2 --model_dir {model_dir}  --no_speech_threshold 0.3 --vad_method pyannote_v3 --vad_threshold 0.3 --standard  --output_dir source --output_format json --beep_off --skip", cancellationToken);
       
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
        await RunCmdAsync(_Service.FFmpeg, $"-y -i {tmppath}  -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le {savepath}", cancellationToken);
      
        _logger.LogInformation($"video file {filepath} is end");
    }
 

  
   
 
    async Task  RunCmdAsync(string fileName, string arguments, CancellationToken token)
    {
        _logger.LogInformation($"{fileName} {arguments}");
        uint originalMode = SetErrorMode(0);
        // 2. 启用崩溃抑制 + 关键错误抑制
        SetErrorMode(originalMode | SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);

        // 3. 防止系统在崩溃后自动重启进程
        SetProcessShutdownParameters(0x100 | SHUTDOWN_NORETRY, 0);
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                UseShellExecute = false
            },
            EnableRaisingEvents = true
        };

        process.Start();

        // 读取输出
        var output =   process.StandardOutput.ReadToEndAsync(token);
        var error =   process.StandardError.ReadToEndAsync(token);
        /* var task=  await Task.WhenAll(output, error);
           if (!string.IsNullOrEmpty(task[0]))
            {
                _logger.LogInformation($"output: {task[0]}");
            }
            if (!string.IsNullOrEmpty(task[1]))
            {
                _logger.LogInformation($"error: {task[1]}");
            }*/
        await Task.WhenAll(output, error, process.WaitForExitAsync(token));
        SetErrorMode(originalMode);

    }
 
        static    List<(string realpath, string virtualDisk)> ProcessFolder(string path, string virtualDisk)
        {
            var bdmv = "BDMV";
           
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
            return [.. list.Select(x => (x, virtualDisk))];
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


