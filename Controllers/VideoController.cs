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
    static string Bluray => "bluray:";



    public async Task<IActionResult> SubittleProcessAsync(List<Dictionary<string, string>> pathList, CancellationToken cancellationToken)
    {
       
        var list = pathList.Select(x => ProcessFolder(x["path"], x["virtualDisk"])).SelectMany(x => x).ToList();
        if (list.Count == 0)
        {
            return Ok("not found any video file, skipping...");
        }
        var m3u = new ConcurrentBag<string>();
        var m3upath = Path.Combine(Directory.GetCurrentDirectory(),  "wavfile.m3u");
        System.IO.File.Delete(m3upath);
        var wavfile = ".wav";
        var jsonfile = ".json";
        var srtfile = ".srt";
        var assfile = ".ass";
        var wavlist = new List<(string inputpath, string savepath, string bluraypath)>();
        var addjson = new List<(string savepath, string srtpath)>();
        var finallyjson= new List<(string savepath, string srtpath)>();
        foreach (var (path, bluraypath, isfile) in list)
        {
            var wav = Path.ChangeExtension(path, wavfile);
            var json = Path.ChangeExtension(path, jsonfile);
            var srt = Path.ChangeExtension(path, srtfile);
            var ass = Path.ChangeExtension(path, assfile);
            if (!isfile)
            {
                var filename = Path.GetFileName(path);
                wav = Path.Combine(path, $"{filename}{wavfile}");
                json = Path.Combine(path, $"{filename}{jsonfile}");
                srt = Path.Combine(path, $"{filename}{srtfile}");
                ass = Path.Combine(path, $"{filename}{assfile}");
            }
            if (FileExists(srt) || FileExists(ass))
            {
                continue;
            }
            if (FileExists(json))
            {
                finallyjson.Add((json, srt));
                continue;
            }

            if (FileExists(wav))
            {
                m3u.Add(wav);
            }
            else
            {
                wavlist.Add((path, wav, bluraypath));
            }
        }
        await Parallel.ForEachAsync(wavlist, new ParallelOptions() { CancellationToken = cancellationToken }, async (x, _) =>
             {
                 try
                 {

                     await GetWavFileByVideo(x.inputpath, x.savepath, x.bluraypath, _);
                     m3u.Add(x.savepath);
                 }
                 catch (System.Exception ex)
                 {

                     _logger.LogError(exception: ex, $"LogError: {x}");
                 }
             });
        if (!m3u.IsEmpty)
        {

            await System.IO.File.WriteAllLinesAsync(m3upath, m3u, cancellationToken);
            await GetJsonFile(m3upath, cancellationToken);
            addjson.AddRange(m3u
                  .Where(x => FileExists(Path.ChangeExtension(x, jsonfile)))
                  .Select(x => (
                        Path.ChangeExtension(x, jsonfile),
                        Path.ChangeExtension(x, srtfile)
                      
                  )));

        }
        List<(string savepath, string srtpath)> jsonlist = [.. addjson, .. finallyjson];
        await Parallel.ForEachAsync(jsonlist, new ParallelOptions() { CancellationToken = cancellationToken }, async (x, _) =>
             {
                 try
                 {
                     await GetSrtFile(x.savepath, x.srtpath, _);

                 }
                 catch (Exception ex)
                 {

                     _logger.LogError(exception: ex, $"LogError: {x}");
                 }
             });
        var dic= jsonlist.Where(x=>FileExists(x.srtpath) ).Select(x=> Path.ChangeExtension(x.srtpath, null)).ToHashSet();
        List<string> removelist= [.. wavlist.Select(x =>   x.savepath), .. jsonlist.Select(x => x.savepath)];
       removelist.Where(x=> dic.Contains(Path.ChangeExtension(x, null))).ToList().ForEach(System.IO.File.Delete);
        return Ok();
    }
    private static bool FileExists(string path) => System.IO.File.Exists(path);
    async Task GetSrtFile(string jsonpath, string srtpath, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"json file {jsonpath} is begin");
        var RootElement = JsonElement.Parse(await System.IO.File.ReadAllTextAsync(jsonpath, Encoding.UTF8, cancellationToken), JsonDocumentOptions);
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

          var fulltext = string.Join("", origin.Values.Select(v => v.text));
        var rawBatches = origin
            .Chunk( _Service.OpenApi.Chunk)
            .Select(chunk => chunk.ToList())
            .ToList();

        var batches = rawBatches
            .Select(chunk => JsonSerializer.Serialize(
                chunk.Select(x => new { id = x.Key, x.Value.text }), JsonSerializerOptions))
            .ToList();
   

        var outputList = new List<string>();
        for (int i = 0; i < batches.Count; i++)
        {
            var usercontent = $"{batches[i]}";
            var send = _Service.OpenApi.Request.DeepClone();
            if (batches.Count > 1)
            {
                send["messages"].AsArray().Add(new JsonObject
                {
                    ["role"] = "user",
                    ["content"] = $"字幕的全文{Environment.NewLine}{fulltext}"
                });
            }

            send["messages"].AsArray().Add(new JsonObject
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
                    using var response = await PostJsonAsync<Stream>(_Service.OpenApi.Chat, send, _Service.OpenApi.Headers, cancellationToken);
                    if (response == null)
                    {
                        logger.LogWarning($"response is null {jsonpath}");
                        return;
                    }

                    using var doc = await JsonDocument.ParseAsync(response, JsonDocumentOptions, cancellationToken: cancellationToken);
                    var firstChoice = doc.RootElement.GetProperty("choices").EnumerateArray().FirstOrDefault();
                    var message = firstChoice.GetProperty("message");

                    string content = string.Empty;

                    // 1. 优先校验并提取 tool_calls (Function Calling)
                    if (message.TryGetProperty("tool_calls", out var toolCalls) &&
                        toolCalls.ValueKind == JsonValueKind.Array &&
                        toolCalls.GetArrayLength() > 0)
                    {
                        var firstToolCall = toolCalls[0];
                        if (firstToolCall.TryGetProperty("function", out var functionObj) &&
                            functionObj.TryGetProperty("arguments", out var argumentsProp))
                        {
                            content = argumentsProp.GetString() ?? string.Empty;
                        }
                    }
                    // 2. 如果没有 tool_calls，则回退读取普通的 content 文本
                    else if (message.TryGetProperty("content", out var contentProp))
                    {
                        content = contentProp.GetString() ?? string.Empty;
                        if (!string.IsNullOrEmpty(content))
                        {
                            content = Regex.Replace(content, "<think>.*?</think>", "", RegexOptions.Singleline)
                                           .Replace("```json", "")
                                           .Replace("```", "")
                                           .Trim();
                        }
                    }

                    System.Console.WriteLine($"result: {content}");

                    var airoot = JsonElement.Parse(content, JsonDocumentOptions).EnumerateArray()
                        .ToDictionary(key => key.GetProperty("id").GetInt32(), value => value.GetProperty("text").GetString().Trim());

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
 
    static string SecondsToSrtTime(double seconds)
    {
        int hours = (int)(seconds / 3600);
        int minutes = (int)(seconds % 3600 / 60);
        int secs = (int)(seconds % 60);
        int millis = (int)((seconds - Math.Floor(seconds)) * 1000);
        return $"{hours:D2}:{minutes:D2}:{secs:D2},{millis:D3}";
    }




    [DllImport(Kernel32)]
    static extern uint SetErrorMode(uint uMode);

    [DllImport(Kernel32, SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool SetProcessShutdownParameters(uint dwLevel, uint dwFlags);
    const uint SEM_NOGPFAULTERRORBOX = 0x0002; // 阻止崩溃对话框
    const uint SEM_FAILCRITICALERRORS = 0x0001; // 阻止磁盘错误对话框
    const uint SHUTDOWN_NORETRY = 0x1; // 防止自动重启
    async Task GetJsonFile(string m3upath, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"m3u file {m3upath} is begin");

        //var model_dir = $"\"{_Service.FasterWhisper.model}\"";
        await RunCmdAsync(_Service.FasterWhisper.path, [m3upath, "-l", "Japanese", "-m", "large-v2", "--model_dir", _Service.FasterWhisper.model, "--no_speech_threshold", "0.3", "--vad_method", "pyannote_v3", "--vad_threshold", "0.3", "--standard", "--output_dir", "source", "--output_format", "json", "--beep_off", "--skip"], cancellationToken);
       
        _logger.LogInformation($"m3u file {m3upath} is end");
    }
    async Task GetWavFileByVideo(string inputpath, string savepath,string bluraypath, CancellationToken cancellationToken )
    {
        _logger.LogInformation($"video file {inputpath} is begin");
        //var tmppath = $"\"{filepath}\"";
        if (!string.IsNullOrEmpty(bluraypath))
        {
            inputpath = bluraypath;
            //var tmpvirtualDisk = $"{Bluray}{virtualDisk}";
            //tmppath = $"\"{tmpvirtualDisk}\"";
        }
 
        //savepath= $"\"{savepath}\"";
        await RunCmdAsync(_Service.FFmpeg, ["-y", "-i", inputpath, "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", savepath], cancellationToken);
      
        _logger.LogInformation($"video file {inputpath} is end");
    }



    const string Kernel32 = "kernel32.dll";

    async Task  RunCmdAsync(string fileName, List<string> ArgumentList, CancellationToken token)
    {
        _logger.LogInformation($"{fileName} {string.Join(" ", ArgumentList)}");
        uint originalMode = SetErrorMode(0);
        // 2. 启用崩溃抑制 + 关键错误抑制
        _ = SetErrorMode(originalMode | SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);

        // 3. 防止系统在崩溃后自动重启进程
        SetProcessShutdownParameters(0x100 | SHUTDOWN_NORETRY, 0);
        await Process.RunAsync(fileName, ArgumentList,true, token);
        _ = SetErrorMode(originalMode);

    }
 
        static    List<(string path, string bluraypath,bool isfile)> ProcessFolder(string path, string virtualDisk)
        {
            var bdmv = "BDMV";
           var iso= ".iso";
            var list = new List<(string path, string bluraypath,bool isfile)>();
         if (FileExists(path))
        {
            return (Path.GetExtension(path).Equals(iso, StringComparison.InvariantCultureIgnoreCase), !string.IsNullOrEmpty(virtualDisk)) switch
            {
                (false, false) => [(path, "", true)],
                (true, false) => [(path, $"{Bluray}{path}", true)],
                (true, true) => [(path, $"{Bluray}{virtualDisk}", true)],
                _ => list,
            };
        }
        if (Path.GetFileName(
    Path.TrimEndingDirectorySeparator(path))
    .Equals(bdmv, StringComparison.OrdinalIgnoreCase))
        {
            string parentDir = Directory.GetParent(path).FullName;
            return [(parentDir,$"{Bluray}{parentDir}",  false)];
        }
        var pathlist = Directory.EnumerateFileSystemEntries(path, "*.*", SearchOption.AllDirectories).ToList();
        list.AddRange(pathlist.Where(x => Directory.Exists(x) && Path.GetFileName(x).Equals(bdmv, StringComparison.OrdinalIgnoreCase)).Select(x =>
        {
            string parentDir = Directory.GetParent(x).FullName;
            return(parentDir, $"{Bluray}{parentDir}",false);
        }));
        var blurayRoots =list.Select(x=>x.path).ToHashSet(StringComparer.OrdinalIgnoreCase);
        list.AddRange( pathlist.Where(x => FileExists(x)&& !blurayRoots.Any(root =>
                         x.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))).Select(x => (x, Path.GetExtension(x).Equals(iso, StringComparison.InvariantCultureIgnoreCase) ? $"{Bluray}{x}" : "", true)));
          return list;

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
 


