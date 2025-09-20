using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static WebApiHelper;
using Milvus.Client;
using static TorchSharp.torch.jit;
using static TorchSharp.torch;
using Microsoft.AspNetCore.Mvc.Routing;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Text;

public class MilvusImageService(ILogger<MilvusImageService> logger, IConfiguration configuration)
{
    private readonly ILogger<MilvusImageService> _logger = logger;
    private readonly IConfiguration _configuration = configuration;
public string SystemPrompt { get; set; }
    public MilvusClient MilvusClient { get; private set; }
    public MilvusCollection MilvusCollection { get; private set; }
    public ScriptModule ScriptModule { get; private set; }
 public (string Token,string Chat,int MaxTokens, JsonNode Request) OpenApi { get; set; }
    public (string path,string model) FasterWhisper  { get; set; }
    public string CollectionName { get; private set; }
    public string FFmpeg { get; private set; }
    public string FFprobe { get; private set; }
    public int MaxDegreeOfParallelism { get; private set; }
    public string Socks { get; private set; }

    // 异步初始化方法，放耗时操作
    public async Task InitAsync(CancellationToken cancellationToken)
    {
        SystemPrompt = await File.ReadAllTextAsync(Path.Combine(Directory.GetCurrentDirectory(), "SystemPrompt.txt"), Encoding.UTF8, cancellationToken);
        FFmpeg = _configuration["AppConfiguration:FFmpeg"];
        FFprobe = _configuration["AppConfiguration:FFprobe"];
        Socks = _configuration["AppConfiguration:Socks"];
        MaxDegreeOfParallelism = int.Parse(_configuration["AppConfiguration:MaxDegreeOfParallelism"] ?? "4");

        // 初始化 Milvus Client 和 Collection
        var milvusConf = _configuration.GetSection("AppConfiguration:MilvusClientConn");
        MilvusClient = new MilvusClient(milvusConf["Host"], int.Parse(milvusConf["Port"]));
        CollectionName = milvusConf["CollectionName"];
        MilvusCollection = MilvusClient.GetCollection(CollectionName);
        var FasterWhisperConf = _configuration.GetSection("AppConfiguration:FasterWhisper");
        FasterWhisper = (FasterWhisperConf["Path"], FasterWhisperConf["ModeDir"]);
        var OpenApiConf = _configuration.GetSection("AppConfiguration:OpenApi");
       
      
        var request = JsonNode.Parse(OpenApiConf.GetSection("Request").Value);

        var messages = request["messages"].AsArray();
        messages.Clear();
        messages.Add(new JsonObject
        {
            ["role"] = "system",
            ["content"] = SystemPrompt
        });
        OpenApi = (OpenApiConf["Token"], OpenApiConf["Chat"],Convert.ToInt32( OpenApiConf["MaxTokens"]), request);
       
        // 初始化 GPU 模型
        var modelPath = _configuration["AppConfiguration:ModelPath"];
        ScriptModule = jit.load(modelPath).to(torch.CUDA);
        ScriptModule.eval();

        _logger.LogInformation("MilvusImageService initialized: model and Milvus loaded.");
    
    }
}
