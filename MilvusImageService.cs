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

public class MilvusImageService(ILogger<MilvusImageService> logger, IConfiguration configuration)
{
    private readonly ILogger<MilvusImageService> _logger = logger;
    private readonly IConfiguration _configuration = configuration;

    public MilvusClient MilvusClient { get; private set; }
    public MilvusCollection MilvusCollection { get; private set; }
    public ScriptModule ScriptModule { get; private set; }

    public string CollectionName { get; private set; }
    public string FFmpeg { get; private set; }
    public string FFprobe { get; private set; }
    public int MaxDegreeOfParallelism { get; private set; }
    public string Socks { get; private set; }

    // 异步初始化方法，放耗时操作
    public async Task InitAsync()
    {
        FFmpeg = _configuration["AppConfiguration:FFmpeg"];
        FFprobe = _configuration["AppConfiguration:FFprobe"];
        Socks = _configuration["AppConfiguration:Socks"];
        MaxDegreeOfParallelism = int.Parse(_configuration["AppConfiguration:MaxDegreeOfParallelism"] ?? "4");

        // 初始化 Milvus Client 和 Collection
        var milvusConf = _configuration.GetSection("AppConfiguration:MilvusClientConn");
        MilvusClient = new MilvusClient(milvusConf["Host"], int.Parse(milvusConf["Port"]));
        CollectionName = milvusConf["CollectionName"];
        MilvusCollection = MilvusClient.GetCollection(CollectionName);

        // 初始化 GPU 模型
        var modelPath = _configuration["AppConfiguration:ModelPath"];
        ScriptModule = jit.load(modelPath).to(torch.CUDA);
        ScriptModule.eval();

        _logger.LogInformation("MilvusImageService initialized: model and Milvus loaded.");
        await Task.CompletedTask;
    }
}
