using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static WebApiHelper;
using Milvus.Client;
using static TorchSharp.torch.jit;
using static TorchSharp.torch;

public class MilvusImageService
{
    public int MaxDegreeOfParallelism { get; set; }
    private readonly ILogger<MilvusImageService> _logger;
    private readonly IConfiguration _configuration;
    public string FFmpeg;
    public string FFprobe;
    public MilvusClient _milvusClient;
    private MilvusCollection _milvusCollection;
    private ScriptModule _scriptModule;
    public string CollectionName;
    private bool _isLoaded = false;
    public string Query { get; set; }
    public string Load { get; set; }
    public string Socks { get; set; }      
    public MilvusImageService(ILogger<MilvusImageService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        FFmpeg = _configuration["AppConfiguration:FFmpeg"];
        FFprobe = _configuration["AppConfiguration:FFprobe"];
        // 初始化 Milvus Client 和 Collection
        var milvusConf = _configuration.GetSection("AppConfiguration:MilvusClientConn");
        _milvusClient = new MilvusClient(milvusConf["Host"], int.Parse(milvusConf["Port"]));
        Query = $"http://{milvusConf["Host"]}:{milvusConf["Port"]}{milvusConf["Query"]}";
        Load = $"http://{milvusConf["Host"]}:{milvusConf["Port"]}{milvusConf["Load"]}";
        CollectionName = milvusConf["CollectionName"];
        _milvusCollection = _milvusClient.GetCollection(CollectionName);
        MaxDegreeOfParallelism = Convert.ToInt32(_configuration["AppConfiguration:MaxDegreeOfParallelism"]);
        Socks= _configuration["AppConfiguration:Socks"];
        // 初始化 GPU 模型
        var modelPath = _configuration["AppConfiguration:ModelPath"];
        _scriptModule = jit.load(modelPath).to(torch.CUDA);
        _scriptModule.eval();

        _logger.LogInformation("MilvusImageService initialized: model and Milvus loaded.");
    }

    // 确保 Milvus 集合 Load 完毕
    public async Task EnsureCollectionLoadedAsync(CancellationToken cancellationToken)
    {
        if (_isLoaded) return;
       await PostJsonAsync<string>(Load, new { collectionName = CollectionName }, cancellationToken);
       
        
                       // await _milvusCollection.LoadAsync(cancellationToken: cancellationToken);
                        //await _milvusCollection.WaitForCollectionLoadAsync(cancellationToken: cancellationToken);
                     
        _isLoaded = true;
        _logger.LogInformation("Milvus collection loaded into memory.");
    }

    public MilvusCollection MilvusCollection => _milvusCollection;
    public ScriptModule ScriptModule => _scriptModule;
}
