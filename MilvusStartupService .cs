using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;

 
public class MilvusStartupService : IHostedService
 
{
    private readonly ILogger<MilvusStartupService> _logger;
    private readonly MilvusImageService _milvusService;

    public MilvusStartupService(ILogger<MilvusStartupService> logger, MilvusImageService milvusService)
    {
        _logger = logger;
        _milvusService = milvusService;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Application starting: loading model and Milvus collection...");
        await _milvusService.EnsureCollectionLoadedAsync(cancellationToken);
        _logger.LogInformation("Model and Milvus collection loaded.");
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
