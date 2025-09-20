using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;


public class MilvusStartupService(ILogger<MilvusStartupService> logger, MilvusImageService milvusService) : IHostedService
{
    private readonly ILogger<MilvusStartupService> _logger = logger;
    private readonly MilvusImageService _milvusService = milvusService;

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting MilvusStartupService...");
        await _milvusService.InitAsync(cancellationToken);
        _logger.LogInformation("MilvusStartupService started.");
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Stopping MilvusStartupService...");
        return Task.CompletedTask;
    }
}
