using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
 
using static WebApiHelper;
 
using Microsoft.AspNetCore.Mvc.Routing;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Text;
using System.Runtime.Intrinsics.X86;

public class VideoProcessService(IConfiguration configuration) : IHostedService
{
   
    private readonly IConfiguration _configuration = configuration;
public string SystemPrompt { get; set; }

 public (string Token,string Chat,int MaxTokens, JsonNode Request,Dictionary<string,string> Headers) OpenApi { get; set; }
    public (string path,string model) FasterWhisper  { get; set; }
  
    public string FFmpeg { get; private set; }
   
 


 
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        SystemPrompt = await File.ReadAllTextAsync(Path.Combine(Directory.GetCurrentDirectory(), "SystemPrompt.txt"), Encoding.UTF8, cancellationToken);
        FFmpeg = _configuration["AppConfiguration:FFmpeg"];
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
        OpenApi = (OpenApiConf["Token"], OpenApiConf["Chat"], Convert.ToInt32(OpenApiConf["MaxTokens"]), request,
            OpenApiConf.GetSection("Headers").GetChildren().ToDictionary(x => x.Key, x => x.Value)
            );
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }
}
