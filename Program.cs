var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers(options =>
{
    options.SuppressAsyncSuffixInActionNames = false;
});

builder.Services.AddSingleton<MilvusImageService>();
builder.Services.AddHostedService<MilvusStartupService>();
builder.WebHost.ConfigureKestrel(options =>
{
    var Kestrel = builder.Configuration.GetSection("Kestrel");
    options.Configure(Kestrel);
});
 
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
//builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

 

app.UseAuthorization();

app.MapControllers();

await app.RunAsync();
