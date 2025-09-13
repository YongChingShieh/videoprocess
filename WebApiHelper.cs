using System.Net.Http.Json;
using System.Text.Json;

public static class WebApiHelper{
public static HttpClient HttpClient = new(new HttpClientHandler()
{
    ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator,

})
{

    Timeout = Timeout.InfiniteTimeSpan,

};
public static async Task<T> PostJsonAsync<T>(string url, object payload, CancellationToken cancellationToken = default)
{
    var response = await HttpClient.PostAsJsonAsync(url, payload, cancellationToken);
    if (!response.IsSuccessStatusCode)
    {
        return (T)(object)response;
    }
        if (typeof(T) == typeof(string))
        {

            return (T)(object)await response.Content.ReadAsStringAsync(cancellationToken); ;
        }
        else if (typeof(T) == typeof(Stream))
        {
            return (T)(object)await response.Content.ReadAsStreamAsync(cancellationToken);
        }
        else if (typeof(T) == typeof(byte[]))
        {
            return (T)(object)await response.Content.ReadAsByteArrayAsync(cancellationToken);
        }
   
    return (T)(object)response;

}
}