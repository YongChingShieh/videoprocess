using System.Net.Http.Json;
using System.Text.Json;

public static class WebApiHelper{
public static HttpClient HttpClient = new(new HttpClientHandler()
{
    ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator,

})
{

    Timeout = Timeout.InfiniteTimeSpan
};
    public static async Task<T> PostJsonAsync<T>(string url, object payload, Dictionary<string, string> headers, CancellationToken cancellationToken = default)
    {
        // 1. 为本次请求创建独立的 HttpRequestMessage
        using var request = new HttpRequestMessage(HttpMethod.Post, url);

        // 2. 设置请求体 (使用 JsonContent 代替 PostAsJsonAsync 以便绑定到 RequestMessage)
        request.Content = JsonContent.Create(payload);

        // 3. 将 Headers 添加到本次请求上，完全避开操作全局 DefaultRequestHeaders
        if (headers != null)
        {
            foreach (var header in headers)
            {
                // 防止由于配置意外读出 null 值导致崩溃
                if (!string.IsNullOrEmpty(header.Key) && !string.IsNullOrEmpty(header.Value))
                {
                    request.Headers.TryAddWithoutValidation(header.Key, header.Value);
                }
            }
        }

        // 4. 发送请求
        var response = await HttpClient.SendAsync(request, cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException($"HTTP 请求失败 [{response.StatusCode}]: {await response.Content.ReadAsStringAsync(cancellationToken)}");
        }

        if (typeof(T) == typeof(string))
        {
            return (T)(object)await response.Content.ReadAsStringAsync(cancellationToken);
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