using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Connectors
{
    public static class OllamaChatCompletionExtension
    {
        public static IKernelBuilder AddOllamaChatCompletion(this IKernelBuilder builder, string modelId, Uri endpoint)
        {
            builder.Services.AddSingleton<IChatCompletionService>(new OllamaChatCompletionService(modelId, endpoint));
            return builder;
        }
    }

    public class OllamaChatCompletionService : IChatCompletionService
    {
        private readonly string _modelId;
        private readonly Uri _endpoint;
        private readonly HttpClient _httpClient;

        public OllamaChatCompletionService(string modelId, Uri endpoint, HttpClient? httpClient = null)
        {
            _modelId = modelId;
            _endpoint = endpoint;
            _httpClient = httpClient ?? new HttpClient { BaseAddress = endpoint };
        }

        public IReadOnlyDictionary<string, object?> Attributes => new Dictionary<string, object?>
        {
            { "ModelId", _modelId },
            { "Endpoint", _endpoint.ToString() }
        };

        public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? settings = null,
            Kernel? kernel = null,
            CancellationToken cancellationToken = default)
        {
            var request = BuildRequest(chatHistory, stream: false);
            using var response = await _httpClient.PostAsJsonAsync("/api/chat", request, cancellationToken);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaChatResponse>(cancellationToken: cancellationToken);
            var content = result?.Message?.Content ?? string.Empty;

            return new List<ChatMessageContent>
            {
                new(AuthorRole.Assistant, content)
            };
        }

        public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? settings = null,
            Kernel? kernel = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var request = BuildRequest(chatHistory, stream: true);
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
            {
                Content = JsonContent.Create(request)
            };

            using var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            response.EnsureSuccessStatusCode();

            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            while (!reader.EndOfStream)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var line = await reader.ReadLineAsync(cancellationToken);
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var chunk = JsonSerializer.Deserialize<OllamaChatResponse>(line);
                if (!string.IsNullOrEmpty(chunk?.Message?.Content))
                {
                    yield return new StreamingChatMessageContent(AuthorRole.Assistant, chunk.Message.Content);
                }

                if (chunk?.Done == true)
                    break;
            }
        }

        private OllamaChatRequest BuildRequest(ChatHistory chatHistory, bool stream)
        {
            return new OllamaChatRequest
            {
                Model = _modelId,
                Stream = stream,
                Messages = chatHistory
                    .Select(m => new OllamaChatMessage
                    {
                        Role = MapRole(m.Role),
                        Content = m.Content ?? string.Empty
                    })
                    .ToList()
            };
        }

        private static string MapRole(AuthorRole role)
        {
            if (role == AuthorRole.System)
                return "system";
            if (role == AuthorRole.Assistant)
                return "assistant";

            return "user";
        }

        private sealed class OllamaChatRequest
        {
            [JsonPropertyName("model")]
            public string Model { get; set; } = string.Empty;

            [JsonPropertyName("stream")]
            public bool Stream { get; set; }

            [JsonPropertyName("messages")]
            public List<OllamaChatMessage> Messages { get; set; } = new();
        }

        private sealed class OllamaChatMessage
        {
            [JsonPropertyName("role")]
            public string Role { get; set; } = "user";

            [JsonPropertyName("content")]
            public string Content { get; set; } = string.Empty;
        }

        private sealed class OllamaChatResponse
        {
            [JsonPropertyName("message")]
            public OllamaChatMessage? Message { get; set; }

            [JsonPropertyName("done")]
            public bool Done { get; set; }
        }
    }
}
