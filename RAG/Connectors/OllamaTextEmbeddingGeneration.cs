using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

namespace Connectors
{
    public class OllamaTextEmbeddingGeneration : ITextEmbeddingGenerationService
    {
        private readonly string _modelId;
        private readonly Uri _endpoint;
        private readonly HttpClient _httpClient;

        public OllamaTextEmbeddingGeneration(string modelId, Uri endpoint, HttpClient? httpClient = null)
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

        public async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
            IList<string> data,
            Kernel? kernel = null,
            CancellationToken cancellationToken = default)
        {
            var embeddings = new List<ReadOnlyMemory<float>>();

            foreach (var text in data)
            {
                var request = new OllamaEmbeddingRequest
                {
                    Model = _modelId,
                    Prompt = text
                };

                using var response = await _httpClient.PostAsJsonAsync("/api/embeddings", request, cancellationToken);
                response.EnsureSuccessStatusCode();

                var result = await response.Content.ReadFromJsonAsync<OllamaEmbeddingResponse>(cancellationToken: cancellationToken);
                if (result?.Embedding == null || result.Embedding.Length == 0)
                    throw new InvalidOperationException($"Ollama returned an empty embedding for model '{_modelId}'.");

                embeddings.Add(new ReadOnlyMemory<float>(result.Embedding));
            }

            return embeddings;
        }

        private sealed class OllamaEmbeddingRequest
        {
            [JsonPropertyName("model")]
            public string Model { get; set; } = string.Empty;

            [JsonPropertyName("prompt")]
            public string Prompt { get; set; } = string.Empty;
        }

        private sealed class OllamaEmbeddingResponse
        {
            [JsonPropertyName("embedding")]
            public float[] Embedding { get; set; } = Array.Empty<float>();
        }
    }

    public static class OllamaTextEmbeddingGenerationExtension
    {
        public static IKernelBuilder AddOllamaTextEmbeddingGeneration(
            this IKernelBuilder builder,
            string modelId,
            Uri endpoint)
        {
            builder.Services.AddSingleton<ITextEmbeddingGenerationService>(
                new OllamaTextEmbeddingGeneration(modelId, endpoint));
            return builder;
        }
    }
}
