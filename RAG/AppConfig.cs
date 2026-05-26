namespace RAG;

public static class AppConfig
{
    public static readonly Uri OllamaBaseUrl = new("http://127.0.0.1:11434/");
    public const string ChatModel = "llama3";
    public const string EmbeddingModel = "nomic-embed-text";
    public const string DbPath = "Ragdata.db";
    public const string CollectionName = "DDB1";
    public const int RetrievalLimit = 3;
    public const int MaxHistoryMessages = 6;
}
