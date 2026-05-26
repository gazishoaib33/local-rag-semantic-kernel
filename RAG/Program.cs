using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Embeddings;
using System.Text;
using Connectors;
using RAG;

class Program
{
    static async Task Main(string[] args)
    {
        var mode = args.Length > 0 ? args[0].ToLowerInvariant() : "chat";

        IKernelBuilder kb = Kernel.CreateBuilder();

#pragma warning disable SKEXP0070
        kb.AddOllamaChatCompletion(AppConfig.ChatModel, AppConfig.OllamaBaseUrl);
        kb.AddOllamaTextEmbeddingGeneration(AppConfig.EmbeddingModel, AppConfig.OllamaBaseUrl);
#pragma warning restore SKEXP0070

        var kernel = kb.Build();
        var store = new SqliteMemoryStore(AppConfig.DbPath);
        var embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

#pragma warning disable SKEXP0020
        ISemanticTextMemory memory = new MemoryBuilder()
            .WithMemoryStore(store)
            .WithTextEmbeddingGeneration(embeddingService)
            .Build();
#pragma warning restore SKEXP0020

        if (mode is "--ingest" or "ingest")
        {
            await IngestAsync(store, memory);
            return;
        }

        if (mode is "--chat" or "chat")
        {
            await ChatAsync(kernel, store, memory);
            return;
        }

        PrintUsage();
    }

    static void PrintUsage()
    {
        Console.WriteLine("Local RAG Assistant");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project RAG -- ingest   Load and index bio.txt");
        Console.WriteLine("  dotnet run --project RAG -- chat     Ask questions (default)");
    }

    static async Task IngestAsync(SqliteMemoryStore store, ISemanticTextMemory memory)
    {
        var bioPath = Path.Combine(AppContext.BaseDirectory, "Data", "bio.txt");
        if (!File.Exists(bioPath))
            bioPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "bio.txt");

        if (!File.Exists(bioPath))
            throw new FileNotFoundException($"Could not find bio.txt at {bioPath}");

        var text = await File.ReadAllTextAsync(bioPath);

        if (await store.DoesCollectionExistAsync(AppConfig.CollectionName))
        {
            Console.WriteLine($"Collection '{AppConfig.CollectionName}' already exists. Delete {AppConfig.DbPath} to re-ingest.");
            return;
        }

        var paragraphs = ChunkText(text, maxChunkLength: 300);

        Console.WriteLine($"Ingesting {paragraphs.Count} chunks into '{AppConfig.CollectionName}'...");

        foreach (var paragraph in paragraphs)
        {
            await memory.SaveInformationAsync(AppConfig.CollectionName, paragraph, Guid.NewGuid().ToString());
        }

        Console.WriteLine("Ingest complete.");
    }

    static async Task ChatAsync(Kernel kernel, SqliteMemoryStore store, ISemanticTextMemory memory)
    {
        if (!await store.DoesCollectionExistAsync(AppConfig.CollectionName))
        {
            Console.WriteLine($"Collection '{AppConfig.CollectionName}' not found. Run: dotnet run -- ingest");
            return;
        }

        var ai = kernel.GetRequiredService<IChatCompletionService>();
        var systemPrompt = "You are an AI assistant that helps people find information. Reply short and concise. Use the provided context when it is relevant. If the context does not contain the answer, respond with 'I don't know about this topic.'";
        var conversation = new List<(AuthorRole Role, string Content)>();

        Console.WriteLine("RAG chat ready. Type 'exit' to quit.");

        while (true)
        {
            Console.Write("\nQuestion: ");
            var question = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(question))
                continue;

            if (question.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            var contextBuilder = new StringBuilder();
            var matchIndex = 0;

            await foreach (var result in memory.SearchAsync(AppConfig.CollectionName, question, limit: AppConfig.RetrievalLimit))
            {
                matchIndex++;
                var preview = result.Metadata.Text?.Length > 120
                    ? result.Metadata.Text[..120] + "..."
                    : result.Metadata.Text;

                Console.WriteLine($"  [{matchIndex}] score={result.Relevance:F3} | {preview}");
                contextBuilder.AppendLine(result.Metadata.Text);
            }

            if (matchIndex == 0)
                Console.WriteLine("  (no matches found)");

            var userMessage = matchIndex > 0
                ? $"Context:\n{contextBuilder}\n\nQuestion: {question}"
                : question;

            var chatHistory = BuildChatHistory(systemPrompt, conversation, userMessage);

            Console.Write("\nAnswer: ");
            var responseBuilder = new StringBuilder();

            await foreach (var response in ai.GetStreamingChatMessageContentsAsync(chatHistory))
            {
                Console.Write(response);
                responseBuilder.Append(response);
            }

            Console.WriteLine();

            conversation.Add((AuthorRole.User, userMessage));
            conversation.Add((AuthorRole.Assistant, responseBuilder.ToString()));

            while (conversation.Count > AppConfig.MaxHistoryMessages)
                conversation.RemoveAt(0);
        }
    }

    static ChatHistory BuildChatHistory(
        string systemPrompt,
        List<(AuthorRole Role, string Content)> conversation,
        string currentUserMessage)
    {
        var chatHistory = new ChatHistory(systemPrompt);

        foreach (var (role, content) in conversation)
            chatHistory.Add(new ChatMessageContent(role, content));

        chatHistory.AddUserMessage(currentUserMessage);
        return chatHistory;
    }

    static List<string> ChunkText(string text, int maxChunkLength)
    {
        var chunks = new List<string>();
        var sentences = text.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var current = new StringBuilder();

        foreach (var sentence in sentences)
        {
            var piece = sentence + ".";
            if (current.Length + piece.Length > maxChunkLength && current.Length > 0)
            {
                chunks.Add(current.ToString().Trim());
                current.Clear();
            }

            current.Append(piece).Append(' ');
        }

        if (current.Length > 0)
            chunks.Add(current.ToString().Trim());

        return chunks;
    }
}
