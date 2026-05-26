using Microsoft.SemanticKernel.Memory;
using Microsoft.Data.Sqlite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace Connectors
{
    public class SqliteMemoryStore : IMemoryStore
    {
        private readonly string _connectionString;

        public SqliteMemoryStore(string dbPath)
        {
            _connectionString = $"Data Source={dbPath}";
            InitializeDatabase();
        }

        private void InitializeDatabase()
        {
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();

            var command = connection.CreateCommand();
            command.CommandText =
            @"
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS Collections (
                    CollectionName TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS MemoryRecords (
                    Id TEXT PRIMARY KEY,
                    CollectionName TEXT NOT NULL,
                    Text TEXT NOT NULL,
                    Embedding BLOB NOT NULL,
                    FOREIGN KEY(CollectionName) REFERENCES Collections(CollectionName) ON DELETE CASCADE
                );
            ";
            command.ExecuteNonQuery();
        }

        public async Task CreateCollectionAsync(string collectionName, CancellationToken cancellationToken = default)
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText = "INSERT OR IGNORE INTO Collections (CollectionName) VALUES ($collectionName)";
            cmd.Parameters.AddWithValue("$collectionName", collectionName);

            await cmd.ExecuteNonQueryAsync(cancellationToken);
        }

        public async IAsyncEnumerable<string> GetCollectionsAsync(
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText = "SELECT CollectionName FROM Collections";

            using var reader = await cmd.ExecuteReaderAsync(cancellationToken);
            while (await reader.ReadAsync(cancellationToken))
            {
                yield return reader.GetString(0);
            }
        }

        public async Task DeleteCollectionAsync(string collectionName, CancellationToken cancellationToken = default)
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText = "DELETE FROM Collections WHERE CollectionName = $collectionName";
            cmd.Parameters.AddWithValue("$collectionName", collectionName);

            await cmd.ExecuteNonQueryAsync(cancellationToken);
        }

        public async Task<bool> DoesCollectionExistAsync(string collectionName, CancellationToken cancellationToken = default)
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText = "SELECT 1 FROM Collections WHERE CollectionName = $collectionName LIMIT 1";
            cmd.Parameters.AddWithValue("$collectionName", collectionName);

            var result = await cmd.ExecuteScalarAsync(cancellationToken);
            return result != null;
        }

        public async Task<string> UpsertAsync(string collectionName, MemoryRecord record, CancellationToken cancellationToken = default)
        {
            await CreateCollectionAsync(collectionName, cancellationToken);

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText =
            @"
                INSERT INTO MemoryRecords (Id, CollectionName, Text, Embedding)
                VALUES ($id, $collectionName, $text, $embedding)
                ON CONFLICT(Id) DO UPDATE SET
                    Text = excluded.Text,
                    Embedding = excluded.Embedding
            ";
            cmd.Parameters.AddWithValue("$id", record.Metadata.Id);
            cmd.Parameters.AddWithValue("$collectionName", collectionName);
            cmd.Parameters.AddWithValue("$text", record.Metadata.Text ?? string.Empty);

            byte[] embeddingBlob = FloatArrayToByteArray(record.Embedding.Span);
            cmd.Parameters.AddWithValue("$embedding", embeddingBlob);

            await cmd.ExecuteNonQueryAsync(cancellationToken);
            return record.Metadata.Id;
        }

        public async IAsyncEnumerable<string> UpsertBatchAsync(
            string collectionName,
            IEnumerable<MemoryRecord> records,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var record in records)
            {
                yield return await UpsertAsync(collectionName, record, cancellationToken);
            }
        }

        public async Task<MemoryRecord?> GetAsync(
            string collectionName,
            string key,
            bool withEmbedding = false,
            CancellationToken cancellationToken = default)
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText =
            @"
                SELECT Text, Embedding
                FROM MemoryRecords
                WHERE CollectionName = $collectionName AND Id = $id
            ";
            cmd.Parameters.AddWithValue("$collectionName", collectionName);
            cmd.Parameters.AddWithValue("$id", key);

            using var reader = await cmd.ExecuteReaderAsync(cancellationToken);

            if (!await reader.ReadAsync(cancellationToken))
                return null;

            return ReadMemoryRecord(key, reader, withEmbedding);
        }

        public async IAsyncEnumerable<MemoryRecord> GetBatchAsync(
            string collectionName,
            IEnumerable<string> keys,
            bool withEmbeddings = false,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var key in keys)
            {
                var record = await GetAsync(collectionName, key, withEmbeddings, cancellationToken);
                if (record != null)
                    yield return record;
            }
        }

        public async Task RemoveAsync(string collectionName, string key, CancellationToken cancellationToken = default)
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText = "DELETE FROM MemoryRecords WHERE CollectionName = $collectionName AND Id = $id";
            cmd.Parameters.AddWithValue("$collectionName", collectionName);
            cmd.Parameters.AddWithValue("$id", key);

            await cmd.ExecuteNonQueryAsync(cancellationToken);
        }

        public async Task RemoveBatchAsync(
            string collectionName,
            IEnumerable<string> keys,
            CancellationToken cancellationToken = default)
        {
            foreach (var key in keys)
            {
                await RemoveAsync(collectionName, key, cancellationToken);
            }
        }

        public async IAsyncEnumerable<(MemoryRecord, double)> GetNearestMatchesAsync(
            string collectionName,
            ReadOnlyMemory<float> embedding,
            int limit,
            double minRelevanceScore = 0,
            bool withEmbeddings = false,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var candidates = await LoadCollectionRecordsAsync(collectionName, cancellationToken);

            var ranked = candidates
                .Select(c => (Record: ToMemoryRecord(c, withEmbeddings), Score: CosineSimilarity(embedding.Span, c.Embedding)))
                .Where(x => x.Score >= minRelevanceScore)
                .OrderByDescending(x => x.Score)
                .Take(limit);

            foreach (var item in ranked)
            {
                yield return item;
            }
        }

        public async Task<(MemoryRecord, double)?> GetNearestMatchAsync(
            string collectionName,
            ReadOnlyMemory<float> embedding,
            double minRelevanceScore = 0,
            bool withEmbeddings = false,
            CancellationToken cancellationToken = default)
        {
            await foreach (var match in GetNearestMatchesAsync(
                collectionName,
                embedding,
                limit: 1,
                minRelevanceScore,
                withEmbeddings,
                cancellationToken))
            {
                return match;
            }

            return null;
        }

        private async Task<List<StoredRecord>> LoadCollectionRecordsAsync(string collectionName, CancellationToken cancellationToken)
        {
            var records = new List<StoredRecord>();

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync(cancellationToken);

            var cmd = connection.CreateCommand();
            cmd.CommandText =
            @"
                SELECT Id, Text, Embedding
                FROM MemoryRecords
                WHERE CollectionName = $collectionName
            ";
            cmd.Parameters.AddWithValue("$collectionName", collectionName);

            using var reader = await cmd.ExecuteReaderAsync(cancellationToken);
            while (await reader.ReadAsync(cancellationToken))
            {
                var id = reader.GetString(0);
                var text = reader.GetString(1);
                var embeddingBlob = (byte[])reader[2];
                records.Add(new StoredRecord(id, text, ByteArrayToFloatArray(embeddingBlob)));
            }

            return records;
        }

        private static MemoryRecord ReadMemoryRecord(string key, SqliteDataReader reader, bool withEmbedding)
        {
            var text = reader.GetString(0);
            ReadOnlyMemory<float> embedding = default;

            if (withEmbedding && !reader.IsDBNull(1))
            {
                var embeddingBlob = (byte[])reader[1];
                embedding = new ReadOnlyMemory<float>(ByteArrayToFloatArray(embeddingBlob));
            }

            return MemoryRecord.LocalRecord(
                key,
                text,
                description: null,
                additionalMetadata: null,
                embedding: embedding);
        }

        private static MemoryRecord ToMemoryRecord(StoredRecord record, bool withEmbedding)
        {
            return MemoryRecord.LocalRecord(
                record.Id,
                record.Text,
                description: null,
                additionalMetadata: null,
                embedding: withEmbedding ? new ReadOnlyMemory<float>(record.Embedding) : default);
        }

        private static double CosineSimilarity(ReadOnlySpan<float> a, float[] b)
        {
            if (a.Length == 0 || b.Length == 0 || a.Length != b.Length)
                return 0;

            double dot = 0;
            double normA = 0;
            double normB = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            if (normA == 0 || normB == 0)
                return 0;

            return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        }

        private static byte[] FloatArrayToByteArray(ReadOnlySpan<float> floatArray)
        {
            var byteArray = new byte[floatArray.Length * sizeof(float)];
            Buffer.BlockCopy(floatArray.ToArray(), 0, byteArray, 0, byteArray.Length);
            return byteArray;
        }

        private static float[] ByteArrayToFloatArray(byte[] byteArray)
        {
            var floatArray = new float[byteArray.Length / sizeof(float)];
            Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
            return floatArray;
        }

        private sealed record StoredRecord(string Id, string Text, float[] Embedding);
    }
}
