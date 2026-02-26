using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace ScannerAdminApp.Services
{
    /// <summary>
    /// Provides Retrieval-Augmented Generation (RAG) functionality using a local
    /// Ollama instance. GPU acceleration is handled by Ollama automatically when
    /// CUDA drivers are present; set <see cref="GpuDevice"/> and call
    /// <see cref="Configure"/> before using the service to specify the device.
    /// </summary>
    public class RagService
    {
        private static readonly HttpClient _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(5)
        };

        private readonly List<(float[] Embedding, string Text, string Source)> _index = new();
        private string _baseUrl = "http://localhost:11434";

        public int IndexedChunkCount => _index.Count;

        /// <summary>Configures the Ollama endpoint.</summary>
        /// <param name="ollamaEndpoint">Base URL of the running Ollama server.</param>
        /// <param name="gpuDevice">
        /// CUDA device index (informational). To use a specific GPU, launch Ollama with
        /// <c>CUDA_VISIBLE_DEVICES=&lt;device&gt; ollama serve</c> before starting this application.
        /// </param>
        public void Configure(string ollamaEndpoint, string gpuDevice = null)
        {
            _baseUrl = (ollamaEndpoint ?? "http://localhost:11434").TrimEnd('/');
        }

        /// <summary>Removes all indexed document chunks from memory.</summary>
        public void ClearIndex() => _index.Clear();

        /// <summary>Returns true when the Ollama server is reachable.</summary>
        public async Task<bool> TestConnectionAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags").ConfigureAwait(false);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Indexes all <c>.txt</c>, <c>.md</c>, and <c>.csv</c> files inside
        /// <paramref name="folder"/>, splitting them into overlapping chunks and
        /// storing their embeddings in memory.
        /// </summary>
        /// <returns>Total number of indexed text chunks.</returns>
        public async Task<int> IndexDocumentsAsync(
            string folder,
            string embeddingModel,
            IProgress<string> progress = null)
        {
            if (!Directory.Exists(folder))
                throw new DirectoryNotFoundException($"Folder not found: {folder}");

            _index.Clear();

            var supportedExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                ".txt", ".md", ".csv"
            };

            var files = Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)
                .Where(f => supportedExtensions.Contains(Path.GetExtension(f)))
                .ToList();

            foreach (var file in files)
            {
                progress?.Report($"Indexing {Path.GetFileName(file)}…");
                var text = File.ReadAllText(file, Encoding.UTF8);
                var chunks = ChunkText(text);

                foreach (var chunk in chunks)
                {
                    if (string.IsNullOrWhiteSpace(chunk)) continue;
                    var embedding = await GetEmbeddingAsync(chunk, embeddingModel).ConfigureAwait(false);
                    _index.Add((embedding, chunk, Path.GetFileName(file)));
                }
            }

            return _index.Count;
        }

        /// <summary>
        /// Retrieves the most relevant document chunks for <paramref name="question"/>
        /// and asks the LLM to answer using that context.
        /// </summary>
        public async Task<string> QueryAsync(
            string question,
            string llmModel,
            string embeddingModel,
            int topK = 3)
        {
            if (_index.Count == 0)
                return "No documents have been indexed yet. Please index your documents first.";

            var questionEmbedding = await GetEmbeddingAsync(question, embeddingModel).ConfigureAwait(false);

            var context = _index
                .Select(item => (item.Text, item.Source, Score: CosineSimilarity(questionEmbedding, item.Embedding)))
                .OrderByDescending(x => x.Score)
                .Take(topK)
                .Aggregate(new StringBuilder(), (sb, x) =>
                {
                    sb.AppendLine($"[Source: {x.Source}]");
                    sb.AppendLine(x.Text);
                    sb.AppendLine();
                    return sb;
                });

            var prompt = $"Use only the following context from scanned documents to answer the question.\n\n" +
                         $"Context:\n{context}\nQuestion: {question}\n\nAnswer:";

            return await GenerateAsync(llmModel, prompt).ConfigureAwait(false);
        }

        // ── Internal helpers ────────────────────────────────────────────────

        private async Task<float[]> GetEmbeddingAsync(string text, string model)
        {
            var json = $"{{\"model\":{JsonString(model)},\"prompt\":{JsonString(text)}}}";
            var response = await PostJsonAsync($"{_baseUrl}/api/embeddings", json).ConfigureAwait(false);
            return ExtractFloatArray(response, "embedding");
        }

        private async Task<string> GenerateAsync(string model, string prompt)
        {
            var json = $"{{\"model\":{JsonString(model)},\"prompt\":{JsonString(prompt)},\"stream\":false}}";
            var response = await PostJsonAsync($"{_baseUrl}/api/generate", json).ConfigureAwait(false);
            return ExtractJsonString(response, "response") ?? "No response received.";
        }

        private static async Task<string> PostJsonAsync(string url, string json)
        {
            using var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync(url, content).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        }

        // ── Text processing ─────────────────────────────────────────────────

        private static List<string> ChunkText(string text, int chunkWords = 200, int overlapWords = 20)
        {
            var words = text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            var chunks = new List<string>();
            for (int i = 0; i < words.Length; i += chunkWords - overlapWords)
            {
                int len = Math.Min(chunkWords, words.Length - i);
                chunks.Add(string.Join(" ", words, i, len));
            }
            return chunks;
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length == 0 || b.Length == 0) return 0f;
            float dot = 0f, normA = 0f, normB = 0f;
            int len = Math.Min(a.Length, b.Length);
            for (int i = 0; i < len; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return (float)(dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-10));
        }

        // ── Minimal JSON helpers (no external dependencies) ─────────────────

        private static string JsonString(string s)
        {
            if (s == null) return "null";
            var sb = new StringBuilder("\"");
            foreach (var c in s)
            {
                switch (c)
                {
                    case '"':  sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n");  break;
                    case '\r': sb.Append("\\r");  break;
                    case '\t': sb.Append("\\t");  break;
                    default:   sb.Append(c);      break;
                }
            }
            return sb.Append('"').ToString();
        }

        private static string ExtractJsonString(string json, string field)
        {
            var key = $"\"{field}\":\"";
            var idx = json.IndexOf(key, StringComparison.Ordinal);
            if (idx < 0) return null;
            idx += key.Length;
            var sb = new StringBuilder();
            while (idx < json.Length)
            {
                var c = json[idx++];
                if (c == '"') break;
                if (c == '\\' && idx < json.Length)
                {
                    var next = json[idx++];
                    switch (next)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 'r': sb.Append('\r'); break;
                        case 't': sb.Append('\t'); break;
                        default:  sb.Append(next); break;
                    }
                }
                else sb.Append(c);
            }
            return sb.ToString();
        }

        private static float[] ExtractFloatArray(string json, string field)
        {
            var key = $"\"{field}\":[";
            var idx = json.IndexOf(key, StringComparison.Ordinal);
            if (idx < 0) return Array.Empty<float>();
            idx += key.Length;
            var end = json.IndexOf(']', idx);
            if (end < 0) return Array.Empty<float>();
            var parts = json.Substring(idx, end - idx)
                            .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var result = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                result[i] = float.Parse(parts[i].Trim(), CultureInfo.InvariantCulture);
            return result;
        }
    }
}
