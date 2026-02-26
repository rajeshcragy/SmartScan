using System;
using System.Windows.Input;
using ScannerAdminApp.Helpers;
using ScannerAdminApp.Services;

namespace ScannerAdminApp.ViewModels
{
    public class RagSettingsViewModel : BaseViewModel
    {
        private readonly RagService _ragService = new RagService();

        // ── LLM configuration ────────────────────────────────────────────────
        public string OllamaEndpoint   { get => Get<string>(); set => Set(value); }
        public string LlmModel         { get => Get<string>(); set => Set(value); }
        public string EmbeddingModel   { get => Get<string>(); set => Set(value); }

        /// <summary>CUDA device index passed via CUDA_VISIBLE_DEVICES to Ollama.</summary>
        public string GpuDevice        { get => Get<string>(); set => Set(value); }

        // ── Document indexing ────────────────────────────────────────────────
        public string DocumentsFolder  { get => Get<string>(); set => Set(value); }
        public int    IndexedChunkCount { get => Get<int>();    set => Set(value); }

        // ── Query / response ─────────────────────────────────────────────────
        public string QueryText        { get => Get<string>(); set { Set(value); OnPropertyChanged(nameof(CanSendQuery)); } }
        public string ResponseText     { get => Get<string>(); set => Set(value); }

        // ── Status ───────────────────────────────────────────────────────────
        public string StatusMessage    { get => Get<string>(); set => Set(value); }

        public bool IsBusy
        {
            get => Get<bool>();
            set
            {
                if (Set(value))
                    OnPropertyChanged(nameof(IsNotBusy));
            }
        }
        public bool IsNotBusy  => !IsBusy;
        public bool CanSendQuery => IsNotBusy && !string.IsNullOrWhiteSpace(QueryText);

        // ── Commands ─────────────────────────────────────────────────────────
        public ICommand TestConnectionCommand  { get; }
        public ICommand IndexDocumentsCommand  { get; }
        public ICommand SendQueryCommand       { get; }
        public ICommand ClearIndexCommand      { get; }

        public RagSettingsViewModel()
        {
            OllamaEndpoint  = "http://localhost:11434";
            LlmModel        = "llama3.2";
            EmbeddingModel  = "nomic-embed-text";
            GpuDevice       = "0";
            DocumentsFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            StatusMessage   = "Ready. Ensure Ollama is running with GPU support enabled.";

            TestConnectionCommand = new RelayCommand(_ => TestConnection(),   _ => IsNotBusy);
            IndexDocumentsCommand = new RelayCommand(_ => IndexDocuments(),   _ => IsNotBusy);
            SendQueryCommand      = new RelayCommand(_ => SendQuery(),        _ => CanSendQuery);
            ClearIndexCommand     = new RelayCommand(_ => ClearIndex(),       _ => IsNotBusy && IndexedChunkCount > 0);
        }

        // ── Command implementations ───────────────────────────────────────────

        private async void TestConnection()
        {
            IsBusy = true;
            StatusMessage = "Testing connection to Ollama…";
            ConfigureService();
            try
            {
                bool ok = await _ragService.TestConnectionAsync().ConfigureAwait(false);
                StatusMessage = ok
                    ? "✓ Connected to Ollama successfully."
                    : "✗ Cannot reach Ollama. Start it with: ollama serve";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Connection error: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }

        private async void IndexDocuments()
        {
            if (string.IsNullOrWhiteSpace(DocumentsFolder))
            {
                StatusMessage = "Please specify a documents folder.";
                return;
            }

            IsBusy = true;
            IndexedChunkCount = 0;
            ConfigureService();
            var progress = new Progress<string>(msg => StatusMessage = msg);
            try
            {
                int count = await _ragService.IndexDocumentsAsync(DocumentsFolder, EmbeddingModel, progress)
                                             .ConfigureAwait(false);
                IndexedChunkCount = count;
                StatusMessage = $"✓ Indexed {count} text chunks from documents in '{DocumentsFolder}'.";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Indexing error: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
                OnPropertyChanged(nameof(CanSendQuery));
            }
        }

        private async void SendQuery()
        {
            if (string.IsNullOrWhiteSpace(QueryText)) return;
            IsBusy = true;
            ResponseText  = string.Empty;
            StatusMessage = "Querying… this may take a moment.";
            ConfigureService();
            try
            {
                string answer = await _ragService.QueryAsync(QueryText, LlmModel, EmbeddingModel)
                                                 .ConfigureAwait(false);
                ResponseText  = answer;
                StatusMessage = "✓ Query complete.";
            }
            catch (Exception ex)
            {
                ResponseText  = $"Error: {ex.Message}";
                StatusMessage = "Query failed.";
            }
            finally
            {
                IsBusy = false;
            }
        }

        private void ClearIndex()
        {
            _ragService.ClearIndex();
            IndexedChunkCount = 0;
            ResponseText      = string.Empty;
            StatusMessage     = "Index cleared.";
            OnPropertyChanged(nameof(CanSendQuery));
        }

        private void ConfigureService() => _ragService.Configure(OllamaEndpoint, GpuDevice);
    }
}
