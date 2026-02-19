namespace ScannerAdminApp.Models
{
    public class ScannerModel
    {
        public string PrimaryScanner { get; set; }
        public string SelectedResolution { get; set; }
        public string SelectedPaperSize { get; set; }
        public string SelectedColorMode { get; set; }
        public string SelectedDuplexMode { get; set; }
        public bool AutoFeedDetection { get; set; }
        public bool BlankPageRemoval { get; set; }
    }
}
