using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;
using ScannerAdminApp.Helpers;

namespace ScannerAdminApp.ViewModels
{
    public class ScannerConfigViewModel : BaseViewModel
    {
        public string PrimaryScanner { get => Get<string>(); set => Set(value); }

        public ObservableCollection<string> Resolutions { get; } = new()
        {
            "200 DPI (Draft)", "300 DPI", "600 DPI"
        };
        public string SelectedResolution { get => Get<string>(); set => Set(value); }

        public ObservableCollection<string> PaperSizes { get; } = new()
        {
            "A4 (210x297mm)", "Letter (8.5x11in)"
        };
        public string SelectedPaperSize { get => Get<string>(); set => Set(value); }

        public ObservableCollection<string> ColorModes { get; } = new()
        {
            "Black & White", "Grayscale", "Color"
        };
        public string SelectedColorMode { get => Get<string>(); set => Set(value); }

        public ObservableCollection<string> DuplexModes { get; } = new()
        {
            "Single-sided", "Double-sided"
        };
        public string SelectedDuplexMode { get => Get<string>(); set => Set(value); }

        public ICommand TestScannerCommand { get; }
        public ICommand AutoDetectCommand { get; }

        public ScannerConfigViewModel()
        {
            PrimaryScanner = "Canon imageFORMULA DR-S150";
            SelectedResolution = Resolutions[0];
            SelectedPaperSize = PaperSizes[0];
            SelectedColorMode = ColorModes[0];
            SelectedDuplexMode = DuplexModes[0];

            TestScannerCommand = new RelayCommand(_ => TestScanner());
            AutoDetectCommand = new RelayCommand(_ => AutoDetect());
        }

        private void TestScanner()
        {
            MessageBox.Show("Testing scanner...", "Test Scanner");
        }

        private void AutoDetect()
        {
            MessageBox.Show("Auto-detecting scanner...", "Auto Detect");
        }
    }
}
