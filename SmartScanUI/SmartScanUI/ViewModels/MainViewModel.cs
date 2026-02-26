using ScannerAdminApp.Helpers;

namespace ScannerAdminApp.ViewModels
{
    public class MainViewModel : BaseViewModel
    {
        public ScannerConfigViewModel ScannerConfig { get; }
        public AdvancedSettingsViewModel AdvancedSettings { get; }
        public RagSettingsViewModel RagSettings { get; }
        public SidebarViewModel Sidebar { get; }

        public MainViewModel()
        {
            ScannerConfig    = new ScannerConfigViewModel();
            AdvancedSettings = new AdvancedSettingsViewModel();
            RagSettings      = new RagSettingsViewModel();
            Sidebar          = new SidebarViewModel();
        }
    }
}
