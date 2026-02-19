using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace ScannerAdminApp.Helpers
{
    public class BaseViewModel : INotifyPropertyChanged
    {
        private readonly Dictionary<string, object> _values = new();

        protected T Get<T>([CallerMemberName] string name = null)
        {
            if (name == null) throw new ArgumentNullException(nameof(name));
            if (_values.TryGetValue(name, out var val)) return (T)val;
            return default;
        }

        protected bool Set<T>(T value, [CallerMemberName] string name = null)
        {
            if (name == null) throw new ArgumentNullException(nameof(name));
            if (_values.TryGetValue(name, out var existing) && Equals(existing, value)) return false;
            _values[name] = value;
            OnPropertyChanged(name);
            return true;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string name = null) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
