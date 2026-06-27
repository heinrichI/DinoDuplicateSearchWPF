using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using DinoDuplicateSearch.Models;

namespace DinoDuplicateSearch.ViewModels;

public class ResultsViewModel : INotifyPropertyChanged
{
    private List<DuplicateGroup> _groups = new();
    public List<DuplicateGroup> Groups
    {
        get => _groups;
        set { _groups = value; OnPropertyChanged(); OnPropertyChanged(nameof(CountText)); }
    }

    public string CountText => $"Found: {Groups.Count} groups";

    public ICommand OpenImageCommand { get; }

    public ResultsViewModel()
    {
        OpenImageCommand = new RelayCommand(OpenImage);
    }

    private void OpenImage(object? path)
    {
        if (path is string filePath && File.Exists(filePath))
        {
            try
            {
                Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true });
            }
            catch { }
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    protected void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
