using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace DinoDuplicateSearch.ViewModels;

public class MainViewModel : INotifyPropertyChanged
{
    private int _selectedTabIndex;
    public int SelectedTabIndex
    {
        get => _selectedTabIndex;
        set { _selectedTabIndex = value; OnPropertyChanged(); }
    }

    public SearchViewModel Search { get; } = new();
    public ResultsViewModel Results { get; } = new();
    public ICommand OpenImageCommand { get; }
    public ICommand SwitchTabCommand { get; }

    public MainViewModel()
    {
        OpenImageCommand = new RelayCommand(OpenImage);
        SwitchTabCommand = new RelayCommand(SwitchTab);
    }

    private void OpenImage(object? path)
    {
        if (path is string filePath && File.Exists(filePath))
        {
            try { Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true }); }
            catch { }
        }
    }

    private void SwitchTab(object? parameter)
    {
        if (parameter is string s && int.TryParse(s, out int idx))
            SelectedTabIndex = idx;
        else if (parameter is int i)
            SelectedTabIndex = i;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

public class RelayCommand : ICommand
{
    private readonly Action<object?> _execute;
    private readonly Func<object?, bool>? _canExecute;

    public RelayCommand(Action<object?> execute, Func<object?, bool>? canExecute = null)
    {
        _execute = execute;
        _canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged
    {
        add => CommandManager.RequerySuggested += value;
        remove => CommandManager.RequerySuggested -= value;
    }

    public bool CanExecute(object? parameter) => _canExecute?.Invoke(parameter) ?? true;
    public void Execute(object? parameter) => _execute(parameter);
}
