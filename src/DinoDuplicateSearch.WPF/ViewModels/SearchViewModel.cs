using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Windows;
using System.Windows.Input;
using DinoDuplicateSearch.Core;
using DinoDuplicateSearch.Models;

namespace DinoDuplicateSearch.ViewModels;

public class SearchViewModel : INotifyPropertyChanged
{
    private const string ConfigFile = "config.json";
    private readonly DuplicatesFinder _finder = new();
    private CancellationTokenSource? _cts;

    private string _directoryPath = "";
    public string DirectoryPath
    {
        get => _directoryPath;
        set { _directoryPath = value; OnPropertyChanged(); }
    }

    private double _distanceThreshold = 0.45;
    public double DistanceThreshold
    {
        get => _distanceThreshold;
        set { _distanceThreshold = value; OnPropertyChanged(); OnPropertyChanged(nameof(ThresholdText)); }
    }

    public string ThresholdText => DistanceThreshold.ToString("F2");

    private bool _geometricCheckEnabled = true;
    public bool GeometricCheckEnabled
    {
        get => _geometricCheckEnabled;
        set { _geometricCheckEnabled = value; OnPropertyChanged(); }
    }

    private bool _searchSubfolders;
    public bool SearchSubfolders
    {
        get => _searchSubfolders;
        set { _searchSubfolders = value; OnPropertyChanged(); }
    }

    private double _wgcThreshold = 0.30;
    public double WgcThreshold
    {
        get => _wgcThreshold;
        set { _wgcThreshold = value; OnPropertyChanged(); OnPropertyChanged(nameof(WgcThresholdText)); }
    }

    public string WgcThresholdText => WgcThreshold.ToString("F2");

    private double _minSimilarityForPair = 0.5;
    public double MinSimilarityForPair
    {
        get => _minSimilarityForPair;
        set { _minSimilarityForPair = value; OnPropertyChanged(); OnPropertyChanged(nameof(MinSimilarityText)); }
    }

    public string MinSimilarityText => MinSimilarityForPair.ToString("F2");

    private int _batchSize = 32;
    public int BatchSize
    {
        get => _batchSize;
        set { _batchSize = value; OnPropertyChanged();  }
    }


    private int _prefetchCount = 2;
    public int PrefetchCount
    {
        get => _prefetchCount;
        set { _prefetchCount = value; OnPropertyChanged(); }
    }

    private int _maxClusterSize = 50;
    public int MaxClusterSize
    {
        get => _maxClusterSize;
        set { _maxClusterSize = value; OnPropertyChanged(); OnPropertyChanged(nameof(MaxClusterSizeText)); }
    }

    public string MaxClusterSizeText => MaxClusterSize.ToString();

    private double _transitivityRatio = 0.7;
    public double TransitivityRatio
    {
        get => _transitivityRatio;
        set { _transitivityRatio = value; OnPropertyChanged(); OnPropertyChanged(nameof(TransitivityRatioText)); }
    }

    public string TransitivityRatioText => TransitivityRatio.ToString("F2");

    private double _progressValue;
    public double ProgressValue
    {
        get => _progressValue;
        set { _progressValue = value; OnPropertyChanged(); }
    }

    private string _statusText = "";
    public string StatusText
    {
        get => _statusText;
        set { _statusText = value; OnPropertyChanged(); }
    }

    private string _fileText = "";
    public string FileText
    {
        get => _fileText;
        set { _fileText = value; OnPropertyChanged(); }
    }

    private bool _isSearching;
    public bool IsSearching
    {
        get => _isSearching;
        set { _isSearching = value; OnPropertyChanged(); }
    }

    private bool _isProgressVisible;
    public bool IsProgressVisible
    {
        get => _isProgressVisible;
        set { _isProgressVisible = value; OnPropertyChanged(); }
    }

    private string _findButtonText = "Find Duplicates";
    public string FindButtonText
    {
        get => _findButtonText;
        set { _findButtonText = value; OnPropertyChanged(); }
    }

    public ICommand BrowseCommand { get; }
    public ICommand FindCommand { get; }
    public ICommand CancelCommand { get; }
    public ICommand ClearCacheCommand { get; }

    public event Action<List<DuplicateGroup>>? SearchCompleted;
    public event Action<int>? SwitchToResults;

    public SearchViewModel()
    {
        BrowseCommand = new RelayCommand(_ => Browse());
        FindCommand = new RelayCommand(_ => FindDuplicates(), _ => !IsSearching);
        CancelCommand = new RelayCommand(_ => _cts?.Cancel(), _ => IsSearching);
        ClearCacheCommand = new RelayCommand(_ => ClearCache(), _ => !IsSearching);
        LoadConfig();
    }

    private void Browse()
    {
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Select Folder",
            InitialDirectory = string.IsNullOrEmpty(DirectoryPath) ? Environment.GetFolderPath(Environment.SpecialFolder.MyPictures) : DirectoryPath
        };

        if (dialog.ShowDialog() == true)
            DirectoryPath = dialog.FolderName;
    }

    private async void FindDuplicates()
    {
        if (string.IsNullOrWhiteSpace(DirectoryPath) || !Directory.Exists(DirectoryPath))
        {
            MessageBox.Show("Please select a valid directory", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        SaveConfig();
        IsSearching = true;
        IsProgressVisible = true;
        FindButtonText = "Searching...";
        ProgressValue = 0;
        StatusText = "Initializing...";

        _cts = new CancellationTokenSource();
        var ct = _cts.Token;

        var progress = new ThrottledProgress<ProgressData>(data =>
        {
            if (data.Percent >= 0)
                ProgressValue = data.Percent;
            StatusText = data.Status;
            FileText = data.FilePath ?? "";
        }, TimeSpan.FromMilliseconds(250));

        try
        {
            var settings = new SearchSettings
            {
                DirectoryPath = DirectoryPath,
                DistanceThreshold = (float)DistanceThreshold,
                GeometricCheckEnabled = GeometricCheckEnabled,
                WgcThreshold = (float)WgcThreshold,
                MinSimilarityForPair = (float)MinSimilarityForPair,
                SearchSubfolders = SearchSubfolders,
                BatchSize = BatchSize,
                PrefetchCount = PrefetchCount,
                MaxClusterSize = MaxClusterSize,
                TransitivityRatio = (float)TransitivityRatio
            };

            var results = await Task.Run(() =>
            {
                return _finder.FindDuplicates(settings, progress, ct);
            }, ct);

            SearchCompleted?.Invoke(results);
            SwitchToResults?.Invoke(1);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsSearching = false;
            IsProgressVisible = false;
            FindButtonText = "Find Duplicates";
        }
    }

    private async void ClearCache()
    {
        IsSearching = true;
        IsProgressVisible = true;
        FindButtonText = "Clearing cache...";
        StatusText = "Deleting cache entries...";

        try
        {
            var freed = await Task.Run(() => _finder.ClearCache());
            var mb = freed / (1024.0 * 1024);
            MessageBox.Show($"Cache cleared successfully.\n{mb:F1} MB freed.", "Cache Cleared", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to clear cache: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsSearching = false;
            IsProgressVisible = false;
            FindButtonText = "Find Duplicates";
        }
    }

    private void LoadConfig()
    {
        try
        {
            if (File.Exists(ConfigFile))
            {
                var json = File.ReadAllText(ConfigFile);
                var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("last_directory", out var dir))
                    DirectoryPath = dir.GetString() ?? "";
                if (doc.RootElement.TryGetProperty("search_subfolders", out var sub))
                    SearchSubfolders = sub.GetBoolean();
                if (doc.RootElement.TryGetProperty("batch_size", out var bs))
                    BatchSize = bs.GetInt32();
                if (doc.RootElement.TryGetProperty("prefetch_count", out var pc))
                    PrefetchCount = pc.GetInt32();
                if (doc.RootElement.TryGetProperty("max_cluster_size", out var mc))
                    MaxClusterSize = mc.GetInt32();
                if (doc.RootElement.TryGetProperty("transitivity_ratio", out var tr))
                    TransitivityRatio = tr.GetDouble();
            }
        }
        catch { }
    }

    private void SaveConfig()
    {
        try
        {
            var json = JsonSerializer.Serialize(new
            {
                last_directory = DirectoryPath,
                search_subfolders = SearchSubfolders,
                batch_size = BatchSize,
                prefetch_count = PrefetchCount,
                max_cluster_size = MaxClusterSize,
                transitivity_ratio = TransitivityRatio
            });
            File.WriteAllText(ConfigFile, json);
        }
        catch { }
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    protected void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
