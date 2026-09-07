using System.IO;
using System.Text.Json;
using System.Windows;
using DinoDuplicateSearch.Abstractions;
using DinoDuplicateSearch.Core;
using DinoDuplicateSearch.CV;
using DinoDuplicateSearch.CV.Extensions;
using DinoDuplicateSearch.Database.Extensions;
using DinoDuplicateSearch.Infrastructure;
using DinoDuplicateSearch.ML;
using DinoDuplicateSearch.ViewModels;
using DinoDuplicateSearch.Views;
using Microsoft.Extensions.DependencyInjection;

namespace DinoDuplicateSearch;

public partial class App : Application
{
    private IServiceProvider? _serviceProvider;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        AddCudaPaths();

        var downloader = new ModelDownloader();
        try
        {
            await downloader.EnsureModelsAsync();
        }
        catch (InvalidOperationException ex)
        {
            MessageBox.Show(ex.Message, "Models Required", MessageBoxButton.OK, MessageBoxImage.Warning);
            Shutdown();
            return;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to download models: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown();
            return;
        }

        var services = new ServiceCollection();
        ConfigureServices(services);
        _serviceProvider = services.BuildServiceProvider();

        var mainWindow = _serviceProvider.GetRequiredService<MainWindow>();
        mainWindow.DataContext = _serviceProvider.GetRequiredService<MainViewModel>();
        mainWindow.Show();
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        services.AddDatabase();
        services.AddComputerVision();

        services.AddSingleton<EmbeddingExtractor>(sp =>
            new EmbeddingExtractor(
                "Models/dinov2-base.onnx",
                sp.GetRequiredService<IFeatureCache>()));

        services.AddSingleton<LightGlueGeometricVerifier>(sp =>
            new LightGlueGeometricVerifier(sp.GetRequiredService<ISuperPointLightGluePipeline>()));

        services.AddSingleton<SiftGeometricVerifier>();

        services.AddSingleton<IGeometricVerifierFactory>(sp =>
            new GeometricVerifierFactory(
                sp.GetRequiredService<LightGlueGeometricVerifier>(),
                sp.GetRequiredService<SiftGeometricVerifier>()));

        services.AddSingleton<DuplicatesFinder>(sp =>
            new DuplicatesFinder(
                sp.GetRequiredService<IFeatureCache>(),
                sp.GetRequiredService<EmbeddingExtractor>(),
                sp.GetRequiredService<IGeometricVerifierFactory>()));

        services.AddTransient<SearchViewModel>(sp =>
            new SearchViewModel(sp.GetRequiredService<DuplicatesFinder>()));

        services.AddSingleton<ResultsViewModel>();
        services.AddSingleton<MainViewModel>(sp =>
            new MainViewModel(sp.GetRequiredService<SearchViewModel>()));

        services.AddSingleton<MainWindow>();
    }

    private static void AddCudaPaths()
    {
        try
        {
            var configFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "appsettings.json");
            if (!File.Exists(configFile)) return;

            var json = File.ReadAllText(configFile);
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var paths = new List<string>();
            if (root.TryGetProperty("cuda_path", out var cuda) && !string.IsNullOrWhiteSpace(cuda.GetString()))
                paths.Add(cuda.GetString()!);
            if (root.TryGetProperty("cudnn_path", out var cudnn) && !string.IsNullOrWhiteSpace(cudnn.GetString()))
                paths.Add(cudnn.GetString()!);

            if (paths.Count > 0)
            {
                var currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                var newPath = string.Join(";", paths) + ";" + currentPath;
                Environment.SetEnvironmentVariable("PATH", newPath);
            }
        }
        catch { }
    }
}
