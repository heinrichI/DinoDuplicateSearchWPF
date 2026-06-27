using System.IO;
using System.Text.Json;
using System.Windows;

namespace DinoDuplicateSearch;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        AddCudaPaths();
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
