using System.Diagnostics;
using DinoDuplicateSearch.Abstractions;

namespace DinoDuplicateSearch.Infrastructure;

public class ModelDownloader : IModelDownloader
{
    private readonly string _modelsDir;
    private readonly HttpClient _http;

    private static readonly ModelInfo[] Models = new[]
    {
        new ModelInfo(
            "superpoint_lightglue_pipeline.onnx",
            "SuperPoint+LightGlue pipeline (fused feature extraction + matching)",
            "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v2.0/superpoint_lightglue_pipeline.onnx",
            RequiresPythonExport: false),
        new ModelInfo(
            "dinov2-base.onnx",
            "DINOv2 embedding model (768-D vectors)",
            null,
            RequiresPythonExport: true)
    };

    public ModelDownloader(string? modelsDir = null)
    {
        _modelsDir = modelsDir ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
        _http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
    }

    public IReadOnlyList<ModelInfo> GetRequiredModels() => Models;

    public async Task EnsureModelsAsync(IProgress<double>? progress = null, CancellationToken ct = default)
    {
        Directory.CreateDirectory(_modelsDir);

        var downloadable = Models.Where(m => !File.Exists(Path.Combine(_modelsDir, m.FileName))).ToList();
        if (downloadable.Count == 0) return;

        int total = downloadable.Count;
        for (int i = 0; i < total; i++)
        {
            var model = downloadable[i];
            var path = Path.Combine(_modelsDir, model.FileName);

            if (model.RequiresPythonExport)
            {
                throw new InvalidOperationException(
                    $"Model '{model.FileName}' not found. " +
                    $"Run 'python export_model.py' to generate it, or place it manually in {_modelsDir}");
            }

            Debug.WriteLine($"[MODEL] Downloading {model.FileName}...");
            progress?.Report((double)i / total * 100);

            using var response = await _http.GetAsync(model.DownloadUrl, ct);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? 0;
            using var stream = await response.Content.ReadAsStreamAsync(ct);
            using var fileStream = File.Create(path);

            var buffer = new byte[81920];
            long downloaded = 0;
            int read;
            while ((read = await stream.ReadAsync(buffer, ct)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                downloaded += read;
                if (totalBytes > 0)
                    progress?.Report(((double)i / total + (double)downloaded / totalBytes / total) * 100);
            }

            Debug.WriteLine($"[MODEL] Downloaded {model.FileName} ({downloaded / 1024 / 1024} MB)");
        }

        progress?.Report(100);
    }
}
