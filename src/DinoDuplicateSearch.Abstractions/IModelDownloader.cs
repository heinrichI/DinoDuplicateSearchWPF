namespace DinoDuplicateSearch.Abstractions;

public interface IModelDownloader
{
    Task EnsureModelsAsync(IProgress<double>? progress = null, CancellationToken ct = default);
    IReadOnlyList<ModelInfo> GetRequiredModels();
}

public record ModelInfo(string FileName, string Description, string? DownloadUrl, bool RequiresPythonExport);
