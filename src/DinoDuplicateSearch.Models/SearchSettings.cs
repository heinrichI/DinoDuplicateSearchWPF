namespace DinoDuplicateSearch.Models;

public class SearchSettings
{
    public string DirectoryPath { get; set; } = "";
    public float DistanceThreshold { get; set; } = 0.45f;
    public bool GeometricCheckEnabled { get; set; } = true;
    public float WgcThreshold { get; set; } = 0.3f;
    public float MinSimilarityForPair { get; set; } = 0.5f;
    public bool SearchSubfolders { get; set; }
    public int BatchSize { get; set; } = 32;
    public int PrefetchCount { get; set; } = 2;
    public int MaxClusterSize { get; set; } = 50;
    public float TransitivityRatio { get; set; } = 0.7f;
}
