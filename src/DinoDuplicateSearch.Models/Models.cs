namespace DinoDuplicateSearch.Models;

using System.Text.Json;
using System.Text.Json.Serialization;

public record ProgressData(double Percent, string Status, string? FilePath = null);

public class DuplicatePair
{
    public string Path1 { get; set; } = "";
    public string Path2 { get; set; } = "";
    public float Similarity { get; set; }
    public bool GeometricVerified { get; set; }
    public float GeometricAngle { get; set; }
    public int GeometricAngleVotes { get; set; }
    public float GeometricScale { get; set; }
    public int GeometricScaleVotes { get; set; }
}

public class DuplicateGroup
{
    public int ClusterId { get; set; }
    public List<DuplicatePair> Pairs { get; set; } = new();

    [JsonIgnore]
    public List<string> Paths
    {
        get
        {
            var all = new HashSet<string>();
            foreach (var pair in Pairs)
            {
                all.Add(pair.Path1);
                all.Add(pair.Path2);
            }
            return all.ToList();
        }
    }

    [JsonIgnore]
    public bool IsGeometricVerified => Pairs.Any(p => p.GeometricVerified);

    [JsonIgnore]
    public float AvgSimilarity
    {
        get
        {
            if (Pairs.Count == 0) return 0;
            return Pairs.Average(p => p.Similarity);
        }
    }
}

public class SearchResult
{
    public string DirectoryPath { get; set; } = "";
    public List<DuplicateGroup> Groups { get; set; } = new();
    public DateTime SavedAt { get; set; } = DateTime.Now;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public static void Save(string filePath, SearchResult result)
    {
        var json = JsonSerializer.Serialize(result, JsonOptions);
        File.WriteAllText(filePath, json);
    }

    public static SearchResult? Load(string filePath)
    {
        if (!File.Exists(filePath)) return null;
        try
        {
            var json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<SearchResult>(json, JsonOptions);
        }
        catch { return null; }
    }
}
