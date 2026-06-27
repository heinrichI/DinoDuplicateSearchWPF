namespace DinoDuplicateSearch.Models;

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

    public bool IsGeometricVerified => Pairs.Any(p => p.GeometricVerified);

    public float AvgSimilarity
    {
        get
        {
            if (Pairs.Count == 0) return 0;
            return Pairs.Average(p => p.Similarity);
        }
    }
}
