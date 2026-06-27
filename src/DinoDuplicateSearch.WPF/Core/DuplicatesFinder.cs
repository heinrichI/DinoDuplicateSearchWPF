using System.IO;
using DinoDuplicateSearch.Models;
using DinoDuplicateSearch.Database;
using DinoDuplicateSearch.CV;
using DinoDuplicateSearch.ML;

namespace DinoDuplicateSearch.Core;

public class DuplicatesFinder
{
    private readonly EmbeddingExtractor _embeddingExtractor;
    private readonly FeatureCache _cache;
    private float _wgcThreshold = 0.3f;
    private float _minSimilarityForUnion = 0.5f;

    private static readonly string[] SupportedExtensions = { ".jpg", ".jpeg", ".png", ".webp" };

    public DuplicatesFinder(string modelPath = "Models/dinov2-base.onnx")
    {
        _cache = new FeatureCache();
        _embeddingExtractor = new EmbeddingExtractor(modelPath, _cache);
    }

    public static List<string> ListImages(string folder, bool searchSubfolders = false)
    {
        var option = searchSubfolders ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return Directory.GetFiles(folder, "*.*", option)
            .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .ToList();
    }

    public List<DuplicateGroup> FindDuplicates(
        string folderPath,
        float distanceThreshold = 0.45f,
        bool enableGeometricCheck = false,
        float wgcThreshold = 0.3f,
        float minSimilarityForUnion = 0.5f,
        bool searchSubfolders = false,
        IProgress<ProgressData>? progress = null,
        CancellationToken ct = default)
    {
        _wgcThreshold = wgcThreshold;
        _minSimilarityForUnion = minSimilarityForUnion;
        _embeddingExtractor.SetProgress(progress);

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(0, "Scanning folder for images..."));
        var paths = ListImages(folderPath, searchSubfolders);
        if (paths.Count == 0)
        {
            progress?.Report(new ProgressData(100, "No images found"));
            return new List<DuplicateGroup>();
        }
        progress?.Report(new ProgressData(2, $"Found {paths.Count} images. Loading model..."));

        progress?.Report(new ProgressData(5, "Computing embeddings..."));
        var embs = _embeddingExtractor.EmbedImagesBatch(paths);

        var embs2D = new float[paths.Count, 768];
        for (int i = 0; i < paths.Count; i++)
            for (int d = 0; d < 768; d++)
                embs2D[i, d] = embs[i][d];

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(40, "Clustering images..."));
        var labels = AgglomerativeClustering.FitPredict(embs2D, distanceThreshold);

        var clusters = new Dictionary<int, List<string>>();
        for (int i = 0; i < paths.Count; i++)
        {
            if (!clusters.ContainsKey(labels[i]))
                clusters[labels[i]] = new List<string>();
            clusters[labels[i]].Add(paths[i]);
        }

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(45, "Verifying geometric consistency..."));
        int pairCount = 0;
        int totalPairs = clusters.Values.Sum(items => items.Count * (items.Count - 1) / 2);

        var clusterPairs = new Dictionary<int, List<(DuplicatePair pair, string p1, string p2)>>();
        var clusterWgcGraph = new Dictionary<int, Dictionary<string, HashSet<string>>>();

        foreach (var clusterKvp in clusters)
        {
            var clusterId = clusterKvp.Key;
            var items = clusterKvp.Value;
            if (items.Count <= 1) continue;

            var pairs = new List<(DuplicatePair pair, string p1, string p2)>();
            var wgcGraph = new Dictionary<string, HashSet<string>>();
            foreach (var item in items) wgcGraph[item] = new HashSet<string>();

            for (int i = 0; i < items.Count; i++)
            {
                for (int j = i + 1; j < items.Count; j++)
                {
                    var idx1 = paths.IndexOf(items[i]);
                    var idx2 = paths.IndexOf(items[j]);
                    var sim = DotProduct(embs[idx1], embs[idx2]);

                    var pair = new DuplicatePair
                    {
                        Path1 = items[i],
                        Path2 = items[j],
                        Similarity = sim
                    };

                    if (enableGeometricCheck)
                    {
                        var (ok, angle, scale, angleVotes, scaleVotes) = VerifyGeometric(items[i], items[j]);
                        pair.GeometricVerified = ok;
                        pair.GeometricAngle = angle;
                        pair.GeometricAngleVotes = angleVotes;
                        pair.GeometricScale = scale;
                        pair.GeometricScaleVotes = scaleVotes;

                        var fn1 = Path.GetFileName(items[i]);
                        var fn2 = Path.GetFileName(items[j]);
                        DebugLog.Write($"[WGC] cluster={clusterId} {fn1} <-> {fn2} : {(ok ? "PASS" : "FAIL")} sim={sim:F4} angle={angle:F1}({angleVotes}) scale={scale:F2}({scaleVotes})");

                        if (ok && sim >= _minSimilarityForUnion)
                        {
                            wgcGraph[items[i]].Add(items[j]);
                            wgcGraph[items[j]].Add(items[i]);
                        }

                        var b1 = Path.GetFileName(items[i]);
                        var b2 = Path.GetFileName(items[j]);
                        var status = ok ? "PASS" : "FAIL";
                        var basePct = totalPairs > 0 ? 45 + (int)((double)pairCount / totalPairs * 35) : 45;
                        progress?.Report(new ProgressData(basePct, $"WGC: {b1} vs {b2}", $"[{status}] sim={sim:F3} angle={angleVotes}v scale={scaleVotes}v"));
                    }

                    pairs.Add((pair, items[i], items[j]));
                    pairCount++;
                }
            }
            clusterPairs[clusterId] = pairs;
            clusterWgcGraph[clusterId] = wgcGraph;
        }

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(80, "Building duplicate groups..."));
        var groups = new List<DuplicateGroup>();
        int groupId = 0;

        if (enableGeometricCheck)
        {
            foreach (var clusterKvp in clusters)
            {
                var clusterId = clusterKvp.Key;
                if (!clusterWgcGraph.ContainsKey(clusterId)) continue;

                var cliques = FindCliques(clusterWgcGraph[clusterId]);
                foreach (var clique in cliques)
                {
                    if (clique.Count < 2) continue;

                    var group = new DuplicateGroup { ClusterId = groupId };
                    var seen = new HashSet<(string, string)>();

                    foreach (var (pair, p1, p2) in clusterPairs[clusterId])
                    {
                        if (!pair.GeometricVerified) continue;
                        if (!clique.Contains(p1) || !clique.Contains(p2)) continue;

                        var key = string.Compare(p1, p2) < 0 ? (p1, p2) : (p2, p1);
                        if (seen.Add(key))
                            group.Pairs.Add(pair);
                    }

                    if (group.Pairs.Count > 0)
                    {
                        var groupFiles = group.Paths.Select(p => Path.GetFileName(p)).ToList();
                        DebugLog.Write($"[GROUP {groupId}] from cluster={clusterId} ({groupFiles.Count} images): {string.Join(", ", groupFiles)}");
                        foreach (var p in group.Pairs)
                            DebugLog.Write($"  PAIR: {Path.GetFileName(p.Path1)} <-> {Path.GetFileName(p.Path2)} sim={p.Similarity:F4} geo={p.GeometricVerified}");

                        groups.Add(group);
                        groupId++;
                    }
                }
            }
        }
        else
        {
            foreach (var clusterKvp in clusters)
            {
                var clusterId = clusterKvp.Key;
                var items = clusterKvp.Value;
                if (items.Count <= 1) continue;
                if (!clusterPairs.ContainsKey(clusterId)) continue;

                var group = new DuplicateGroup { ClusterId = clusterId };
                foreach (var (pair, p1, p2) in clusterPairs[clusterId])
                {
                    group.Pairs.Add(pair);
                }
                if (group.Pairs.Count > 0)
                    groups.Add(group);
            }
        }

        progress?.Report(new ProgressData(100, $"Found {groups.Count} duplicate groups"));
        return groups;
    }

    private (bool ok, float angle, float scale, int angleVotes, int scaleVotes) VerifyGeometric(string path1, string path2)
    {
        var mtime1 = GetMtime(path1);
        var mtime2 = GetMtime(path2);

        var cached = _cache.GetWgc(path1, path2, mtime1, mtime2);
        if (cached.HasValue)
            return (cached.Value.result, cached.Value.angle, cached.Value.scale, cached.Value.angleVotes, cached.Value.scaleVotes);

        var kp1 = GeometricConsistency.ExtractSiftFeaturesWithDescriptors(ImageUtils.ReadImageCv2(path1));
        var kp2 = GeometricConsistency.ExtractSiftFeaturesWithDescriptors(ImageUtils.ReadImageCv2(path2));

        if (kp1.keypoints.Length == 0 || kp2.keypoints.Length == 0 || kp1.descriptors == null || kp2.descriptors == null)
            return (false, 0, 0, 0, 0);

        var result = GeometricConsistency.CheckGeometricConsistency(kp1.keypoints, kp1.descriptors, kp2.keypoints, kp2.descriptors, _wgcThreshold);

        try { _cache.SetWgc(path1, path2, mtime1, mtime2, result.isValid, result.avgAngle, result.avgScale, result.angleVotes, result.scaleVotes); }
        catch { }

        return (result.isValid, result.avgAngle, result.avgScale, result.angleVotes, result.scaleVotes);
    }

    private static double GetMtime(string path)
    {
        try { return File.GetLastWriteTimeUtc(path).Ticks / (double)TimeSpan.TicksPerSecond; }
        catch { return 0.0; }
    }

    public long ClearCache() => _cache.ClearAll();

    private static float DotProduct(float[] a, float[] b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static List<List<string>> FindCliques(Dictionary<string, HashSet<string>> graph)
    {
        var result = new List<List<string>>();
        var candidates = new HashSet<string>(graph.Keys);
        var excluded = new HashSet<string>();
        BronKerbosch(new List<string>(), candidates, excluded, graph, result);
        return result;
    }

    private static void BronKerbosch(
        List<string> current, HashSet<string> candidates, HashSet<string> excluded,
        Dictionary<string, HashSet<string>> graph, List<List<string>> result)
    {
        if (candidates.Count == 0 && excluded.Count == 0)
        {
            if (current.Count >= 2)
                result.Add(new List<string>(current));
            return;
        }

        var candidatesList = candidates.ToList();
        foreach (var v in candidatesList)
        {
            current.Add(v);
            var newCandidates = new HashSet<string>(candidates.Intersect(graph[v]));
            var newExcluded = new HashSet<string>(excluded.Intersect(graph[v]));
            BronKerbosch(current, newCandidates, newExcluded, graph, result);
            current.RemoveAt(current.Count - 1);
            candidates.Remove(v);
            excluded.Add(v);
        }
    }
}
