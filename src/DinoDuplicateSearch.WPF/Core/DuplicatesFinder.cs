using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading.Tasks;
using DinoDuplicateSearch.Models;
using DinoDuplicateSearch.Database;
using DinoDuplicateSearch.CV;
using DinoDuplicateSearch.ML;
using DinoDuplicateSearch.WPF.Core;

namespace DinoDuplicateSearch.Core;

public class DuplicatesFinder
{
    private const int PQThreshold = 50_000;
    private const int PQSubvectorCount = 96;
    private const int PQCentroidsPerSubspace = 256;
    private const int PQTopK = 5;

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
        SearchSettings settings,
        IProgress<ProgressData>? progress = null,
        CancellationToken ct = default)
    {
        _wgcThreshold = settings.WgcThreshold;
        _minSimilarityForUnion = settings.MinSimilarityForPair;
        _embeddingExtractor.SetProgress(progress);
        _embeddingExtractor.BatchSize = settings.BatchSize;
        _embeddingExtractor.PrefetchCount = settings.PrefetchCount;

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(0, "Scanning folder for images..."));
        var paths = ListImages(settings.DirectoryPath, settings.SearchSubfolders);
        if (paths.Count == 0)
        {
            progress?.Report(new ProgressData(100, "No images found"));
            return new List<DuplicateGroup>();
        }
        progress?.Report(new ProgressData(2, $"Found {paths.Count} images. Loading model..."));

        progress?.Report(new ProgressData(5, "Computing embeddings..."));
        var embs = _embeddingExtractor.EmbedImagesBatch(paths, ct);

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(40, "Clustering images..."));
        int[] labels;
        if (embs.Length > PQThreshold)
        {
            progress?.Report(new ProgressData(40, $"Large dataset ({embs.Length} images), using Product Quantization..."));
            labels = ClusterWithPQ(embs, settings.DistanceThreshold, settings.TransitivityRatio, settings.DirectoryPath, progress, ct);
        }
        else
        {
            labels = AgglomerativeClustering.FitPredict(embs, settings.DistanceThreshold, progress, settings.TransitivityRatio);
        }

        int maxClusterSize = settings.MaxClusterSize;
        float transitivityRatio = settings.TransitivityRatio;
        var labelCounts = new Dictionary<int, int>();
        for (int i = 0; i < labels.Length; i++)
        {
            if (!labelCounts.ContainsKey(labels[i]))
                labelCounts[labels[i]] = 0;
            labelCounts[labels[i]]++;
        }
        int nextLabel = labels.Length > 0 ? labels.Max() + 1 : 0;
        for (int i = 0; i < labels.Length; i++)
        {
            if (labelCounts[labels[i]] > maxClusterSize)
                labels[i] = nextLabel++;
        }

        var clusters = new Dictionary<int, List<string>>();
        for (int i = 0; i < paths.Count; i++)
        {
            if (!clusters.ContainsKey(labels[i]))
                clusters[labels[i]] = new List<string>();
            clusters[labels[i]].Add(paths[i]);
        }

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(45, "Verifying geometric consistency..."));
        int totalPairs = clusters.Values.Sum(items => items.Count * (items.Count - 1) / 2);
        var bigClusters = clusters.Values.Where(items => items.Count > 100).OrderByDescending(items => items.Count).Take(5);
        foreach (var items in bigClusters)
            DebugLog.Write($"[WGC] Big cluster: {items.Count} images, {items.Count * (items.Count - 1) / 2} pairs");
        DebugLog.Write($"[WGC] Total clusters: {clusters.Count}, total pairs: {totalPairs}");

        var clusterPairs = new System.Collections.Concurrent.ConcurrentDictionary<int, List<(DuplicatePair pair, string p1, string p2)>>();
        var clusterWgcGraph = new System.Collections.Concurrent.ConcurrentDictionary<int, Dictionary<string, HashSet<string>>>();
        int pairCount = 0;

        Parallel.ForEach(clusters, new ParallelOptions { CancellationToken = ct }, clusterKvp =>
        {
            var clusterId = clusterKvp.Key;
            var items = clusterKvp.Value;
            if (items.Count <= 1) return;

            var pairs = new List<(DuplicatePair pair, string p1, string p2)>();
            var wgcGraph = new Dictionary<string, HashSet<string>>();
            foreach (var item in items) wgcGraph[item] = new HashSet<string>();

            for (int i = 0; i < items.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
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

                    if (settings.GeometricCheckEnabled)
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
                            lock (wgcGraph[items[i]]) wgcGraph[items[i]].Add(items[j]);
                            lock (wgcGraph[items[j]]) wgcGraph[items[j]].Add(items[i]);
                        }

                        var b1 = Path.GetFileName(items[i]);
                        var b2 = Path.GetFileName(items[j]);
                        var status = ok ? "PASS" : "FAIL";
                        int done = Interlocked.Increment(ref pairCount);
                        var basePct = totalPairs > 0 ? 45 + (int)((double)done / totalPairs * 35) : 45;
                        progress?.Report(new ProgressData(basePct, $"WGC: {done}/{totalPairs} {b1} vs {b2}", $"[{status}] sim={sim:F3} angle={angleVotes}v scale={scaleVotes}v"));
                    }

                    pairs.Add((pair, items[i], items[j]));
                }
            }
            clusterPairs[clusterId] = pairs;
            clusterWgcGraph[clusterId] = wgcGraph;
        });

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(80, "Building duplicate groups..."));
        var groups = new List<DuplicateGroup>();
        int groupId = 0;

        if (settings.GeometricCheckEnabled)
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

    private int[] ClusterWithPQ(float[][] embeddings, float distanceThreshold, float transitivityRatio, string directoryPath, IProgress<ProgressData>? progress, CancellationToken ct)
    {
        int n = embeddings.Length;
        int dim = embeddings[0].Length;
        var pq = new ProductQuantizer(dim, PQSubvectorCount, PQCentroidsPerSubspace);
        string cacheDir = AppDomain.CurrentDomain.BaseDirectory;
        string centroidsPath = Path.Combine(cacheDir, "pq_centroids.bin");

        progress?.Report(new ProgressData(40, "PQ: Loading centroids..."));
        if (pq.TryLoadCentroids(centroidsPath, embeddings))
        {
            progress?.Report(new ProgressData(42, "PQ: Centroids loaded from cache"));
        }
        else
        {
            progress?.Report(new ProgressData(40, "PQ: Training centroids..."));
            var pqProgress = new Progress<string>(msg => progress?.Report(new ProgressData(41, $"PQ: {msg}")));
            pq.Train(embeddings, maxIterations: 50, seed: 42, progress: pqProgress);
            pq.SaveCentroids(centroidsPath, embeddings);
        }

        ct.ThrowIfCancellationRequested();
        progress?.Report(new ProgressData(42, "PQ: Encoding database..."));
        var encodedDb = pq.Encode(embeddings);

        ct.ThrowIfCancellationRequested();
        string neighborsPath = Path.Combine(cacheDir, "pq_neighbors.bin");
        var neighbors = new System.Collections.Concurrent.ConcurrentBag<(int, int)>();
        float pqThreshold = 2 * distanceThreshold;

        if (!TryLoadNeighbors(neighborsPath, embeddings, neighbors))
        {
            progress?.Report(new ProgressData(43, "PQ: Searching nearest neighbors..."));
            int completed = 0;
            int reportInterval = Math.Max(1, n / 100);

            Parallel.For(0, n, i =>
            {
                ct.ThrowIfCancellationRequested();
                var results = pq.Search(encodedDb, embeddings[i], PQTopK + 1);
                foreach (var result in results)
                {
                    if (result.index != i && result.distance < pqThreshold)
                        neighbors.Add((i, result.index));
                }

                int done = Interlocked.Increment(ref completed);
                if (done % reportInterval == 0)
                    progress?.Report(new ProgressData(43.0 + 2.0 * done / n, $"PQ: {done}/{n}"));
            });

            SaveNeighbors(neighborsPath, embeddings, neighbors.ToArray());
        }
        else
        {
            progress?.Report(new ProgressData(45, "PQ: Neighbors loaded from cache"));
        }

        progress?.Report(new ProgressData(45, "PQ: Union-Find clustering..."));
        var edgeList = new List<(int i, int j, float dist)>();
        foreach (var pair in neighbors)
        {
            float dist = 1 - DotProduct(embeddings[pair.Item1], embeddings[pair.Item2]);
            edgeList.Add((pair.Item1, pair.Item2, dist));
        }
        edgeList.Sort((a, b) => a.dist.CompareTo(b.dist));

        float pqTransitivityThreshold = distanceThreshold * transitivityRatio;
        var uf = new UnionFind(n);
        foreach (var (i, j, dist) in edgeList)
        {
            if (uf.Find(i) == uf.Find(j)) continue;
            if (dist < pqTransitivityThreshold)
            {
                uf.Union(i, j);
            }
            else if (uf.Size(uf.Find(i)) + uf.Size(uf.Find(j)) <= 50)
            {
                uf.Union(i, j);
            }
        }

        return uf.GetLabels();
    }

    private bool TryLoadNeighbors(string path, float[][] embeddings, ConcurrentBag<(int, int)> neighbors)
    {
        if (!File.Exists(path)) return false;

        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            long storedHash = br.ReadInt64();
            int count = br.ReadInt32();

            long currentHash = ComputeEmbeddingsHash(embeddings);
            if (storedHash != currentHash) return false;

            for (int i = 0; i < count; i++)
            {
                int a = br.ReadInt32();
                int b = br.ReadInt32();
                neighbors.Add((a, b));
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private void SaveNeighbors(string path, float[][] embeddings, (int, int)[] neighbors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        bw.Write(ComputeEmbeddingsHash(embeddings));
        bw.Write(neighbors.Length);

        foreach (var (a, b) in neighbors)
        {
            bw.Write(a);
            bw.Write(b);
        }
    }

    private static long ComputeEmbeddingsHash(float[][] embeddings)
    {
        long hash = embeddings.Length;
        int step = Math.Max(1, embeddings.Length / 1000);
        for (int i = 0; i < embeddings.Length; i += step)
        {
            for (int j = 0; j < embeddings[i].Length; j += 8)
                hash = hash * 31 + BitConverter.SingleToInt32Bits(embeddings[i][j]);
        }
        return hash;
    }

    private class UnionFind
    {
        private readonly int[] _parent;
        private readonly int[] _size;

        public UnionFind(int n)
        {
            _parent = new int[n];
            _size = new int[n];
            for (int i = 0; i < n; i++)
            {
                _parent[i] = i;
                _size[i] = 1;
            }
        }

        public int Find(int x)
        {
            while (_parent[x] != x)
            {
                _parent[x] = _parent[_parent[x]];
                x = _parent[x];
            }
            return x;
        }

        public void Union(int a, int b)
        {
            a = Find(a);
            b = Find(b);
            if (a == b) return;
            if (_size[a] < _size[b])
                (a, b) = (b, a);
            _parent[b] = a;
            _size[a] += _size[b];
        }

        public int Size(int x) => _size[Find(x)];

        public int[] GetLabels()
        {
            int n = _parent.Length;
            var rootToLabel = new Dictionary<int, int>();
            int nextLabel = 0;
            var labels = new int[n];
            for (int i = 0; i < n; i++)
            {
                int root = Find(i);
                if (!rootToLabel.TryGetValue(root, out int label))
                {
                    label = nextLabel++;
                    rootToLabel[root] = label;
                }
                labels[i] = label;
            }
            return labels;
        }
    }

    private (bool ok, float angle, float scale, int angleVotes, int scaleVotes) VerifyGeometric(string path1, string path2)
    {
        var mtime1 = GetMtime(path1);
        var mtime2 = GetMtime(path2);

        var cached = _cache.GetWgc(path1, path2, mtime1, mtime2);
        if (cached.HasValue)
        {
            DebugLog.Write($"[WGC] Cache hit: {path1} {path2} {mtime1} {mtime2}");
            return (cached.Value.result, cached.Value.angle, cached.Value.scale, cached.Value.angleVotes, cached.Value.scaleVotes);
        }

        var kp1 = GeometricConsistency.ExtractSiftFeaturesWithDescriptors(ImageUtils.ReadImageCv2(path1));
        var kp2 = GeometricConsistency.ExtractSiftFeaturesWithDescriptors(ImageUtils.ReadImageCv2(path2));

        if (kp1.keypoints.Length == 0 || kp2.keypoints.Length == 0 || kp1.descriptors == null || kp2.descriptors == null)
            return (false, 0, 0, 0, 0);

        var result = GeometricConsistency.CheckGeometricConsistency(kp1.keypoints, kp1.descriptors, kp2.keypoints, kp2.descriptors, _wgcThreshold);

        try { _cache.SetWgc(path1, path2, mtime1, mtime2, result.isValid, result.avgAngle, result.avgScale, result.angleVotes, result.scaleVotes); }
        catch (Exception ex) { DebugLog.Write($"[WGC] Cache write error: {ex.Message}"); }

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
