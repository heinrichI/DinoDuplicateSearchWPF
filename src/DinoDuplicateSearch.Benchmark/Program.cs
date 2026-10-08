using System.Diagnostics;
using System.IO;
using DinoDuplicateSearch.Abstractions;
using DinoDuplicateSearch.Core;
using DinoDuplicateSearch.CV;
using DinoDuplicateSearch.Database;
using DinoDuplicateSearch.ML;
using DinoDuplicateSearch.Models;

namespace DinoDuplicateSearch.Benchmark;

class Program
{
    static void Main(string[] args)
    {
        string testDir = args.Length > 0 ? args[0] : @"F:\E\SourceAntiDupl\TestBluring";
        // необязательный размер батча (дефолт 32), например: "TestCropHard" 1
        int batchSize = args.Length > 1 && int.TryParse(args[1], out var bs) ? bs : 32;
        string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");

        Console.WriteLine("=== Duplicate Image Detection Benchmark ===");
        Console.WriteLine($"Test folder: {testDir}");
        Console.WriteLine($"Embedding batch size: {batchSize}");
        Console.WriteLine($"Models: {modelDir}");
        Console.WriteLine();

        var images = DuplicatesFinder.ListImages(testDir, false);
        Console.WriteLine($"Found {images.Count} images");
        Console.WriteLine();

        // Benchmark configurations
        var configs = new (string Name, SearchSettings Settings)[]
        {
            ("SIFT (baseline)", new SearchSettings
            {
                DirectoryPath = testDir,
                DistanceThreshold = 0.45f,
                GeometricCheckEnabled = true,
                UseLightGlue = false,
                MinMatchScore = 0.5f,
                MaxClusterSize = 50,
                TransitivityRatio = 0.7f,
                WgcParallelism = Environment.ProcessorCount
            }),
            ("LightGlue (default)", new SearchSettings
            {
                DirectoryPath = testDir,
                DistanceThreshold = 0.45f,
                GeometricCheckEnabled = true,
                UseLightGlue = true,
                MinMatchScore = 0.5f,
                MaxClusterSize = 50,
                TransitivityRatio = 0.7f,
                WgcParallelism = Environment.ProcessorCount
            }),
            ("LightGlue (relaxed)", new SearchSettings
            {
                DirectoryPath = testDir,
                DistanceThreshold = 0.60f,
                GeometricCheckEnabled = true,
                UseLightGlue = true,
                MinMatchScore = 0.3f,
                MaxClusterSize = 50,
                TransitivityRatio = 0.7f,
                WgcParallelism = Environment.ProcessorCount
            }),
            ("LightGlue (strict)", new SearchSettings
            {
                DirectoryPath = testDir,
                DistanceThreshold = 0.45f,
                GeometricCheckEnabled = true,
                UseLightGlue = true,
                MinMatchScore = 0.7f,
                MaxClusterSize = 50,
                TransitivityRatio = 0.7f,
                WgcParallelism = Environment.ProcessorCount
            }),
            ("No geometric check", new SearchSettings
            {
                DirectoryPath = testDir,
                DistanceThreshold = 0.45f,
                GeometricCheckEnabled = false,
                UseLightGlue = false,
                MinMatchScore = 0.5f,
                MaxClusterSize = 50,
                TransitivityRatio = 0.7f,
                WgcParallelism = Environment.ProcessorCount
            }),
        };

        var results = new List<BenchmarkResult>();

        foreach (var (name, settings) in configs)
        {
            Console.Write($"Running: {name}... ");
            var result = RunBenchmark(name, settings, modelDir, batchSize);
            results.Add(result);
            Console.WriteLine($"Done ({result.ElapsedMs}ms, {result.Groups} groups, {result.TotalPairs} pairs)");
        }

        // Print summary table
        Console.WriteLine();
        Console.WriteLine("=== Results ===");
        Console.WriteLine($"{"Config".PadRight(25)} {"Time".PadLeft(8)} {"Groups".PadLeft(7)} {"Pairs".PadLeft(6)} {"Backend"}");
        Console.WriteLine(new string('-', 80));
        foreach (var r in results)
        {
            Console.WriteLine($"{r.Name.PadRight(25)} {r.ElapsedMs.ToString().PadLeft(6)}ms {r.Groups.ToString().PadLeft(7)} {r.TotalPairs.ToString().PadLeft(6)} {r.Backend}");
        }

        // Print pair details for each result
        Console.WriteLine();
        foreach (var r in results)
        {
            Console.WriteLine($"--- {r.Name} ---");
            if (r.Groups == 0)
            {
                Console.WriteLine("  No duplicates found");
            }
            else
            {
                foreach (var g in r.GroupDetails)
                {
                    Console.WriteLine($"  Group ({g.ImageCount} images):");
                    foreach (var p in g.PairSummaries)
                        Console.WriteLine($"    {p}");
                }
            }
            Console.WriteLine();
        }
    }

    static BenchmarkResult RunBenchmark(string name, SearchSettings settings, string modelDir, int batchSize)
    {
        var sw = Stopwatch.StartNew();
        var cache = new FeatureCache();
        var embeddingExtractor = new EmbeddingExtractor(Path.Combine(modelDir, "dinov2-base.onnx"), cache);
        embeddingExtractor.BatchSize = batchSize;
        var pipelinePath = Path.Combine(modelDir, "superpoint_lightglue_pipeline.onnx");
        var pipeline = new SuperPointLightGluePipeline(pipelinePath);
        var lightGlueVerifier = new LightGlueGeometricVerifier(pipeline);
        var siftVerifier = new SiftGeometricVerifier();
        var factory = new GeometricVerifierFactory(lightGlueVerifier, siftVerifier);
        var finder = new DuplicatesFinder(cache, embeddingExtractor, factory);
        var groups = finder.FindDuplicates(settings);
        sw.Stop();

        int totalPairs = groups.Sum(g => g.Pairs.Count);
        var backend = finder.LastFeatureBackend;
        finder.Dispose();

        var groupDetails = groups.Select(g => new GroupDetail
        {
            ImageCount = g.Paths.Count,
            PairSummaries = g.Pairs.Select(p =>
                $"{Path.GetFileName(p.Path1)} <-> {Path.GetFileName(p.Path2)} sim={p.Similarity:F3}"
            ).ToList()
        }).ToList();

        return new BenchmarkResult
        {
            Name = name,
            ElapsedMs = (int)sw.ElapsedMilliseconds,
            Groups = groups.Count,
            TotalPairs = totalPairs,
            Backend = backend,
            GroupDetails = groupDetails
        };
    }
}

record BenchmarkResult
{
    public string Name { get; init; } = "";
    public int ElapsedMs { get; init; }
    public int Groups { get; init; }
    public int TotalPairs { get; init; }
    public string Backend { get; init; } = "";
    public List<GroupDetail> GroupDetails { get; init; } = new();
}

record GroupDetail
{
    public int ImageCount { get; init; }
    public List<string> PairSummaries { get; init; } = new();
}
