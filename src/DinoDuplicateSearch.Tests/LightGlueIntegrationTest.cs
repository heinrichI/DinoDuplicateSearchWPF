using System.IO;
using DinoDuplicateSearch.CV;
using DinoDuplicateSearch.Database;
using DinoDuplicateSearch.ML;
using OpenCvSharp;
using Xunit.Abstractions;

namespace DinoDuplicateSearch.Tests;

public class LightGlueIntegrationTest
{
    private readonly ITestOutputHelper _output;
    private readonly string _modelsDir;
    private readonly string _testDir;

    public LightGlueIntegrationTest(ITestOutputHelper output)
    {
        _output = output;
        _modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
        _testDir = @"F:\E\SourceAntiDupl\TestBluring";
    }

    [Fact]
    public void SuperPoint_ExtractsKeypoints()
    {
        var modelPath = Path.Combine(_modelsDir, "superpoint.onnx");
        if (!File.Exists(modelPath)) { _output.WriteLine($"SKIP: {modelPath} not found"); return; }

        using var sp = new SuperPointExtractor(modelPath);
        using var img = Cv2.ImRead(Path.Combine(_testDir, "22.jpg"));
        Assert.False(img.Empty());

        var (kpts, descs) = sp.Extract(img);
        _output.WriteLine($"SuperPoint: {kpts.Length / 2} keypoints, desc [{descs.GetLength(0)}, {descs.GetLength(1)}]");
        Assert.True(kpts.Length >= 4);
        Assert.Equal(256, descs.GetLength(1));
    }

    [Fact]
    public void LightGlue_MatchesSimilarImages()
    {
        var spPath = Path.Combine(_modelsDir, "superpoint.onnx");
        var lgPath = Path.Combine(_modelsDir, "lightglue.onnx");
        if (!File.Exists(spPath) || !File.Exists(lgPath)) { _output.WriteLine("SKIP: models not found"); return; }

        using var sp = new SuperPointExtractor(spPath);
        using var lg = new LightGlueMatcher(lgPath);
        using var img1 = Cv2.ImRead(Path.Combine(_testDir, "22.jpg"));
        using var img2 = Cv2.ImRead(Path.Combine(_testDir, "22_2.jpg"));

        var (kp1, des1) = sp.Extract(img1);
        var (kp2, des2) = sp.Extract(img2);
        _output.WriteLine($"Kpts: {kp1.Length / 2} vs {kp2.Length / 2}");

        var matches = lg.Match(kp1, des1, Math.Max(img1.Rows, img1.Cols), kp2, des2, Math.Max(img2.Rows, img2.Cols), 0.5f);
        _output.WriteLine($"LightGlue matches: {matches.Length}");
        Assert.True(matches.Length > 5, $"Expected >5 matches, got {matches.Length}");
    }

    [Fact]
    public void DINOv2_SimilarImagesHighSimilarity()
    {
        var modelPath = Path.Combine(_modelsDir, "dinov2-base.onnx");
        if (!File.Exists(modelPath)) { _output.WriteLine("SKIP: dinov2 not found"); return; }

        using var cache = new FeatureCache(Path.GetTempFileName());
        using var extractor = new EmbeddingExtractor(modelPath, cache);
        var emb1 = extractor.EmbedImage(Path.Combine(_testDir, "22.jpg"));
        var emb2 = extractor.EmbedImage(Path.Combine(_testDir, "22_2.jpg"));

        float sim = 0;
        for (int i = 0; i < emb1.Length; i++) sim += emb1[i] * emb2[i];
        _output.WriteLine($"DINOv2 similarity: {sim:F4}");
        Assert.True(sim > 0.7f, $"Expected >0.7, got {sim:F4}");
    }

    [Fact]
    public void LightGlue_GeometricConsistency_Passes()
    {
        var spPath = Path.Combine(_modelsDir, "superpoint.onnx");
        var lgPath = Path.Combine(_modelsDir, "lightglue.onnx");
        if (!File.Exists(spPath) || !File.Exists(lgPath)) { _output.WriteLine("SKIP: models not found"); return; }

        using var sp = new SuperPointExtractor(spPath);
        using var lg = new LightGlueMatcher(lgPath);
        using var img1 = Cv2.ImRead(Path.Combine(_testDir, "22.jpg"));
        using var img2 = Cv2.ImRead(Path.Combine(_testDir, "22_2.jpg"));

        var (kp1, des1) = sp.Extract(img1);
        var (kp2, des2) = sp.Extract(img2);

        var result = GeometricConsistency.CheckGeometricConsistencyLightGlue(
            kp1, des1, Math.Max(img1.Rows, img1.Cols),
            kp2, des2, Math.Max(img2.Rows, img2.Cols),
            lg, 0.3f);

        _output.WriteLine($"WGC: valid={result.isValid}, angle={result.avgAngle:F1}, matches={result.matchCount}");
        Assert.True(result.isValid, "Expected geometric consistency");
        Assert.True(result.matchCount >= 10, $"Expected >=10, got {result.matchCount}");
    }
}
