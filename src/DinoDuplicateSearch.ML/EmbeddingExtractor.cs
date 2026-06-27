using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DinoDuplicateSearch.Database;
using DinoDuplicateSearch.Models;

namespace DinoDuplicateSearch.ML;

public class EmbeddingExtractor : IDisposable
{
    private readonly string _modelPath;
    private InferenceSession? _session;
    private readonly FeatureCache _cache;
    private IProgress<ProgressData>? _progress;
    private bool _useGpu;
    private const int BatchSize = 8;

    public EmbeddingExtractor(string modelPath = "Models/dinov2-base.onnx", FeatureCache? cache = null)
    {
        _modelPath = modelPath;
        _cache = cache ?? new FeatureCache();
    }

    public void SetProgress(IProgress<ProgressData>? progress)
    {
        _progress = progress;
    }

    private void LoadModel()
    {
        if (_session != null) return;
        _progress?.Report(new ProgressData(0, "Loading ONNX model..."));
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        try
        {
            options.AppendExecutionProvider_CUDA();
            _useGpu = true;
            _progress?.Report(new ProgressData(5, "Using GPU (CUDA)..."));
            DebugLog.Write("[MODEL] GPU (CUDA) available, using GPU");
        }
        catch
        {
            _progress?.Report(new ProgressData(5, "GPU unavailable (install cuDNN for CUDA), using CPU..."));
            DebugLog.Write("[MODEL] GPU unavailable, using CPU");
        }
        _session = new InferenceSession(_modelPath, options);
        _progress?.Report(new ProgressData(100, "Model loaded"));
    }

    public float[] EmbedImage(string path)
    {
        var results = EmbedImagesBatch(new[] { path });
        return results[0];
    }

    public List<float[]> EmbedImagesBatch(List<string> paths)
    {
        LoadModel();

        var result = new float[paths.Count][];
        var toEmbed = new List<(int index, string path)>();

        for (int i = 0; i < paths.Count; i++)
        {
            var currentMtime = 0.0;
            try { currentMtime = File.GetLastWriteTimeUtc(paths[i]).Ticks / (double)TimeSpan.TicksPerSecond; }
            catch { }

            var cached = _cache.GetEmbedding(paths[i]);
            if (cached.HasValue && Math.Abs(cached.Value.mtime - currentMtime) < 0.01)
            {
                result[i] = cached.Value.embedding;
            }
            else
            {
                toEmbed.Add((i, paths[i]));
            }
        }

        if (toEmbed.Count == 0)
            return result;

        for (int batchStart = 0; batchStart < toEmbed.Count; batchStart += BatchSize)
        {
            var batch = toEmbed.Skip(batchStart).Take(BatchSize).ToList();
            var batchTensors = new List<DenseTensor<float>>();
            var batchPaths = new List<(int index, string path, double mtime)>();

            foreach (var (index, path) in batch)
            {
                var mtime = 0.0;
                try { mtime = File.GetLastWriteTimeUtc(path).Ticks / (double)TimeSpan.TicksPerSecond; }
                catch { }

                var basename = Path.GetFileName(path);
                var pct = 5 + (int)((double)(batchStart + batch.IndexOf((index, path))) / toEmbed.Count * 35);
                _progress?.Report(new ProgressData(pct, $"Embedding ({batchStart + batch.IndexOf((index, path)) + 1}/{toEmbed.Count})", basename));

                using var mat = OpenCvSharp.Cv2.ImRead(path);
                if (mat.Empty())
                {
                    result[index] = new float[768];
                    continue;
                }

                OpenCvSharp.Cv2.CvtColor(mat, mat, OpenCvSharp.ColorConversionCodes.BGR2RGB);
                OpenCvSharp.Cv2.Resize(mat, mat, new OpenCvSharp.Size(224, 224));

                var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                for (int y = 0; y < 224; y++)
                    for (int x = 0; x < 224; x++)
                    {
                        var pixel = mat.At<OpenCvSharp.Vec3b>(y, x);
                        tensor[0, 0, y, x] = pixel[0] / 255f;
                        tensor[0, 1, y, x] = pixel[1] / 255f;
                        tensor[0, 2, y, x] = pixel[2] / 255f;
                    }

                batchTensors.Add(tensor);
                batchPaths.Add((index, path, mtime));
            }

            if (batchTensors.Count == 0) continue;

            var batchTensor = new DenseTensor<float>(new[] { batchTensors.Count, 3, 224, 224 });
            for (int b = 0; b < batchTensors.Count; b++)
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < 224; y++)
                        for (int x = 0; x < 224; x++)
                            batchTensor[b, c, y, x] = batchTensors[b][0, c, y, x];

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("pixel_values", batchTensor)
            };

            _progress?.Report(new ProgressData(-1, _useGpu ? $"Running batch on GPU ({batchTensors.Count})..." : $"Running batch on CPU ({batchTensors.Count})..."));
            using var results = _session!.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();

            for (int b = 0; b < batchPaths.Count; b++)
            {
                var embedding = new float[768];
                Array.Copy(output, b * 768, embedding, 0, 768);

                float norm = 0;
                for (int i = 0; i < embedding.Length; i++) norm += embedding[i] * embedding[i];
                norm = MathF.Sqrt(norm);
                if (norm > 0) for (int i = 0; i < embedding.Length; i++) embedding[i] /= norm;

                var (index, path, mtime) = batchPaths[b];
                result[index] = embedding;

                try { _cache.SetEmbedding(path, mtime, embedding); }
                catch { }
            }

            foreach (var t in batchTensors) t.Dispose();
        }

        return result;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
