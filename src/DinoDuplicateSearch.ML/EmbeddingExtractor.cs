using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
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
    private int _batchSize = 32;
    public int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    private int _prefetchCount = 2;
    public int PrefetchCount
    {
        get => _prefetchCount;
        set => _prefetchCount = Math.Max(0, value);
    }

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

    public float[] EmbedImage(string path, CancellationToken ct = default)
    {
        var results = EmbedImagesBatch(new List<string> { path }, ct);
        return results[0];
    }

    private readonly record struct PreparedBatch(
        int BatchStart,
        int TotalToEmbed,
        List<(int Index, string Path, double Mtime)> Meta,
        DenseTensor<float>? Tensor);

    public float[][] EmbedImagesBatch(List<string> paths, CancellationToken ct = default)
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

        if (PrefetchCount == 0)
        {
            ProcessBatchesSequential(toEmbed, result, ct);
            return result;
        }

        var queue = new BlockingCollection<PreparedBatch>(PrefetchCount);
        var totalBatches = (toEmbed.Count + BatchSize - 1) / BatchSize;

        var producer = Task.Run(() =>
        {
            try
            {
                for (int batchStart = 0; batchStart < toEmbed.Count; batchStart += BatchSize)
                {
                    ct.ThrowIfCancellationRequested();
                    var batch = toEmbed.Skip(batchStart).Take(BatchSize).ToList();
                    var prepared = new (int index, string path, double mtime, float[,,] tensor)[batch.Count];

                    Parallel.For(0, batch.Count, new ParallelOptions { CancellationToken = ct }, b =>
                    {
                        var (index, path) = batch[b];
                        var mtime = 0.0;
                        try { mtime = File.GetLastWriteTimeUtc(path).Ticks / (double)TimeSpan.TicksPerSecond; }
                        catch { }

                        var mat = OpenCvSharp.Cv2.ImRead(path);
                        if (mat.Empty())
                        {
                            prepared[b] = (index, path, mtime, null!);
                            return;
                        }

                        OpenCvSharp.Cv2.CvtColor(mat, mat, OpenCvSharp.ColorConversionCodes.BGR2RGB);
                        OpenCvSharp.Cv2.Resize(mat, mat, new OpenCvSharp.Size(224, 224));

                        var tensor = new float[3, 224, 224];
                        for (int y = 0; y < 224; y++)
                            for (int x = 0; x < 224; x++)
                            {
                                var pixel = mat.At<OpenCvSharp.Vec3b>(y, x);
                                tensor[0, y, x] = pixel[0] / 255f;
                                tensor[1, y, x] = pixel[1] / 255f;
                                tensor[2, y, x] = pixel[2] / 255f;
                            }

                        mat.Dispose();
                        prepared[b] = (index, path, mtime, tensor);
                    });

                    int validCount = prepared.Count(p => p.tensor != null);
                    if (validCount == 0)
                    {
                        foreach (var (index, _, _, _) in prepared)
                            result[index] = new float[768];
                        continue;
                    }

                    var batchTensor = new DenseTensor<float>(new[] { validCount, 3, 224, 224 });
                    var batchMeta = new List<(int index, string path, double mtime)>();
                    int slot = 0;

                    foreach (var (index, path, mtime, tensor) in prepared)
                    {
                        if (tensor == null)
                        {
                            result[index] = new float[768];
                            continue;
                        }

                        for (int c = 0; c < 3; c++)
                            for (int y = 0; y < 224; y++)
                                for (int x = 0; x < 224; x++)
                                    batchTensor[slot, c, y, x] = tensor[c, y, x];

                        batchMeta.Add((index, path, mtime));
                        slot++;
                    }

                    queue.Add(new PreparedBatch(batchStart, toEmbed.Count, batchMeta, batchTensor));
                }
            }
            finally
            {
                queue.CompleteAdding();
            }
        }, ct);

        foreach (var batch in queue.GetConsumingEnumerable(ct))
        {
            var pct = 5 + (int)((double)batch.BatchStart / batch.TotalToEmbed * 35);
            var batchNum = batch.BatchStart / BatchSize + 1;
            _progress?.Report(new ProgressData(pct, _useGpu
                ? $"Batch {batchNum}/{totalBatches} on GPU ({batch.Meta.Count})..."
                : $"Batch {batchNum}/{totalBatches} on CPU ({batch.Meta.Count})..."));

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("pixel_values", batch.Tensor!)
            };

            using var gpuResults = _session!.Run(inputs);
            var output = gpuResults.First().AsTensor<float>().ToArray();

            for (int b = 0; b < batch.Meta.Count; b++)
            {
                var embedding = new float[768];
                Array.Copy(output, b * 768, embedding, 0, 768);

                float norm = 0;
                for (int i = 0; i < embedding.Length; i++) norm += embedding[i] * embedding[i];
                norm = MathF.Sqrt(norm);
                if (norm > 0) for (int i = 0; i < embedding.Length; i++) embedding[i] /= norm;

                var (index, path, mtime) = batch.Meta[b];
                result[index] = embedding;

                try { _cache.SetEmbedding(path, mtime, embedding); }
                catch { }
            }
        }

        producer.Wait();
        return result;
    }

    private void ProcessBatchesSequential(List<(int index, string path)> toEmbed, float[][] result, CancellationToken ct)
    {
        for (int batchStart = 0; batchStart < toEmbed.Count; batchStart += BatchSize)
        {
            ct.ThrowIfCancellationRequested();
            var batch = toEmbed.Skip(batchStart).Take(BatchSize).ToList();
            var prepared = new (int index, string path, double mtime, float[,,] tensor)[batch.Count];

            Parallel.For(0, batch.Count, new ParallelOptions { CancellationToken = ct }, b =>
            {
                var (index, path) = batch[b];
                var mtime = 0.0;
                try { mtime = File.GetLastWriteTimeUtc(path).Ticks / (double)TimeSpan.TicksPerSecond; }
                catch { }

                var mat = OpenCvSharp.Cv2.ImRead(path);
                if (mat.Empty())
                {
                    prepared[b] = (index, path, mtime, null!);
                    return;
                }

                OpenCvSharp.Cv2.CvtColor(mat, mat, OpenCvSharp.ColorConversionCodes.BGR2RGB);
                OpenCvSharp.Cv2.Resize(mat, mat, new OpenCvSharp.Size(224, 224));

                var tensor = new float[3, 224, 224];
                for (int y = 0; y < 224; y++)
                    for (int x = 0; x < 224; x++)
                    {
                        var pixel = mat.At<OpenCvSharp.Vec3b>(y, x);
                        tensor[0, y, x] = pixel[0] / 255f;
                        tensor[1, y, x] = pixel[1] / 255f;
                        tensor[2, y, x] = pixel[2] / 255f;
                    }

                mat.Dispose();
                prepared[b] = (index, path, mtime, tensor);
            });

            int validCount = prepared.Count(p => p.tensor != null);
            if (validCount == 0)
            {
                foreach (var (index, _, _, _) in prepared)
                    result[index] = new float[768];
                continue;
            }

            var batchTensor = new DenseTensor<float>(new[] { validCount, 3, 224, 224 });
            var batchMeta = new List<(int index, string path, double mtime)>();
            int slot = 0;

            foreach (var (index, path, mtime, tensor) in prepared)
            {
                if (tensor == null)
                {
                    result[index] = new float[768];
                    continue;
                }

                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < 224; y++)
                        for (int x = 0; x < 224; x++)
                            batchTensor[slot, c, y, x] = tensor[c, y, x];

                batchMeta.Add((index, path, mtime));
                slot++;
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("pixel_values", batchTensor)
            };

            var pct = 5 + (int)((double)batchStart / toEmbed.Count * 35);
            var batchNum = batchStart / BatchSize + 1;
            var totalBatches = (toEmbed.Count + BatchSize - 1) / BatchSize;
            _progress?.Report(new ProgressData(pct, _useGpu
                ? $"Batch {batchNum}/{totalBatches} on GPU ({validCount})..."
                : $"Batch {batchNum}/{totalBatches} on CPU ({validCount})..."));

            using var gpuResults = _session!.Run(inputs);
            var output = gpuResults.First().AsTensor<float>().ToArray();

            for (int b = 0; b < batchMeta.Count; b++)
            {
                var embedding = new float[768];
                Array.Copy(output, b * 768, embedding, 0, 768);

                float norm = 0;
                for (int i = 0; i < embedding.Length; i++) norm += embedding[i] * embedding[i];
                norm = MathF.Sqrt(norm);
                if (norm > 0) for (int i = 0; i < embedding.Length; i++) embedding[i] /= norm;

                var (index, path, mtime) = batchMeta[b];
                result[index] = embedding;

                try { _cache.SetEmbedding(path, mtime, embedding); }
                catch { }
            }
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
