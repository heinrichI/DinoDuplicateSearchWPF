using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DinoDuplicateSearch.Database;

namespace DinoDuplicateSearch.ML;

public class EmbeddingExtractor : IDisposable
{
    private readonly string _modelPath;
    private InferenceSession? _session;
    private readonly FeatureCache _cache;
    private Action<double, string>? _progressCallback;

    public EmbeddingExtractor(string modelPath = "Models/dinov2-base.onnx", FeatureCache? cache = null)
    {
        _modelPath = modelPath;
        _cache = cache ?? new FeatureCache();
    }

    public void SetProgressCallback(Action<double, string>? callback)
    {
        _progressCallback = callback;
    }

    private void LoadModel()
    {
        if (_session != null) return;
        _progressCallback?.Invoke(0, "Loading ONNX model...");
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        try
        {
            options.AppendExecutionProvider_CUDA();
            _progressCallback?.Invoke(5, "Using GPU (CUDA)...");
        }
        catch (Exception ex)
        {
            _progressCallback?.Invoke(5, "GPU unavailable (install cuDNN for CUDA), using CPU...");
        }
        _session = new InferenceSession(_modelPath, options);
        _progressCallback?.Invoke(100, "Model loaded");
    }

    public float[] EmbedImage(string path)
    {
        var currentMtime = 0.0;
        try { currentMtime = File.GetLastWriteTimeUtc(path).Ticks / (double)TimeSpan.TicksPerSecond; }
        catch { }

        var cached = _cache.GetEmbedding(path);
        if (cached.HasValue && Math.Abs(cached.Value.mtime - currentMtime) < 0.01)
            return cached.Value.embedding;

        LoadModel();

        _progressCallback?.Invoke(-1, $"Reading image...\n{Path.GetFileName(path)}");
        using var mat = OpenCvSharp.Cv2.ImRead(path);
        if (mat.Empty()) return new float[768];

        OpenCvSharp.Cv2.CvtColor(mat, mat, OpenCvSharp.ColorConversionCodes.BGR2RGB);
        OpenCvSharp.Cv2.Resize(mat, mat, new OpenCvSharp.Size(224, 224));

        _progressCallback?.Invoke(-1, $"Preparing tensor...\n{Path.GetFileName(path)}");
        var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                var pixel = mat.At<OpenCvSharp.Vec3b>(y, x);
                input[0, 0, y, x] = pixel[0] / 255f;
                input[0, 1, y, x] = pixel[1] / 255f;
                input[0, 2, y, x] = pixel[2] / 255f;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", input)
        };

        _progressCallback?.Invoke(-1, $"Running inference on GPU...\n{Path.GetFileName(path)}");
        using var results = _session!.Run(inputs);
        var output = results.First().AsTensor<float>().ToArray();

        var cls = new float[768];
        Array.Copy(output, cls, 768);

        float norm = 0;
        for (int i = 0; i < cls.Length; i++) norm += cls[i] * cls[i];
        norm = MathF.Sqrt(norm);
        if (norm > 0) for (int i = 0; i < cls.Length; i++) cls[i] /= norm;

        try { _cache.SetEmbedding(path, currentMtime, cls); }
        catch { }

        return cls;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
