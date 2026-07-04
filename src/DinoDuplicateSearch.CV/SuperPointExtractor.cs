using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace DinoDuplicateSearch.CV;

public class SuperPointExtractor : IDisposable
{
    private readonly string _modelPath;
    private InferenceSession? _session;
    private readonly int _maxKeypoints;
    private string? _inputName;
    private string? _scoreOutputName;
    private string? _descOutputName;

    public SuperPointExtractor(string modelPath = "Models/superpoint.onnx", int maxKeypoints = 2000)
    {
        _modelPath = modelPath;
        _maxKeypoints = maxKeypoints;
    }

    private void LoadModel()
    {
        if (_session != null) return;
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        try { options.AppendExecutionProvider_CUDA(); }
        catch { }
        _session = new InferenceSession(_modelPath, options);

        // Auto-detect input/output names from model metadata
        _inputName = _session.InputMetadata.Keys.First();
        var outputNames = _session.OutputMetadata.Keys.ToList();
        // Typically: first output = scores, second = descriptors
        _scoreOutputName = outputNames.Count > 0 ? outputNames[0] : "scores";
        _descOutputName = outputNames.Count > 1 ? outputNames[1] : "descriptors";

        System.Diagnostics.Debug.WriteLine($"[SuperPoint] Input: {_inputName}, Outputs: {string.Join(", ", outputNames)}");
    }

    public (float[] keypoints, float[,] descriptors) Extract(Mat image)
    {
        LoadModel();

        using var gray = new Mat();
        if (image.Channels() > 1)
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        else
            image.CopyTo(gray);

        int origH = gray.Rows;
        int origW = gray.Cols;

        // Cap at 1024px max to avoid OOM
        const int maxSize = 1024;
        float scale = 1.0f;
        if (origH > maxSize || origW > maxSize)
        {
            scale = (float)maxSize / Math.Max(origH, origW);
        }

        int h = (int)(origH * scale / 8) * 8;
        int w = (int)(origW * scale / 8) * 8;
        if (h < 8) h = 8;
        if (w < 8) w = 8;
        using var resized = new Mat();
        Cv2.Resize(gray, resized, new Size(w, h));

        var inputTensor = new DenseTensor<float>(new[] { 1, 1, h, w });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                inputTensor[0, 0, y, x] = resized.At<byte>(y, x) / 255f;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName!, inputTensor)
        };

        using var results = _session!.Run(inputs);

        var scoreTensor = results.First(r => r.Name == _scoreOutputName).AsTensor<float>();
        var descTensor = results.First(r => r.Name == _descOutputName).AsTensor<float>();

        var scoreDims = scoreTensor.Dimensions;
        int descH = scoreDims[2];
        int descW = scoreDims[3];

        // Extract score map
        var scores = new float[descH * descW];
        for (int y = 0; y < descH; y++)
            for (int x = 0; x < descW; x++)
                scores[y * descW + x] = scoreTensor[0, 0, y, x];

        var nmsScores = NonMaxSuppression(scores, descW, descH, 4);

        var candidates = new List<(int idx, float score)>();
        for (int i = 0; i < nmsScores.Length; i++)
        {
            if (nmsScores[i] > 0)
                candidates.Add((i, nmsScores[i]));
        }
        candidates.Sort((a, b) => b.score.CompareTo(a.score));
        if (candidates.Count > _maxKeypoints)
            candidates = candidates.Take(_maxKeypoints).ToList();

        int numKp = candidates.Count;
        var kpCoords = new float[numKp * 2];
        var descriptors = new float[numKp, 256];

        for (int k = 0; k < numKp; k++)
        {
            int idx = candidates[k].idx;
            int dy = idx / descW;
            int dx = idx % descW;

            kpCoords[k * 2] = dx * 8.0f * origW / w;
            kpCoords[k * 2 + 1] = dy * 8.0f * origH / h;

            float norm = 0;
            for (int d = 0; d < 256; d++)
            {
                float val = descTensor[0, d, dy, dx];
                descriptors[k, d] = val;
                norm += val * val;
            }
            norm = MathF.Sqrt(norm);
            if (norm > 1e-6f)
                for (int d = 0; d < 256; d++)
                    descriptors[k, d] /= norm;
        }

        return (kpCoords, descriptors);
    }

    private static float[] NonMaxSuppression(float[] scores, int w, int h, int radius)
    {
        var result = new float[scores.Length];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float val = scores[y * w + x];
                if (val <= 0) continue;
                bool isMax = true;
                for (int dy = -radius; dy <= radius && isMax; dy++)
                {
                    for (int dx = -radius; dx <= radius && isMax; dx++)
                    {
                        if (dy == 0 && dx == 0) continue;
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < h && nx >= 0 && nx < w)
                        {
                            if (scores[ny * w + nx] > val)
                                isMax = false;
                        }
                    }
                }
                if (isMax) result[y * w + x] = val;
            }
        }
        return result;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
