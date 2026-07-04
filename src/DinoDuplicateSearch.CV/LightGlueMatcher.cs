using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DinoDuplicateSearch.CV;

public class LightGlueMatcher : IDisposable
{
    private readonly string _modelPath;
    private InferenceSession? _session;

    public LightGlueMatcher(string modelPath = "Models/lightglue.onnx")
    {
        _modelPath = modelPath;
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
    }

    public int[][] Match(
        float[] kp0, float[,] desc0, float imageSize0,
        float[] kp1, float[,] desc1, float imageSize1,
        float confidenceThreshold = 0.5f)
    {
        LoadModel();

        int n0 = desc0.GetLength(0);
        int n1 = desc1.GetLength(0);

        if (n0 == 0 || n1 == 0)
            return Array.Empty<int[]>();

        // Build input tensors
        var kpts0Tensor = new DenseTensor<float>(new[] { 1, n0, 2 });
        var kpts1Tensor = new DenseTensor<float>(new[] { 1, n1, 2 });
        var desc0Tensor = new DenseTensor<float>(new[] { 1, n0, 256 });
        var desc1Tensor = new DenseTensor<float>(new[] { 1, n1, 256 });
        var size0Tensor = new DenseTensor<float>(new[] { 1, 2 });
        var size1Tensor = new DenseTensor<float>(new[] { 1, 2 });

        size0Tensor[0, 0] = imageSize0;
        size0Tensor[0, 1] = imageSize0;
        size1Tensor[0, 0] = imageSize1;
        size1Tensor[0, 1] = imageSize1;

        for (int i = 0; i < n0; i++)
        {
            kpts0Tensor[0, i, 0] = kp0[i * 2];
            kpts0Tensor[0, i, 1] = kp0[i * 2 + 1];
            for (int d = 0; d < 256; d++)
                desc0Tensor[0, i, d] = desc0[i, d];
        }

        for (int i = 0; i < n1; i++)
        {
            kpts1Tensor[0, i, 0] = kp1[i * 2];
            kpts1Tensor[0, i, 1] = kp1[i * 2 + 1];
            for (int d = 0; d < 256; d++)
                desc1Tensor[0, i, d] = desc1[i, d];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("kpts0", kpts0Tensor),
            NamedOnnxValue.CreateFromTensor("kpts1", kpts1Tensor),
            NamedOnnxValue.CreateFromTensor("desc0", desc0Tensor),
            NamedOnnxValue.CreateFromTensor("desc1", desc1Tensor),
            NamedOnnxValue.CreateFromTensor("image_size0", size0Tensor),
            NamedOnnxValue.CreateFromTensor("image_size1", size1Tensor)
        };

        using var results = _session!.Run(inputs);

        // LightGlue outputs: matches (1, max(n0,n1), 2) and scores (1, max(n0,n1))
        var matchTensor = results.First(r => r.Name == "matches").AsTensor<int>();
        var scoreTensor = results.First(r => r.Name == "scores").AsTensor<float>();

        var matchDims = matchTensor.Dimensions;
        int outputLen = matchDims[1];
        var matches = new List<int[]>();

        for (int i = 0; i < Math.Min(outputLen, n0); i++)
        {
            int j = matchTensor[0, i, 1];
            float score = scoreTensor[0, i];

            if (j >= 0 && j < n1 && score >= confidenceThreshold)
            {
                matches.Add(new[] { i, j });
            }
        }

        return matches.ToArray();
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
