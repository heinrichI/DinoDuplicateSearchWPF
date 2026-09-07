using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace DinoDuplicateSearch.CV;

/// <summary>
/// Fused SuperPoint + LightGlue pipeline from fabio-sim/LightGrue-ONNX.
/// Input: interleaved grayscale images (batch, 1, H, W).
/// Output: keypoints (batch, 1024, 2), matches (M, 3), mscores (M,).
/// </summary>
public class SuperPointLightGluePipeline : ISuperPointLightGluePipeline
{
    private readonly string _modelPath;
    private readonly object _sessionGate = new object();
    private InferenceSession? _session;
    private readonly int _maxKeypoints;
    private readonly int _inputHeight;
    private readonly int _inputWidth;

    public SuperPointLightGluePipeline(
        string modelPath = "Models/superpoint_lightglue_pipeline.onnx",
        int maxKeypoints = 1024,
        int inputHeight = 600,
        int inputWidth = 800)
    {
        _modelPath = modelPath;
        _maxKeypoints = maxKeypoints;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
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

        var inputName = _session.InputMetadata.Keys.First();
        var outputNames = _session.OutputMetadata.Keys.ToList();
        System.Diagnostics.Debug.WriteLine($"[Pipeline] Input: {inputName}, Outputs: {string.Join(", ", outputNames)}");
    }

    /// <summary>
    /// Match two images. Returns matches as (kpIdx0, kpIdx1) pairs with confidence scores.
    /// ONNX Runtime InferenceSession is not thread-safe for concurrent Run() calls,
    /// which crashes the CUDA provider (misaligned address / CUBLAS errors) when the
    /// shared session is invoked from multiple WGC worker threads. Every call is
    /// therefore serialized through _sessionGate, which also guards lazy model loading.
    /// </summary>
    public (int[][] matches, float[] scores, float[,] keypoints0, float[,] keypoints1) Match(
        Mat image0, Mat image1)
    {
        lock (_sessionGate)
        {
            LoadModel();

            using var gray0 = new Mat();
            using var gray1 = new Mat();
            Cv2.CvtColor(image0, gray0, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(image1, gray1, ColorConversionCodes.BGR2GRAY);

            float scale0 = Math.Min(1.0f, Math.Min((float)_inputWidth / gray0.Cols, (float)_inputHeight / gray0.Rows));
            float scale1 = Math.Min(1.0f, Math.Min((float)_inputWidth / gray1.Cols, (float)_inputHeight / gray1.Rows));

            int h0 = (int)(gray0.Rows * scale0 / 8) * 8;
            int w0 = (int)(gray0.Cols * scale0 / 8) * 8;
            int h1 = (int)(gray1.Rows * scale1 / 8) * 8;
            int w1 = (int)(gray1.Cols * scale1 / 8) * 8;
            if (h0 < 8) h0 = 8; if (w0 < 8) w0 = 8;
            if (h1 < 8) h1 = 8; if (w1 < 8) w1 = 8;

            using var resized0 = new Mat();
            using var resized1 = new Mat();
            Cv2.Resize(gray0, resized0, new Size(w0, h0));
            Cv2.Resize(gray1, resized1, new Size(w1, h1));

            // Pad both to same size for batched input
            int maxH = Math.Max(h0, h1);
            int maxW = Math.Max(w0, w1);

            var inputTensor = new DenseTensor<float>(new[] { 2, 1, maxH, maxW });
            for (int y = 0; y < h0; y++)
                for (int x = 0; x < w0; x++)
                    inputTensor[0, 0, y, x] = resized0.At<byte>(y, x) / 255f;
            for (int y = 0; y < h1; y++)
                for (int x = 0; x < w1; x++)
                    inputTensor[1, 0, y, x] = resized1.At<byte>(y, x) / 255f;

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            using var results = _session!.Run(inputs);

            var kpsTensor = results.First(r => r.Name == "keypoints").AsTensor<long>();
            var matchTensor = results.First(r => r.Name == "matches").AsTensor<long>();
            var scoreTensor = results.First(r => r.Name == "mscores").AsTensor<float>();

            // Extract keypoints for each image, scale back to original coordinates
            int numKps = kpsTensor.Dimensions[1];
            var kp0 = new float[numKps, 2];
            var kp1 = new float[numKps, 2];
            for (int k = 0; k < numKps; k++)
            {
                kp0[k, 0] = kpsTensor[0, k, 0] / scale0;
                kp0[k, 1] = kpsTensor[0, k, 1] / scale0;
                kp1[k, 0] = kpsTensor[1, k, 0] / scale1;
                kp1[k, 1] = kpsTensor[1, k, 1] / scale1;
            }

            // Extract matches
            int numMatches = matchTensor.Dimensions[0];
            var matches = new List<int[]>();
            var scores = new List<float>();
            for (int m = 0; m < numMatches; m++)
            {
                long kpIdx0 = matchTensor[m, 1];
                long kpIdx1 = matchTensor[m, 2];
                float score = scoreTensor[m];
                if (kpIdx0 >= 0 && kpIdx1 >= 0 && score > 0.1f)
                {
                    matches.Add(new[] { (int)kpIdx0, (int)kpIdx1 });
                    scores.Add(score);
                }
            }

            return (matches.ToArray(), scores.ToArray(), kp0, kp1);
        }
    }

    public void Dispose()
    {
        lock (_sessionGate)
        {
            _session?.Dispose();
        }
    }
}