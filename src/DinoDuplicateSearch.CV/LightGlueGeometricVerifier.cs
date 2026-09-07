using OpenCvSharp;

namespace DinoDuplicateSearch.CV;

public class LightGlueGeometricVerifier : IGeometricVerifier
{
    private readonly ISuperPointLightGluePipeline _pipeline;
    private readonly float _minMatchScore;

    public string Name => "SuperPoint + LightGlue";

    public LightGlueGeometricVerifier(ISuperPointLightGluePipeline pipeline, float minMatchScore = 0.5f)
    {
        _pipeline = pipeline;
        _minMatchScore = minMatchScore;
    }

    public (bool ok, float angle, float scale, int angleVotes, int scaleVotes) Verify(Mat image1, Mat image2)
    {
        var (matches, scores, kp0, kp1) = _pipeline.Match(image1, image2);

        if (matches.Length < 5)
            return (false, 0, 0, matches.Length, 0);

        var angles = new List<float>();
        foreach (var m in matches)
        {
            float dx = kp1[m[1], 0] - kp0[m[0], 0];
            float dy = kp1[m[1], 1] - kp0[m[0], 1];
            var matchAngle = (float)(Math.Atan2(dy, dx) * 180.0 / Math.PI + 360) % 360;
            angles.Add(matchAngle);
        }

        var histAngles = GeometricConsistency.ComputeHistogramPublic(angles, 24, 0, 360);
        int maxAngleVotes = histAngles.Max();
        float avgAngle = 0;
        if (maxAngleVotes > 0)
        {
            int bestBin = Array.IndexOf(histAngles, maxAngleVotes);
            float binWidth = 360f / 24;
            avgAngle = (bestBin * binWidth + (bestBin + 1) * binWidth) / 2;
        }

        float avgScore = scores.Length > 0 ? scores.Average() : 0;
        bool passed = matches.Length >= 10 && maxAngleVotes >= 5 && avgScore >= _minMatchScore;

        return (passed, avgAngle, 1.0f, matches.Length, (int)(avgScore * 100));
    }
}
