using OpenCvSharp;
using OpenCvSharp.Features2D;

namespace DinoDuplicateSearch.CV;

public static class GeometricConsistency
{
    private const int MinAbsoluteVotes = 50;

    // ─── SuperPoint + LightGlue (new, preferred) ───────────────────────────

    public static (bool isValid, float avgAngle, float avgScale, int matchCount, float avgConfidence)
        CheckGeometricConsistencyLightGlue(
            float[] kpQuery, float[,] desQuery, float imageSizeQuery,
            float[] kpCandidate, float[,] desCandidate, float imageSizeCandidate,
            LightGlueMatcher matcher,
            float thresholdRatio = 0.3f)
    {
        if (desQuery.GetLength(0) < 2 || desCandidate.GetLength(0) < 2)
            return (false, 0, 0, 0, 0);

        int[][] matches;
        try
        {
            matches = matcher.Match(kpQuery, desQuery, imageSizeQuery, kpCandidate, desCandidate, imageSizeCandidate, 0.5f);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[LightGlue] Match failed: {ex.Message}");
            return (false, 0, 0, 0, 0);
        }

        if (matches.Length < 10)
        {
            System.Diagnostics.Debug.WriteLine($"[LightGlue] Only {matches.Length} matches, need 10+");
            return (false, 0, 0, matches.Length, 0);
        }

        var angles = new List<float>();
        var scales = new List<float>();
        float totalConfidence = 0;

        foreach (var m in matches)
        {
            int qi = m[0], ci = m[1];
            var ptQ = new Point2f(kpQuery[qi * 2], kpQuery[qi * 2 + 1]);
            var ptC = new Point2f(kpCandidate[ci * 2], kpCandidate[ci * 2 + 1]);

            var dx = ptC.X - ptQ.X;
            var dy = ptC.Y - ptQ.Y;
            var angle = (float)(Math.Atan2(dy, dx) * 180.0 / Math.PI + 360) % 360;
            angles.Add(angle);

            if (ptQ.X > 0 && ptC.X > 0)
            {
                var dist = MathF.Sqrt(dx * dx + dy * dy);
                scales.Add(dist > 0 ? 1.0f : 1.0f); // placeholder - LightGlue doesn't provide scale
            }
        }

        var histAngles = ComputeHistogram(angles, 24, 0, 360);
        var maxAngleVotes = histAngles.Max();

        var histScales = ComputeHistogram(scales, 20, 0, 2);
        var maxScaleVotes = histScales.Max();

        var threshold = matches.Length * thresholdRatio;
        var isValid = maxAngleVotes > threshold && maxAngleVotes >= MinAbsoluteVotes;

        float avgConf = matches.Length > 0 ? totalConfidence / matches.Length : 0;
        float avgAngle = 0;
        if (isValid)
        {
            var bestAngleBin = Array.IndexOf(histAngles, maxAngleVotes);
            var binWidthA = 360f / 24;
            avgAngle = (bestAngleBin * binWidthA + (bestAngleBin + 1) * binWidthA) / 2;
        }

        return (isValid, avgAngle, 1.0f, matches.Length, avgConf);
    }

    // ─── SIFT + BFMatcher (legacy, kept as fallback) ───────────────────────

    public static KeyPoint[] ExtractSiftFeatures(Mat image)
    {
        using var sift = SIFT.Create();
        var keypoints = sift.Detect(image);
        return keypoints ?? Array.Empty<KeyPoint>();
    }

    public static (KeyPoint[] keypoints, float[,]? descriptors) ExtractSiftFeaturesWithDescriptors(Mat image)
    {
        using var sift = SIFT.Create();
        var keypoints = sift.Detect(image);
        using var descriptors = new Mat();
        sift.Compute(image, ref keypoints, descriptors);
        float[,]? desArray = null;
        if (keypoints.Length > 0 && !descriptors.Empty())
        {
            desArray = new float[descriptors.Rows, descriptors.Cols];
            for (int r = 0; r < descriptors.Rows; r++)
                for (int c = 0; c < descriptors.Cols; c++)
                    desArray[r, c] = descriptors.At<float>(r, c);
        }
        return (keypoints ?? Array.Empty<KeyPoint>(), desArray);
    }

    public static (bool isValid, float avgAngle, float avgScale, int angleVotes, int scaleVotes)
        CheckGeometricConsistency(
            KeyPoint[] kpQuery, float[,]? desQuery,
            KeyPoint[] kpCandidate, float[,]? desCandidate,
            float thresholdRatio = 0.3f)
    {
        if (desQuery == null || desCandidate == null)
            return (false, 0, 0, 0, 0);

        if (desQuery.GetLength(0) < 2 || desCandidate.GetLength(0) < 2)
            return (false, 0, 0, 0, 0);

        using var desQueryMat = FloatArrayToMat(desQuery);
        using var desCandidateMat = FloatArrayToMat(desCandidate);

        DMatch[][]? matches;
        try
        {
            using var bf = new BFMatcher(NormTypes.L2, false);
            matches = bf.KnnMatch(desQueryMat, desCandidateMat, 2);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[WGC] KnnMatch failed: {ex.Message}");
            return (false, 0, 0, 0, 0);
        }

        var goodMatches = new List<DMatch>();
        if (matches != null)
        {
            foreach (var match in matches)
            {
                if (match.Length >= 2 && match[0].Distance < 0.75f * match[1].Distance)
                    goodMatches.Add(match[0]);
            }
        }

        if (goodMatches.Count < 10)
            return (false, 0, 0, 0, 0);

        var angles = new List<float>();
        var scales = new List<float>();

        foreach (var m in goodMatches)
        {
            var ptQ = kpQuery[m.QueryIdx];
            var ptC = kpCandidate[m.TrainIdx];

            var angleDiff = (ptQ.Angle - ptC.Angle) % 360;
            angles.Add(angleDiff);

            if (ptC.Size > 0)
                scales.Add(ptQ.Size / ptC.Size);
        }

        var histAngles = ComputeHistogram(angles, 24, 0, 360);
        var maxAngleVotes = histAngles.Max();

        var logScales = scales.Select(s => (float)Math.Log2(s)).ToList();
        var histScales = ComputeHistogram(logScales, 20, -3, 3);
        var maxScaleVotes = histScales.Max();

        var threshold = goodMatches.Count * thresholdRatio;
        var isValid = maxAngleVotes > threshold && maxScaleVotes > threshold
                   && maxAngleVotes >= MinAbsoluteVotes && maxScaleVotes >= MinAbsoluteVotes;

        if (isValid)
        {
            var bestAngleBin = Array.IndexOf(histAngles, maxAngleVotes);
            var binWidthA = 360f / 24;
            var avgAngle = (bestAngleBin * binWidthA + (bestAngleBin + 1) * binWidthA) / 2;

            var bestScaleBin = Array.IndexOf(histScales, maxScaleVotes);
            var binWidthS = 6f / 20;
            var avgScale = (float)Math.Pow(2, (bestScaleBin * binWidthS - 3 + (bestScaleBin + 1) * binWidthS - 3) / 2);

            return (true, avgAngle, avgScale, maxAngleVotes, maxScaleVotes);
        }

        return (false, 0, 0, 0, 0);
    }

    public static int[] ComputeHistogramPublic(List<float> values, int bins, float min, float max) => ComputeHistogram(values, bins, min, max);

    private static int[] ComputeHistogram(List<float> values, int bins, float min, float max)
    {
        var hist = new int[bins];
        var binWidth = (max - min) / bins;
        foreach (var v in values)
        {
            var idx = (int)((v - min) / binWidth);
            idx = Math.Clamp(idx, 0, bins - 1);
            hist[idx]++;
        }
        return hist;
    }

    private static Mat FloatArrayToMat(float[,] array)
    {
        var rows = array.GetLength(0);
        var cols = array.GetLength(1);
        var mat = new Mat(rows, cols, MatType.CV_32F);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat.Set(r, c, array[r, c]);
        return mat;
    }
}
