using OpenCvSharp;

namespace DinoDuplicateSearch.CV;

public class SiftGeometricVerifier : IGeometricVerifier
{
    private readonly float _wgcThreshold;

    public string Name => "SIFT + WGC";

    public SiftGeometricVerifier(float wgcThreshold = 0.3f)
    {
        _wgcThreshold = wgcThreshold;
    }

    public (bool ok, float angle, float scale, int angleVotes, int scaleVotes) Verify(Mat image1, Mat image2)
    {
        var kp1 = GeometricConsistency.ExtractSiftFeaturesWithDescriptors(image1);
        var kp2 = GeometricConsistency.ExtractSiftFeaturesWithDescriptors(image2);

        if (kp1.keypoints.Length == 0 || kp2.keypoints.Length == 0 || kp1.descriptors == null || kp2.descriptors == null)
            return (false, 0, 0, 0, 0);

        var wgcResult = GeometricConsistency.CheckGeometricConsistency(
            kp1.keypoints, kp1.descriptors, kp2.keypoints, kp2.descriptors, _wgcThreshold);

        return (wgcResult.isValid, wgcResult.avgAngle, wgcResult.avgScale, wgcResult.angleVotes, wgcResult.scaleVotes);
    }
}
