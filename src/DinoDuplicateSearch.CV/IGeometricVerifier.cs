using OpenCvSharp;

namespace DinoDuplicateSearch.CV;

public interface IGeometricVerifier
{
    string Name { get; }
    (bool ok, float angle, float scale, int angleVotes, int scaleVotes) Verify(Mat image1, Mat image2);
}
