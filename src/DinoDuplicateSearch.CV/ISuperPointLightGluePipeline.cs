using OpenCvSharp;

namespace DinoDuplicateSearch.CV;

public interface ISuperPointLightGluePipeline : IDisposable
{
    (int[][] matches, float[] scores, float[,] keypoints0, float[,] keypoints1) Match(Mat image0, Mat image1);
}
