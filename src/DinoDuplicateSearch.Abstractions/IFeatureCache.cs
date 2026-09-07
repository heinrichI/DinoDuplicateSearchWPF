namespace DinoDuplicateSearch.Abstractions;

public interface IFeatureCache : IDisposable
{
    (double mtime, float[] embedding)? GetEmbedding(string path);
    void SetEmbedding(string path, double mtime, float[] embedding);
    (double mtime, float[] keypoints, float[,] descriptors)? GetSift(string path);
    void SetSift(string path, double mtime, float[] keypoints, float[,] descriptors);
    (bool result, float angle, float scale, int angleVotes, int scaleVotes)? GetWgc(string path1, string path2, double mtime1, double mtime2);
    void SetWgc(string path1, string path2, double mtime1, double mtime2, bool result, float angle, float scale, int angleVotes, int scaleVotes);
    long ClearAll();
    long GetEmbeddingsHash();
}
