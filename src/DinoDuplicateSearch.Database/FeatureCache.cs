using System.Data.SQLite;
using System.IO.Compression;

namespace DinoDuplicateSearch.Database;

public class FeatureCache : IDisposable
{
    private readonly SQLiteConnection _conn;

    public FeatureCache(string dbPath = "feature_cache.db")
    {
        _conn = new SQLiteConnection($"Data Source={dbPath};Version=3;");
        _conn.Open();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "PRAGMA journal_mode=WAL";
        cmd.ExecuteNonQuery();
        cmd.CommandText = "PRAGMA synchronous=NORMAL";
        cmd.ExecuteNonQuery();
        cmd.CommandText = "PRAGMA busy_timeout=5000";
        cmd.ExecuteNonQuery();
        cmd.CommandText = "PRAGMA auto_vacuum=INCREMENTAL";
        cmd.ExecuteNonQuery();

        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS embeddings (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )";
        cmd.ExecuteNonQuery();

        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS sift (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                keypoints BLOB,
                keypoints_count INTEGER DEFAULT 0,
                descriptors BLOB,
                descriptors_shape TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )";
        cmd.ExecuteNonQuery();

        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS wgc_results (
                path1 TEXT NOT NULL,
                path2 TEXT NOT NULL,
                mtime1 REAL NOT NULL,
                mtime2 REAL NOT NULL,
                result BOOLEAN NOT NULL,
                angle REAL DEFAULT 0,
                scale REAL DEFAULT 0,
                angle_votes INTEGER DEFAULT 0,
                scale_votes INTEGER DEFAULT 0,
                PRIMARY KEY (path1, path2)
            )";
        cmd.ExecuteNonQuery();
    }

    public (double mtime, float[] embedding)? GetEmbedding(string path)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT mtime, embedding FROM embeddings WHERE path = @p";
        cmd.Parameters.AddWithValue("@p", path);
        using var reader = cmd.ExecuteReader();
        if (!reader.Read()) return null;

        var mtime = reader.GetDouble(0);
        var blob = (byte[])reader["embedding"];
        var decompressed = ZlibDecompress(blob);
        var embedding = new float[decompressed.Length / 4];
        Buffer.BlockCopy(decompressed, 0, embedding, 0, decompressed.Length);
        return (mtime, embedding);
    }

    public void SetEmbedding(string path, double mtime, float[] embedding)
    {
        var bytes = new byte[embedding.Length * 4];
        Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);
        var blob = ZlibCompress(bytes);

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "INSERT OR REPLACE INTO embeddings (path, mtime, embedding) VALUES (@p, @m, @e)";
        cmd.Parameters.AddWithValue("@p", path);
        cmd.Parameters.AddWithValue("@m", mtime);
        cmd.Parameters.AddWithValue("@e", blob);
        cmd.ExecuteNonQuery();
    }

    public (double mtime, float[] keypoints, float[,] descriptors)? GetSift(string path)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT mtime, keypoints, keypoints_count, descriptors, descriptors_shape FROM sift WHERE path = @p";
        cmd.Parameters.AddWithValue("@p", path);
        using var reader = cmd.ExecuteReader();
        if (!reader.Read()) return null;

        var mtime = reader.GetDouble(0);
        var kpBlob = (byte[])reader["keypoints"];
        var keypoints = BytesToKeyPoints(ZlibDecompress(kpBlob));

        float[,]? descriptors = null;
        if (!reader.IsDBNull(3))
        {
            var desBlob = (byte[])reader["descriptors"];
            var shapeStr = (string)reader["descriptors_shape"];
            descriptors = BytesToDescriptors(ZlibDecompress(desBlob), shapeStr);
        }

        return (mtime, keypoints, descriptors ?? new float[0, 0]);
    }

    public void SetSift(string path, double mtime, float[] keypoints, float[,] descriptors)
    {
        var kpBlob = ZlibCompress(KeyPointsToBytes(keypoints));
        byte[]? desBlob = null;
        string desShape = "0,0";
        if (descriptors.GetLength(0) > 0)
        {
            desBlob = ZlibCompress(DescriptorsToBytes(descriptors));
            desShape = $"{descriptors.GetLength(0)},{descriptors.GetLength(1)}";
        }

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"INSERT OR REPLACE INTO sift
            (path, mtime, keypoints, keypoints_count, descriptors, descriptors_shape)
            VALUES (@p, @m, @k, @c, @d, @s)";
        cmd.Parameters.AddWithValue("@p", path);
        cmd.Parameters.AddWithValue("@m", mtime);
        cmd.Parameters.AddWithValue("@k", kpBlob);
        cmd.Parameters.AddWithValue("@c", keypoints.Length);
        cmd.Parameters.AddWithValue("@d", (object?)desBlob ?? DBNull.Value);
        cmd.Parameters.AddWithValue("@s", desShape);
        cmd.ExecuteNonQuery();
    }

    public (bool result, float angle, float scale, int angleVotes, int scaleVotes)? GetWgc(string path1, string path2, double mtime1, double mtime2)
    {
        var (p1, p2) = string.Compare(path1, path2) < 0 ? (path1, path2) : (path2, path1);
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT result, angle, scale, angle_votes, scale_votes, mtime1, mtime2 FROM wgc_results WHERE path1=@p1 AND path2=@p2";
        cmd.Parameters.AddWithValue("@p1", p1);
        cmd.Parameters.AddWithValue("@p2", p2);
        using var reader = cmd.ExecuteReader();
        if (!reader.Read()) return null;

        var cachedMtime1 = reader.GetDouble(5);
        var cachedMtime2 = reader.GetDouble(6);
        if (Math.Abs(cachedMtime1 - mtime1) > 0.01 || Math.Abs(cachedMtime2 - mtime2) > 0.01)
            return null;

        return (reader.GetBoolean(0), (float)reader.GetDouble(1), (float)reader.GetDouble(2), reader.GetInt32(3), reader.GetInt32(4));
    }

    public void SetWgc(string path1, string path2, double mtime1, double mtime2, bool result, float angle, float scale, int angleVotes, int scaleVotes)
    {
        bool swapped = string.Compare(path1, path2) >= 0;
        var fp1 = swapped ? path2 : path1;
        var fp2 = swapped ? path1 : path2;
        var sm1 = swapped ? mtime2 : mtime1;
        var sm2 = swapped ? mtime1 : mtime2;

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"INSERT OR REPLACE INTO wgc_results
            (path1, path2, mtime1, mtime2, result, angle, scale, angle_votes, scale_votes)
            VALUES (@p1, @p2, @m1, @m2, @r, @a, @s, @av, @sv)";
        cmd.Parameters.AddWithValue("@p1", fp1);
        cmd.Parameters.AddWithValue("@p2", fp2);
        cmd.Parameters.AddWithValue("@m1", sm1);
        cmd.Parameters.AddWithValue("@m2", sm2);
        cmd.Parameters.AddWithValue("@r", result);
        cmd.Parameters.AddWithValue("@a", angle);
        cmd.Parameters.AddWithValue("@s", scale);
        cmd.Parameters.AddWithValue("@av", angleVotes);
        cmd.Parameters.AddWithValue("@sv", scaleVotes);
        cmd.ExecuteNonQuery();
    }

    public long ClearAll()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "PRAGMA page_count";
        var beforePages = (long)cmd.ExecuteScalar()!;
        cmd.CommandText = "PRAGMA page_size";
        var pageSize = (long)cmd.ExecuteScalar()!;

        cmd.CommandText = "DELETE FROM embeddings";
        cmd.ExecuteNonQuery();
        cmd.CommandText = "DELETE FROM sift";
        cmd.ExecuteNonQuery();
        cmd.CommandText = "DELETE FROM wgc_results";
        cmd.ExecuteNonQuery();
        cmd.CommandText = "VACUUM";
        cmd.ExecuteNonQuery();

        cmd.CommandText = "PRAGMA page_count";
        var afterPages = (long)cmd.ExecuteScalar()!;
        return (beforePages - afterPages) * pageSize;
    }

    public long GetEmbeddingsHash()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM embeddings";
        int count = Convert.ToInt32(cmd.ExecuteScalar());
        return count;
    }

    public void Dispose()
    {
        _conn?.Close();
        _conn?.Dispose();
    }

    private static byte[] ZlibCompress(byte[] data)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Optimal))
            zlib.Write(data, 0, data.Length);
        return output.ToArray();
    }

    private static byte[] ZlibDecompress(byte[] data)
    {
        using var input = new MemoryStream(data);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);
        return output.ToArray();
    }

    private static byte[] KeyPointsToBytes(float[] keypoints)
    {
        var bytes = new byte[keypoints.Length * 4];
        Buffer.BlockCopy(keypoints, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] BytesToKeyPoints(byte[] data)
    {
        if (data.Length == 0) return Array.Empty<float>();
        var result = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, result, 0, data.Length);
        return result;
    }

    private static byte[] DescriptorsToBytes(float[,] descriptors)
    {
        var bytes = new byte[descriptors.Length * 4];
        Buffer.BlockCopy(descriptors, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[,] BytesToDescriptors(byte[] data, string shapeStr)
    {
        var parts = shapeStr.Split(',');
        var rows = int.Parse(parts[0]);
        var cols = int.Parse(parts[1]);
        var result = new float[rows, cols];
        Buffer.BlockCopy(data, 0, result, 0, data.Length);
        return result;
    }
}
