namespace DinoDuplicateSearch.ML;

public static class AgglomerativeClustering
{
    public static int[] FitPredict(float[,] embeddings, float distanceThreshold)
    {
        var n = embeddings.GetLength(0);
        var dim = embeddings.GetLength(1);

        var dist = new float[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                float dot = 0;
                for (int d = 0; d < dim; d++)
                    dot += embeddings[i, d] * embeddings[j, d];
                dist[i, j] = 1 - dot;
                dist[j, i] = dist[i, j];
            }
        }

        var labels = Enumerable.Range(0, n).ToArray();
        var clusterSize = Enumerable.Repeat(1, n).ToArray();
        var active = Enumerable.Repeat(true, n).ToArray();

        while (true)
        {
            float minDist = float.MaxValue;
            int minI = -1, minJ = -1;

            for (int i = 0; i < n; i++)
            {
                if (!active[i]) continue;
                for (int j = i + 1; j < n; j++)
                {
                    if (!active[j]) continue;
                    if (dist[i, j] < minDist)
                    {
                        minDist = dist[i, j];
                        minI = i;
                        minJ = j;
                    }
                }
            }

            if (minI == -1 || minDist >= distanceThreshold)
                break;

            for (int k = 0; k < n; k++)
            {
                if (!active[k] || k == minI || k == minJ) continue;
                float newDist = (dist[minI, k] * clusterSize[minI] + dist[minJ, k] * clusterSize[minJ])
                               / (clusterSize[minI] + clusterSize[minJ]);
                dist[minI, k] = newDist;
                dist[k, minI] = newDist;
            }

            var oldLabel = labels[minJ];
            var newLabel = labels[minI];
            for (int k = 0; k < n; k++)
            {
                if (labels[k] == oldLabel)
                    labels[k] = newLabel;
            }

            clusterSize[minI] += clusterSize[minJ];
            active[minJ] = false;
        }

        var uniqueLabels = labels.Distinct().OrderBy(x => x).ToArray();
        var labelMap = new Dictionary<int, int>();
        for (int i = 0; i < uniqueLabels.Length; i++)
            labelMap[uniqueLabels[i]] = i;

        for (int i = 0; i < n; i++)
            labels[i] = labelMap[labels[i]];

        return labels;
    }
}
