namespace DinoDuplicateSearch.Models;

public class UnionFind
{
    private readonly Dictionary<string, string> _parent = new();
    private readonly Dictionary<string, int> _rank = new();

    public string Find(string x)
    {
        if (!_parent.ContainsKey(x))
        {
            _parent[x] = x;
            _rank[x] = 0;
            return x;
        }
        if (_parent[x] != x)
            _parent[x] = Find(_parent[x]);
        return _parent[x];
    }

    public void Union(string x, string y)
    {
        var rootX = Find(x);
        var rootY = Find(y);
        if (rootX == rootY) return;

        if (_rank[rootX] < _rank[rootY])
            _parent[rootX] = rootY;
        else if (_rank[rootX] > _rank[rootY])
            _parent[rootY] = rootX;
        else
        {
            _parent[rootY] = rootX;
            _rank[rootX]++;
        }
    }

    public Dictionary<string, List<string>> GetGroups()
    {
        var groups = new Dictionary<string, List<string>>();
        foreach (var item in _parent.Keys)
        {
            var root = Find(item);
            if (!groups.ContainsKey(root))
                groups[root] = new List<string>();
            groups[root].Add(item);
        }
        return groups;
    }
}
