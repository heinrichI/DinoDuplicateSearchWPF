using System.IO;

namespace DinoDuplicateSearch.Models;

public static class DebugLog
{
    private static readonly object _lock = new();
    private static readonly string _path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "debug.log");

    public static void Write(string msg)
    {
        lock (_lock)
        {
            File.AppendAllText(_path, $"[{DateTime.Now:HH:mm:ss.fff}] {msg}\n");
            System.Diagnostics.Debug.WriteLine(msg);
        }
    }

    public static void Clear()
    {
        lock (_lock) { try { File.WriteAllText(_path, ""); } catch { } }
    }
}
