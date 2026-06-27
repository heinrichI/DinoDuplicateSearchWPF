using System.Windows.Threading;

namespace DinoDuplicateSearch.Core;

public class ThrottledProgress<T> : IProgress<T>, IDisposable
{
    private readonly Action<T> _handler;
    private readonly TimeSpan _interval;
    private readonly DispatcherTimer _timer;
    private T? _pending;
    private bool _disposed;

    public ThrottledProgress(Action<T> handler, TimeSpan interval)
    {
        _handler = handler;
        _interval = interval;
        _timer = new DispatcherTimer { Interval = interval };
        _timer.Tick += (_, _) =>
        {
            _timer.Stop();
            if (_pending is { } value)
            {
                _pending = default;
                _handler(value);
            }
        };
    }

    public void Report(T value)
    {
        if (_disposed) return;
        _pending = value;
        if (!_timer.IsEnabled)
            _timer.Start();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _timer.Stop();
    }
}
