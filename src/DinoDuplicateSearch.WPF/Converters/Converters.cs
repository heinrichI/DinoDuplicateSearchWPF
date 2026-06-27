using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DinoDuplicateSearch.Views;

public class TabIndexToVisibilityConverter : IValueConverter
{
    public static readonly TabIndexToVisibilityConverter Instance = new();
    public static readonly TabIndexToVisibilityConverter Search = new(0);
    public static readonly TabIndexToVisibilityConverter Result = new(1);

    private readonly int? _targetIndex;

    public TabIndexToVisibilityConverter() { }
    public TabIndexToVisibilityConverter(int targetIndex) { _targetIndex = targetIndex; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var idx = (int)value;
        if (_targetIndex.HasValue)
            return idx == _targetIndex.Value ? Visibility.Visible : Visibility.Collapsed;
        return Visibility.Visible;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

public class TabIndexToColorConverter : IValueConverter
{
    public static readonly TabIndexToColorConverter Search = new(0);
    public static readonly TabIndexToColorConverter Result = new(1);

    private readonly int _targetIndex;

    public TabIndexToColorConverter(int targetIndex) { _targetIndex = targetIndex; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var idx = (int)value;
        return idx == _targetIndex
            ? new SolidColorBrush(Color.FromRgb(0x33, 0x99, 0xFF))
            : new SolidColorBrush(Color.FromRgb(0xE8, 0xE8, 0xE8));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

public class BoolToVisibilityConverter : IValueConverter
{
    public static readonly BoolToVisibilityConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => (bool)value ? Visibility.Visible : Visibility.Collapsed;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

public class StringToUriConverter : IValueConverter
{
    public static readonly StringToUriConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string path && !string.IsNullOrEmpty(path))
        {
            try { return new Uri(path, UriKind.Absolute); }
            catch { }
        }
        return null!;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

public class StringToBitmapImageConverter : IValueConverter
{
    public static readonly StringToBitmapImageConverter Instance = new();

    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string path && !string.IsNullOrEmpty(path))
        {
            try
            {
                var uri = new Uri(path, UriKind.Absolute);
                var bitmap = new BitmapImage(uri);
                bitmap.Freeze();
                return bitmap;
            }
            catch { }
        }
        return null;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
