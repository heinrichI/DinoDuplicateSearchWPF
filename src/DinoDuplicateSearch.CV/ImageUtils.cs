using System.IO;
using System.Text;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace DinoDuplicateSearch.CV;

public static class ImageUtils
{
    public static Mat ReadImageCv2(string path)
    {
        var isAscii = true;
        try { Encoding.ASCII.GetBytes(path); }
        catch { isAscii = false; }

        if (isAscii)
        {
            var img = Cv2.ImRead(path);
            if (!img.Empty()) return img;
        }

        try
        {
            using var stream = File.OpenRead(path);
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = stream;
            bitmap.CreateOptions = BitmapCreateOptions.IgnoreColorProfile;
            bitmap.EndInit();
            bitmap.Freeze();
            var mat = BitmapImageToMat(bitmap);
            Cv2.CvtColor(mat, mat, ColorConversionCodes.RGB2BGR);
            return mat;
        }
        catch { }

        return new Mat();
    }

    private static Mat BitmapImageToMat(BitmapImage bitmap)
    {
        var stride = bitmap.PixelWidth * 4;
        var pixels = new byte[stride * bitmap.PixelHeight];
        bitmap.CopyPixels(pixels, stride, 0);
        var mat = new Mat(bitmap.PixelHeight, bitmap.PixelWidth, MatType.CV_8UC4);
        Marshal.Copy(pixels, 0, mat.Data, pixels.Length);
        return mat;
    }
}
