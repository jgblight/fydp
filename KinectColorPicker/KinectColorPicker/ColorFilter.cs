using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;

using AForge.Imaging;
using AForge.Imaging.Filters;

namespace KinectColorPicker
{
    class ColorFilter
    {
        private YCbCr color;
        private YCbCrFiltering aForgeFilter;

        public ColorFilter()
        {
            this.aForgeFilter = new YCbCrFiltering();
        }

        public void SetColor(byte r, byte g, byte b)
        {
            RGB rgbcolor = new RGB(r,g,b);
            this.color = YCbCr.FromRGB(rgbcolor);
            this.aForgeFilter.Cb = new AForge.Range(this.color.Cb - 0.02f, this.color.Cb + 0.02f);
            this.aForgeFilter.Cr = new AForge.Range(this.color.Cr - 0.02f, this.color.Cr + 0.02f);
            this.aForgeFilter.Y = new AForge.Range(this.color.Y - 0.1f, this.color.Y + 0.1f);

        }

        public IntPtr FilterFrame(byte[] pixels,int width,int height)
        {
            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bmData = bmp.LockBits(new System.Drawing.Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, bmp.PixelFormat);
            IntPtr pNative = bmData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(pixels, 0, pNative, pixels.Length);
            bmp.UnlockBits(bmData);

            this.aForgeFilter.ApplyInPlace(bmp);

            return pNative;
        }

    }
}
