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
        private bool applyFilters;
        private YCbCr color;
        private YCbCrFiltering colorFilter;
        private GrayscaleRMY grayFilter;
        private Threshold binaryFilter;
        private BinaryErosion3x3 erosionFilter;
        private GrayscaleToRGB rgbFilter;

        public ColorFilter()
        {
            this.applyFilters = false;
            this.colorFilter = new YCbCrFiltering();
            this.grayFilter = new GrayscaleRMY();
            this.binaryFilter = new Threshold(1);
            this.erosionFilter = new BinaryErosion3x3();
            this.rgbFilter = new GrayscaleToRGB();
        }

        public void SetColor(byte r, byte g, byte b)
        {
            RGB rgbcolor = new RGB(r,g,b);
            this.color = YCbCr.FromRGB(rgbcolor);
            this.colorFilter.Cb = new AForge.Range(this.color.Cb - 0.02f, this.color.Cb + 0.02f);
            this.colorFilter.Cr = new AForge.Range(this.color.Cr - 0.02f, this.color.Cr + 0.02f);
            this.colorFilter.Y = new AForge.Range(this.color.Y - 0.05f, this.color.Y + 0.05f);
            this.applyFilters = true;
        }

        public IntPtr FilterFrame(byte[] pixels,int width,int height)
        {
            
            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bmData = bmp.LockBits(new System.Drawing.Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, bmp.PixelFormat);
            IntPtr pNative = bmData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(pixels, 0, pNative, pixels.Length);
            bmp.UnlockBits(bmData);

            if (this.applyFilters)
            {
                this.colorFilter.ApplyInPlace(bmp);
                Bitmap graybmp = this.grayFilter.Apply(bmp);
                this.binaryFilter.ApplyInPlace(graybmp);
                this.erosionFilter.ApplyInPlace(graybmp);
                bmp = this.rgbFilter.Apply(graybmp).Clone(new Rectangle(0, 0, width, height), bmp.PixelFormat);

                bmData = bmp.LockBits(new System.Drawing.Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, bmp.PixelFormat);
                pNative = bmData.Scan0;
                bmp.UnlockBits(bmData);
            }

            return pNative;
        }

    }
}
