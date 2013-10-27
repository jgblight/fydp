using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using AForge.Imaging;
using AForge.Imaging.Filters;

using Microsoft.Kinect;

namespace KinectColorPicker
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        /// <summary>
        /// Active Kinect sensor
        /// </summary>
        private KinectSensor sensor;

        /// <summary>
        /// Bitmap that will hold color information
        /// </summary>
        private WriteableBitmap colorBitmap;

        /// <summary>
        /// Intermediate storage for the color data received from the camera
        /// </summary>
        private byte[] colorPixels;

        private YCbCrFiltering colorFilter;

        private bool processingFrame;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            // Look through all sensors and start the first connected one.
            // This requires that a Kinect is connected at the time of app startup.
            // To make your app robust against plug/unplug, 
            // it is recommended to use KinectSensorChooser provided in Microsoft.Kinect.Toolkit (See components in Toolkit Browser).
            foreach (var potentialSensor in KinectSensor.KinectSensors)
            {
                if (potentialSensor.Status == KinectStatus.Connected)
                {
                    this.sensor = potentialSensor;
                    break;
                }
            }

            if (null != this.sensor)
            {
                // Turn on the color stream to receive color frames
                this.sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);

                // Allocate space to put the pixels we'll receive
                this.colorPixels = new byte[this.sensor.ColorStream.FramePixelDataLength];

                // This is the bitmap we'll display on-screen
                this.colorBitmap = new WriteableBitmap(this.sensor.ColorStream.FrameWidth, this.sensor.ColorStream.FrameHeight, 96.0, 96.0, PixelFormats.Bgr32, null);

               
                // Set the image we display to point to the bitmap where we'll put the image data
                this.Image.Source = this.colorBitmap;

                // Add an event handler to be called whenever there is new color frame data
                this.sensor.ColorFrameReady += this.SensorColorFrameReady;

                this.colorFilter = new YCbCrFiltering();

                // Start the sensor!
                try
                {
                    this.sensor.Start();
                }
                catch (IOException)
                {
                    this.sensor = null;
                }
            }
        }

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (null != this.sensor)
            {
                this.sensor.Stop();
            }
        }

        /// <summary>
        /// Event handler for Kinect sensor's ColorFrameReady event
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void SensorColorFrameReady(object sender, ColorImageFrameReadyEventArgs e)
        {
            //drop the frame if you're not done with the last one
            if (!this.processingFrame)
            {
                this.processingFrame = true;
                using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
                {
                    if (colorFrame != null)
                    {
                        // Copy the pixel data from the image to a temporary array
                        colorFrame.CopyPixelDataTo(this.colorPixels);

                        Bitmap bmp = new Bitmap(this.colorBitmap.PixelWidth, this.colorBitmap.PixelHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                        System.Drawing.Imaging.BitmapData bmData = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, bmp.PixelFormat);
                        IntPtr pNative = bmData.Scan0;
                        System.Runtime.InteropServices.Marshal.Copy(this.colorPixels, 0, pNative, this.colorPixels.Length);
                        bmp.UnlockBits(bmData);

                        colorFilter.ApplyInPlace(bmp);

                        this.colorBitmap.WritePixels(
                            new Int32Rect(0, 0, this.colorBitmap.PixelWidth, this.colorBitmap.PixelHeight),
                            pNative,
                            this.colorPixels.Length,
                            this.colorBitmap.PixelWidth * sizeof(int)
                            );



                        // Write the pixel data into our bitmap
                        /*this.colorBitmap.WritePixels(
                            new Int32Rect(0, 0, this.colorBitmap.PixelWidth, this.colorBitmap.PixelHeight),
                            this.colorPixels,
                            this.colorBitmap.PixelWidth * sizeof(int),
                            0);
                        */

                    }
                }
                this.processingFrame = false;
            }
        }

        private void ImageClick(object sender, MouseButtonEventArgs e)
        {
            System.Windows.Point clickPoint = e.GetPosition(this.Image);
            int pixelX = (int) Math.Floor(clickPoint.X);
            int pixelY = (int) Math.Floor(clickPoint.Y);

            int pixelIndex = (this.colorBitmap.PixelWidth * pixelY + pixelX)*4;
            byte b = this.colorPixels[pixelIndex];
            byte g = this.colorPixels[pixelIndex + 1];
            byte r = this.colorPixels[pixelIndex + 2];

            RGB rgbColor = new RGB(r, g, b);
            HSL hslColor = HSL.FromRGB(rgbColor);
            YCbCr yuvColor = YCbCr.FromRGB(rgbColor);

            this.colorFilter.Cb = new AForge.Range(yuvColor.Cb - 0.02f, yuvColor.Cb + 0.02f);
            this.colorFilter.Cr = new AForge.Range(yuvColor.Cr - 0.02f, yuvColor.Cr + 0.02f);
            this.colorFilter.Y = new AForge.Range(yuvColor.Y - 0.1f, yuvColor.Y + 0.1f);

            

            this.statusText.Text = "H: " + hslColor.Hue.ToString() + " S:" + hslColor.Saturation.ToString() + " L:" + hslColor.Luminance.ToString();
        }
    }
}
