using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

//using System.Windows.Forms;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace Model
{
    class Program
    {
        static void Main(string[] args)
        {
            String imgName = "20210214172800";
            
            String path1 = "c:\\Data/leafs/test_22/" + imgName + ".jpeg";
            Bitmap bmp1 = new Bitmap(path1);
            Rectangle rect1 = new Rectangle(0, 0, bmp1.Width, bmp1.Height);
            System.Drawing.Imaging.BitmapData bmpData1 =
                bmp1.LockBits(rect1, System.Drawing.Imaging.ImageLockMode.ReadWrite,
                bmp1.PixelFormat);
            IntPtr ptr1 = bmpData1.Scan0;
            int bytes1 = Math.Abs(bmpData1.Stride) * bmp1.Height;
            byte[] rgbValues1 = new byte[bytes1];
            System.Runtime.InteropServices.Marshal.Copy(ptr1, rgbValues1, 0, bytes1);
            int lsize = bmp1.Width * bmp1.Height;
            
            String path2 = "c:\\Data/leafs/test_markdown/" + imgName + ".jpg";
            Bitmap bmp2 = new Bitmap(path2);
            Rectangle rect2 = new Rectangle(0, 0, bmp2.Width, bmp2.Height);
            System.Drawing.Imaging.BitmapData bmpData2 =
                bmp2.LockBits(rect2, System.Drawing.Imaging.ImageLockMode.ReadWrite,
                bmp2.PixelFormat);
            IntPtr ptr2 = bmpData2.Scan0;
            int bytes2 = Math.Abs(bmpData2.Stride) * bmp2.Height;
            byte[] rgbValues2 = new byte[bytes2];
            System.Runtime.InteropServices.Marshal.Copy(ptr2, rgbValues2, 0, bytes2);
            bmp2.UnlockBits(bmpData2);

            int[,] leaves = new int[rgbValues1.Length, 3]; // to hold pixels color data

            int p = 0, max, min, flg;
            int[] hyst = new int[361];
            double H;
            for (int j = 0; j < 361; j++) hyst[j] = 0;

            for (int counter = 2; counter < rgbValues1.Length; counter += 3)
            {
                if (Math.Abs(rgbValues1[counter] - rgbValues2[counter]) +
                    Math.Abs(rgbValues1[counter - 1] - rgbValues2[counter - 1]) +
                    Math.Abs(rgbValues1[counter - 2] - rgbValues2[counter - 2]) > 60)
                {
                    int r, g, b;
                    leaves[p, 0] = rgbValues1[counter]; // red
                    leaves[p, 1] = rgbValues1[counter - 1]; // green
                    leaves[p, 2] = rgbValues1[counter - 2]; // blue
                    r = rgbValues1[counter]; // to obtain Hue
                    g = rgbValues1[counter - 1];
                    b = rgbValues1[counter - 2];

                    H = 0;
                    max = -1;
                    min = 256;
                    flg = 0;
                    for (int k = 0; k < 3; k++)
                    {
                        if (leaves[p, k] > max)
                        {
                            flg = k;
                            max = leaves[p, k];
                        }
                        if (leaves[p, k] < min) min = leaves[p, k];
                    }
                    if (min == max)
                    {
                        rgbValues1[counter] = 255; // red
                        rgbValues1[counter - 1] = 0; // green
                        rgbValues1[counter - 2] = 0; // blue
                        hyst[0]++;
                        p++;
                        continue;
                    }

                    double rm, gm, bm;
                    rm = (double)(max - r) / (max - min);
                    gm = (double)(max - g) / (max - min);
                    bm = (double)(max - b) / (max - min);
                    if (flg == 0) H = 0.0 + bm - gm;
                    else if (flg == 1) H = 2.0 + rm - bm;
                    else H = 4.0 + gm - rm;
                    H = ((H / 6.0) % 1.0) * 360;
                    if (H < 0) H += 360;

                    hyst[(int)Math.Round(H)]++;
                    p++;
                    double x = (1 - Math.Abs(Math.IEEERemainder(H / 60.0, 2) - 1)) * 255;
                    if (x > 255) Console.WriteLine(x);
                    if(H >= 0 && H < 60)
                    {
                        rgbValues1[counter] = 255; // red
                        rgbValues1[counter - 1] = (byte)x; // green
                        rgbValues1[counter - 2] = 0; // blue
                    }
                    else if(H >= 60 && H < 120)
                    {
                        rgbValues1[counter] = (byte)x; // red
                        rgbValues1[counter - 1] = 255; // green
                        rgbValues1[counter - 2] = 0; // blue
                    }
                    else if (H >= 120 && H < 180)
                    {
                        rgbValues1[counter] = 0; // red
                        rgbValues1[counter - 1] = 255; // green
                        rgbValues1[counter - 2] = (byte)x; // blue
                        //Console.WriteLine(x);
                    }
                    else if (H >= 180 && H < 240)
                    {
                        rgbValues1[counter] = 0; // red
                        rgbValues1[counter - 1] = (byte)x; // green
                        rgbValues1[counter - 2] = 255; // blue
                    }
                    else if (H >= 240 && H < 300)
                    {
                        rgbValues1[counter] = (byte)x; // red
                        rgbValues1[counter - 1] = 0; // green
                        rgbValues1[counter - 2] = 255; // blue
                    }
                    else 
                    {
                        rgbValues1[counter] = 255; // red
                        rgbValues1[counter - 1] = 0; // green
                        rgbValues1[counter - 2] = (byte)x; // blue
                    }
                    
                }
            }
            System.Runtime.InteropServices.Marshal.Copy(rgbValues1, 0, ptr1, bytes1);
            bmp1.UnlockBits(bmpData1);
            System.Drawing.Image img = (System.Drawing.Image)bmp1;
            img.Save("c:\\Data/leafs/mask_" + imgName + ".jpg", ImageFormat.Jpeg);
            string writePath = @"C:\Data\leafs\hyst.txt";
            
            int numberY = 0, numberM = 0, numberC = 0;
            using (StreamWriter sw = new StreamWriter(writePath, false, System.Text.Encoding.Default))
            {
                for (int j = 0; j < 361; j++)
                {
                    sw.WriteLine(hyst[j]);
                    if (j <= 120) numberY += hyst[j];
                    else if (j <= 240) numberC += hyst[j];
                    else numberM += hyst[j];
                }
            }

            Variable<double> Sigma = Variable.GammaFromShapeAndScale(1, 1).Named("Sigma");
            Variable<double> yellowMean = Variable.GaussianFromMeanAndPrecision(50, 30).Named("yellowMean");
            Variable<double> magentaMean = Variable.GaussianFromMeanAndPrecision(170, 30).Named("magentaMean");
            Variable<double> cyanMean = Variable.GaussianFromMeanAndPrecision(290, 30).Named("cyanMean");

            var range = new Microsoft.ML.Probabilistic.Models.Range(numberY + numberM + numberC);
            VariableArray<double> hueArray = Variable.Array<double>(range);

            using (Variable.ForEach(range))
            {
                hueArray[range] = Variable.GaussianFromMeanAndPrecision(cyanMean, Sigma) 
                    + Variable.GaussianFromMeanAndPrecision(magentaMean, Sigma)
                    + Variable.GaussianFromMeanAndPrecision(yellowMean, Sigma);
            }

            double[] obs = new double[numberY + numberM + numberC];
            int cnt = 0;

            for (int j = 0; j < 361; j++)
            {
                for(int i=0; i < hyst[j]; i++)
                {
                    obs[cnt] = j;
                    cnt++;
                }
            }

            hueArray.ObservedValue = obs;

            InferenceEngine engine = new InferenceEngine();  
            Console.WriteLine("Ymean=" + engine.Infer(yellowMean));
            Console.WriteLine("Mmean=" + engine.Infer(magentaMean));
            Console.WriteLine("Cmean=" + engine.Infer(cyanMean));
            Console.WriteLine("sigma=" + engine.Infer(Sigma));
        }

        static double distrFunc(int[] hyst, double x)
        {
            if (x > 360) return 1;
            if (x < 0) return 0;
            double sum = 0, ans = 0;
            for (int j = 0; j < 361; j++)
            {
                sum += hyst[j];
            }
            for (int j = 0; j < x; j++)
            {
                ans += hyst[j] / sum;
            }
            return ans;
        }

    }
}
