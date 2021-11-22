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
//<UseWindowsForms>true</UseWindowsForms>
namespace Model
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] Intensities = new double[90];
            bool[] Phases = new bool[90];

            // Calculation of mean intensity for each image
            for (int i = 1; i <= 90; i++)
            {
                if (i > 65) Phases[i - 1] = false;
                else Phases[i - 1] = true;

                String path = "c:\\Data/" + i.ToString() + ".jpeg";
                Bitmap bmp = new Bitmap(path);

                Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
                System.Drawing.Imaging.BitmapData bmpData =
                    bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadWrite,
                    bmp.PixelFormat);

                // Get the address of the first line.
                IntPtr ptr = bmpData.Scan0;

                // Declare an array to hold the bytes of the bitmap.
                int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
                byte[] rgbValues = new byte[bytes];

                // Copy the RGB values into the array.
                System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

                double mean_r = 0.0, mean_g = 0.0, mean_b = 0.0, mean_f;
                int p_num = 0;

                for (int counter = 0; counter < rgbValues.Length; counter += 3)
                {
                    mean_r += rgbValues[counter];
                    mean_g += rgbValues[counter + 1];
                    mean_b += rgbValues[counter + 2];
                }

                p_num = bmp.Width * bmp.Height;
                mean_r /= p_num;
                mean_g /= p_num;
                mean_b /= p_num;
                mean_f = mean_r * 0.36 + 0.53 * mean_g + 0.11 * mean_b;

                Intensities[i - 1] = mean_f;
                //Console.WriteLine("mean_f: " + mean_f);
                bmp.UnlockBits(bmpData);
            }
            
            var range = new Microsoft.ML.Probabilistic.Models.Range(90);
            VariableArray<bool> DayPhase = Variable.Array<bool>(range);
            //DayPhase[range] = Variable.Bernoulli(0.5).ForEach(range);
            Variable<double> Mean = Variable.GaussianFromMeanAndVariance(100, 100).Named("Mean");
            Variable<double> Sigma = Variable.GammaFromShapeAndScale(1, 1).Named("Sigma");
            Variable<double> Threshold = Variable.GaussianFromMeanAndPrecision(Mean, Sigma).Named("Threshold");
            VariableArray<double> v = Variable.Constant(Intensities, range);

            using (Variable.ForEach(range))
            {
                using (Variable.If(v[range] >= Threshold))
                {
                    DayPhase[range] = true;
                }
                using (Variable.IfNot(v[range] >= Threshold))
                {
                    DayPhase[range] = false;
                }
            }

            DayPhase.ObservedValue = Phases;

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new ExpectationPropagation();
            //InferenceEngine.Visualizer = new WindowsVisualizer();
            engine.ShowFactorGraph = false;
            Gaussian InfThreshold = engine.Infer<Gaussian>(Threshold);
            Gaussian InfMean = engine.Infer<Gaussian>(Mean);

            Console.WriteLine("inf threshold: " + InfThreshold);
            Console.WriteLine("inf mean: " + InfMean);
            Console.WriteLine();
        }

    }
}
