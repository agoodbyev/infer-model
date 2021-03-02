using System;
using static System.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace model
{
    public class Program
    {
        static void Main(string[] args)
        {
            int n = 2;
            double[] posX = new double[n];
            double[] posY = new double[n];

            double azimuth, speed, dt;
            posX[0] = 10 + Rand.Normal(0, 1);
            posY[0] = 10 + Rand.Normal(0, 1);
            posX[1] = 20 + Rand.Normal(0, 1);
            posY[1] = 20 + Rand.Normal(0, 1);
            //posX[2] = -20 + Rand.Normal(0, 1);
            //posY[2] = 20 + Rand.Normal(0, 1);

            Variable<bool> IsMoving = Variable.Bernoulli(0.5);

            for (int i = 1; i < n; i++)
            {
                Variable<double> vxMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vxMean");
                Variable<double> vxSigma = Variable.GammaFromShapeAndScale(1, 1).Named("vxSigma");
                Variable<double> vyMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vyMean");
                Variable<double> vySigma = Variable.GammaFromShapeAndScale(1, 1).Named("vySigma");

                Variable<double> vx = Variable.GaussianFromMeanAndPrecision(vxMean, vxSigma).Named("vx");
                Variable<double> vy = Variable.GaussianFromMeanAndPrecision(vyMean, vySigma).Named("vy");

                vx.ObservedValue = posX[i] - posX[i - 1];
                vy.ObservedValue = posY[i] - posY[i - 1];

                dt = 1;
                azimuth = -System.Math.Atan2(posY[i] - posY[i - 1], posX[i] - posX[i - 1]) + System.Math.PI / 2;
                azimuth *= 57.2958; //to degrees
                speed = System.Math.Sqrt((posY[i] - posY[i - 1]) * (posY[i] - posY[i - 1]) + (posX[i] - posX[i - 1]) * (posX[i] - posX[i - 1])) / dt;
                Console.WriteLine("\nMoving from (" + posX[i - 1] + ";" + posY[i - 1] + ") to (" + posX[i] + ";" + posY[i] + "):");
                Console.WriteLine("azimuth=" + azimuth);
                Console.WriteLine("speed=" + speed);

                InferenceEngine engine = new InferenceEngine();
                //engine.ShowFactorGraph = true;
                Console.WriteLine("vxMean=" + engine.Infer(vxMean));
                Console.WriteLine("vxSigma=" + engine.Infer(vxSigma));
                InferenceEngine engine1 = new InferenceEngine();
                //engine1.ShowFactorGraph = true;
                Console.WriteLine("vyMean=" + engine1.Infer(vyMean));
                Console.WriteLine("vySigma=" + engine1.Infer(vySigma));
            }
        }
    }
}


