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

namespace Model
{
    class Program
    {
        static int N = 2, M = 11;
        static double[,] X = new double[N, M];
        static double[,] Y = new double[N, M];
        static double[] t = new double[M];

        static void GenerateData()
        {
            t[0] = 0;
            X[0, 0] = 0;
            Y[0, 0] = 0;
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("Coordinates of point #" + i + ":");
                for (int j = 1; j < M; j++)
                {
                    //X[i, j] = j * 10 + i * 10 + Rand.Normal();
                    //Y[i, j] = j * 10 + i * 10 + Rand.Normal();

                    //X[i, j] = Math.Cos(Math.PI * j / M) * 10 + X[i, j - 1] + Rand.Normal();
                    //Y[i, j] = Math.Sin(Math.PI * j / M) * 10 + Y[i, j - 1] + Rand.Normal();

                    X[i, j] = Math.Cos(Math.PI * 2 * (j - 1) / (M - 1)) * 10 + X[i, j - 1] + Rand.Normal();
                    Y[i, j] = Math.Sin(Math.PI * 2 * (j - 1) / (M - 1)) * 10 + Y[i, j - 1] + Rand.Normal();

                    //X[i, j] = Rand.Normal();
                    //Y[i, j] = Rand.Normal();

                    Console.WriteLine("({0:N3}, {1:N3})", X[i, j], Y[i, j]);
                    if (j > 0)
                        t[j] = t[j - 1] + Rand.NormalBetween(1, 5);

                }
                Console.WriteLine();
            }
            Console.WriteLine("Time nodes:");
            for (int j = 0; j < M; j++)
                Console.WriteLine("{0:N3}", t[j]);
            Console.WriteLine();
        }

        static Bernoulli InferMovement(int Point, int NumberOfSteps)
        {
            Console.WriteLine("Inference for point #" + Point + ":");
            Variable<bool> IsMoving = Variable.Bernoulli(0.5);
            Variable<double> vxMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vxMean");
            Variable<double> vxSigma = Variable.GammaFromShapeAndScale(1, 1).Named("vxSigma");
            Variable<double> vyMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vyMean");
            Variable<double> vySigma = Variable.GammaFromShapeAndScale(1, 1).Named("vySigma");
            Variable<double> stSigma = Variable.GammaFromShapeAndScale(1, 1).Named("staticSigma");
            var range = new Microsoft.ML.Probabilistic.Models.Range(NumberOfSteps);
            VariableArray<double> vx = Variable.Array<double>(range);
            VariableArray<double> vy = Variable.Array<double>(range);
            double[] ArrayX = new double[NumberOfSteps];
            double[] ArrayY = new double[NumberOfSteps];
            double eps = 0.1, sumX = 0, sumY = 0;

            using (Variable.If(IsMoving))
            {
                vx[range] = Variable.GaussianFromMeanAndPrecision(vxMean, vxSigma).ForEach(range);
                vy[range] = Variable.GaussianFromMeanAndPrecision(vyMean, vySigma).ForEach(range);
            }
            using (Variable.IfNot(IsMoving))
            {
                vx[range] = Variable.GaussianFromMeanAndPrecision(0, stSigma).ForEach(range);
                vy[range] = Variable.GaussianFromMeanAndPrecision(0, stSigma).ForEach(range);
            }

            for (int j = M - NumberOfSteps; j < M; j++)
            {
                ArrayX[j - M + NumberOfSteps] = X[Point, j] - X[Point, j - 1];
                ArrayY[j - M + NumberOfSteps] = Y[Point, j] - Y[Point, j - 1];
                sumX += Math.Abs(ArrayX[j - M + NumberOfSteps]);
                sumY += Math.Abs(ArrayY[j - M + NumberOfSteps]);
            }

            vx.ObservedValue = ArrayX;
            vy.ObservedValue = ArrayY;

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new ExpectationPropagation();

            //InferenceEngine.Visualizer = new WindowsVisualizer();
            //engine.ShowFactorGraph = true;

            Bernoulli p = engine.Infer<Bernoulli>(IsMoving);
            Gaussian xMean = engine.Infer<Gaussian>(vxMean);
            Gaussian yMean = engine.Infer<Gaussian>(vyMean);

            Console.WriteLine("xMean: " + xMean);
            Console.WriteLine("yMean: " + yMean);
            Console.WriteLine("inference by vx and vy: " + p);
            Console.WriteLine();

            if (p.GetProbTrue() < (1 - eps) && ((Math.Abs(xMean.GetMean()) + Math.Abs(yMean.GetMean())) > 1 || (sumX + sumY) > 4 * NumberOfSteps))
                return InferByModule(Point, NumberOfSteps);
            return p;
        }

        static Bernoulli InferByModule(int Point, int NumberOfSteps)
        {
            Variable<bool> IsMoving = Variable.Bernoulli(0.5);
            Variable<double> vMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vMean");
            Variable<double> vSigma = Variable.GammaFromShapeAndScale(1, 1).Named("vSigma");
            Variable<double> stSigma = Variable.GammaFromShapeAndScale(1, 1).Named("staticSigma");
            var range = new Microsoft.ML.Probabilistic.Models.Range(NumberOfSteps);
            VariableArray<double> v = Variable.Array<double>(range);
            double[] ArrayV = new double[NumberOfSteps];

            using (Variable.If(IsMoving))
            {
                v[range] = Variable.GaussianFromMeanAndPrecision(vMean, vSigma).ForEach(range);
            }
            using (Variable.IfNot(IsMoving))
            {
                v[range] = Variable.GaussianFromMeanAndPrecision(0, stSigma).ForEach(range);
            }
            for (int j = M - NumberOfSteps; j < M; j++)
            {
                ArrayV[j - M + NumberOfSteps] = Math.Sqrt((X[Point, j] - X[Point, j - 1]) * (X[Point, j] - X[Point, j - 1]) + (Y[Point, j] - Y[Point, j - 1]) * (Y[Point, j] - Y[Point, j - 1]));
            }

            v.ObservedValue = ArrayV;

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new ExpectationPropagation();
            //InferenceEngine.Visualizer = new WindowsVisualizer();
            //engine.ShowFactorGraph = true;

            Bernoulli p = engine.Infer<Bernoulli>(IsMoving);
            Gaussian Mean = engine.Infer<Gaussian>(vMean);

            Console.WriteLine("mean of |v|: " + Mean);
            Console.WriteLine("inference by |v|: " + p);
            Console.WriteLine();
            return p;
        }
            static void Main(string[] args)
        {
            GenerateData();
            for (int i = 0; i < N; i++)
            {
                InferMovement(i, 10);
                Console.WriteLine();
            }
        }
        
    }
}
