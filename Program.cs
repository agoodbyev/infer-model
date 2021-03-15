using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
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
        static int N = 2, M = 15;
        static double[,] X = new double[N, M];
        static double[,] Y = new double[N, M];
        static double[] t = new double[M];

        static void GenerateData()
        {
            t[0] = 0;
            for(int i = 0; i < N; i++)
            {
                Console.WriteLine("Coordinates of point " + i + ":");
                for (int j = 0; j < M; j++)
                {
                    X[i, j] = j * 10 + i * 10 + Rand.Normal();
                    Y[i, j] = j * 10 + i * 10 + Rand.Normal();
                    Console.WriteLine("({0:N3}, {1:N3})", X[i, j], Y[i, j]);
                    if (j > 0)
                        t[j] = t[j - 1] + Rand.NormalBetween(1, 5);

                }
                Console.WriteLine();
            }
            Console.WriteLine("Time points:");
            for (int j = 0; j < M; j++)
                Console.WriteLine("{0:N3}", t[j]);
        }

        static void InferMovement(int Point, int NumberOfSteps)
        {
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
            }

            vx.ObservedValue = ArrayX;
            vy.ObservedValue = ArrayY;
  
            //vx.ObservedValue = Enumerable.Range(0, NumberOfSteps).Select(_ => 10 + Rand.Normal()).ToArray();
            //vy.ObservedValue = Enumerable.Range(0, NumberOfSteps).Select(_ => 10 + Rand.Normal()).ToArray();

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new ExpectationPropagation();
            //InferenceEngine.Visualizer = new WindowsVisualizer();
            engine.ShowFactorGraph = true;
            var v = engine.Infer(IsMoving);
            var xMean = engine.Infer(vxMean);
            var yMean = engine.Infer(vyMean);

            Console.WriteLine(xMean);
            Console.WriteLine(yMean);
            Console.WriteLine(v);
        }
        static void Main(string[] args)
        {
            GenerateData();
            for (int i = 0; i < N; i++) 
                InferMovement(i, 10);

        }
        
    }
}
