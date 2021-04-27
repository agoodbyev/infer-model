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

namespace Model
{
    class Program
    {
        static int N = 3, M = 24;
        static double[,] X = new double[N, M];
        static double[,] Y = new double[N, M];
        static double[] t = new double[M];

        static void GenerateData()
        {
            t[0] = 0;
            X[0, 0] = 0;
            Y[0, 0] = 0;
            X[1, 0] = 0;
            Y[1, 0] = 10;
            X[2, 0] = 0;
            Y[2, 0] = 20;
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("Coordinates of point #" + i + ":");
                for (int j = 1; j < M; j++)
                {
                    //X[i, j] = j * 10 + i * 10 + Rand.Normal();
                    //Y[i, j] = j * 10 + i * 10 + Rand.Normal();

                    //X[i, j] = Math.Cos(Math.PI * j / M) * 10 + X[i, j - 1] + Rand.Normal();
                    //Y[i, j] = Math.Sin(Math.PI * j / M) * 10 + Y[i, j - 1] + Rand.Normal();

                    //X[i, j] = Math.Cos(Math.PI * 2 * (j - 1) / (M - 1)) * 10 + X[i, j - 1] + Rand.Normal();
                    //Y[i, j] = Math.Sin(Math.PI * 2 * (j - 1) / (M - 1)) * 10 + Y[i, j - 1] + Rand.Normal();

                    //X[i, j] = Rand.Normal();
                    //Y[i, j] = Rand.Normal();

                    //X[i, j] = Rand.Normal() + 5 * (j - M / 2) * (j - M / 2); //parabola
                    //Y[i, j] = Rand.Normal() + 5 * (j - M / 2) * (j - M / 2);

                    //X[i, j] = Math.Pow(-1, j) * 10 + X[i, j - 1] + Rand.Normal();
                    //Y[i, j] = Math.Pow(-1, j) * 10 + Y[i, j - 1] + Rand.Normal();

                    //X[i, j] = X[i, j - 1] + 10 + Rand.Normal();
                    //Y[i, j] = 10 * i + Rand.Normal();
                    if(j < 9)
                    {
                        X[i, j] = X[i, j - 1] + 10 + Rand.Normal();
                        Y[i, j] = 10 * i + Rand.Normal();
                    }
                    else if (j < 14)
                    {
                        X[i, j] = X[i, j - 1] + 10 + Rand.Normal();
                        Y[i, j] = Y[i, j - 1] + Rand.Normal() + (10 - Y[i, 8]) / 5;
                    }
                    else if (j < 19)
                    {
                        X[i, j] = X[i, j - 1] + Rand.Normal();
                        Y[i, j] = Y[i, j - 1] + Rand.Normal();
                    }
                    else
                    {
                        X[i, j] = X[i, j - 1] + 10 + Rand.Normal();
                        Y[i, j] = Y[i, j - 1] + Rand.Normal();
                    }
                    Console.Write(j + ") ");
                    Console.WriteLine("({0:N3}, {1:N3})", X[i, j], Y[i, j]);
                    if (j > 0)
                        t[j] = t[j - 1] + Rand.NormalBetween(1, 5);

                }
                Console.WriteLine();
            }
            /*Console.WriteLine("Time nodes:");
            for (int j = 0; j < M; j++)
                Console.WriteLine("{0:N3}", t[j]);*/
            Console.WriteLine();
        }

        static Bernoulli InferMovement(int Point, int NumberOfSteps, int Step)
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

            for (int j = Step - NumberOfSteps; j < Step; j++)
            {
                ArrayX[j - Step + NumberOfSteps] = X[Point, j] - X[Point, j - 1];
                ArrayY[j - Step + NumberOfSteps] = Y[Point, j] - Y[Point, j - 1];
                sumX += Math.Abs(ArrayX[j - Step + NumberOfSteps]);
                sumY += Math.Abs(ArrayY[j - Step + NumberOfSteps]);
                //Console.WriteLine(ArrayX[j - Step + NumberOfSteps] + " " + ArrayY[j - Step + NumberOfSteps]);
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
                return InferByModule(Point, NumberOfSteps, Step);
            return p;
        }

        static Bernoulli InferByModule(int Point, int NumberOfSteps, int Step)
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

            for (int j = Step - NumberOfSteps; j < Step; j++)
            {
                ArrayV[j - Step + NumberOfSteps] = Math.Sqrt((X[Point, j] - X[Point, j - 1]) * (X[Point, j] - X[Point, j - 1]) + (Y[Point, j] - Y[Point, j - 1]) * (Y[Point, j] - Y[Point, j - 1]));
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

        static double AreMovingInSameDirection(int NumberOfSteps, int Step)
        {
            double prob1 = 1, prob2 = 1;
            //for (int i = 1; i < N; i++)
            for (int j = Step - NumberOfSteps; j < Step; j++)
            {
                Variable<bool> AreMovingTogether = Variable.Bernoulli(0.5);
                Variable<double> vxMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vxMean");
                Variable<double> vyMean = Variable.GaussianFromMeanAndVariance(0, 100).Named("vyMean");
                Variable<double> vxSigma = Variable.GammaFromShapeAndScale(1, 1).Named("vxSigma");
                Variable<double> vySigma = Variable.GammaFromShapeAndScale(1, 1).Named("vySigma");
                Variable<double> stSigma = Variable.GammaFromShapeAndScale(1, 1).Named("staticSigma");
                var range = new Microsoft.ML.Probabilistic.Models.Range(N * (N - 1) / 2);
                VariableArray<double> vx = Variable.Array<double>(range);
                VariableArray<double> vy = Variable.Array<double>(range);
                double[] ArrayVx = new double[N * (N - 1) / 2];
                double[] ArrayVy = new double[N * (N - 1) / 2];

                using (Variable.If(AreMovingTogether))
                {
                    vx[range] = Variable.GaussianFromMeanAndPrecision(0, stSigma).ForEach(range);
                    vy[range] = Variable.GaussianFromMeanAndPrecision(0, stSigma).ForEach(range);
                }
                using (Variable.IfNot(AreMovingTogether))
                {
                    vx[range] = Variable.GaussianFromMeanAndPrecision(vxMean, vxSigma).ForEach(range);
                    vy[range] = Variable.GaussianFromMeanAndPrecision(vyMean, vySigma).ForEach(range);
                }

                //for (int j = Step - NumberOfSteps; j < M; j++)
                int s = 0;
                for (int i = 0; i < N; i++)
                {
                    for (int k = i + 1; k < N; k++)
                    {
                        ArrayVx[s] = (X[i, j] - X[i, j - 1]) - (X[k, j] - X[k, j - 1]);
                        ArrayVy[s] = (Y[i, j] - Y[i, j - 1]) - (Y[k, j] - Y[k, j - 1]);
                        s++;
                    }
                }
                vx.ObservedValue = ArrayVx;
                vy.ObservedValue = ArrayVy;

                InferenceEngine engine = new InferenceEngine();
                engine.Algorithm = new ExpectationPropagation();
                //InferenceEngine.Visualizer = new WindowsVisualizer();
                //engine.ShowFactorGraph = true;

                Bernoulli p = engine.Infer<Bernoulli>(AreMovingTogether);
                Gaussian xMean = engine.Infer<Gaussian>(vxMean);
                Gaussian yMean = engine.Infer<Gaussian>(vyMean);

                Console.WriteLine("xMean: " + xMean);
                Console.WriteLine("yMean: " + yMean);
                Console.WriteLine("inference of having the same character: " + p);
                Console.WriteLine();
                prob1 *= p.GetProbTrue();
                prob2 *= 1 - p.GetProbTrue();
            }
            //Console.WriteLine("prob1: " + prob1);
            //Console.WriteLine("prob2: " + (1 - prob2));
            return (1 - prob2);
        }
        static void Main(string[] args)
        {
            GenerateData();
            int NumberOfSteps = 5;
            /*for (int i = 0; i < N; i++)
            {
                InferMovement(i, 10);
                Console.WriteLine();
            }*/
            //Console.WriteLine("PROBABILITY OF MOVING TOGETHER = " + AreMovingInSameDirection(5, M));
            string writePath = @"C:\Users\andrey.bagduev\source\repos\model\model\output.txt";
            StreamWriter sw = new StreamWriter(writePath, false, System.Text.Encoding.Default);
            
            for (int j = NumberOfSteps + 1; j < M; j+= NumberOfSteps)
            {
                sw.WriteLine("Steps " + (j - NumberOfSteps) + "-" + j + ":");
                sw.WriteLine();
                for (int i = 0; i < N; i++)
                {
                    sw.WriteLine("Point #" + i + ":");
                    sw.WriteLine("Prob of moving = " + InferMovement(i, NumberOfSteps, j));
                }
                sw.WriteLine();
                sw.WriteLine("Prob of having the same character of moving = " + AreMovingInSameDirection(NumberOfSteps, j));
                sw.WriteLine("____________");
            }

            sw.Close();
        }

    }
}
