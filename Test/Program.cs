using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Neural_network;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            FeedForwardTest();
            //DataSetTest();
        }

        private static void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                /*
                 * Result: 1 - traffic jams
                 *         0 - no traffic jams on the roads
                 *         
                 * (R) - Raining
                 * (D) - Daylight
                 * (W) - Workday
                 * (S) - Servicebility of traffic lights
                 * 
                 *   R  D  W  S
                 */                                           
                    {0, 0, 0, 0},
                    {0, 0, 0, 1},
                    {0, 0, 1, 0},
                    {0, 0, 1, 1},
                    {0, 1, 0, 0},
                    {0, 1, 0, 1},
                    {0, 1, 1, 0},
                    {0, 1, 1, 1},
                    {1, 0, 0, 0},
                    {1, 0, 0, 1},
                    {1, 0, 1, 0},
                    {1, 0, 1, 1},
                    {1, 1, 0, 0},
                    {1, 1, 0, 1},
                    {1, 1, 1, 0},
                    {1, 1, 1, 1},
            };

            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs, inputs, 100000);

            var results = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;

                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 4);
                var actualValue = Math.Round(results[i], 4);
                //var areEqual = Equals(expected, actualValue);

                Console.WriteLine(expected + "  " + actualValue);

            }
        }

        private static void DataSetTest()
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();

            using (var streamReader = new StreamReader("heart.csv"))
            {
                var header = streamReader.ReadLine();

                while (!streamReader.EndOfStream)
                {
                    var row = streamReader.ReadLine();
                    var values = row.Split(',').Select(v => Convert.ToDouble(v.Replace('.', ','))).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }
            }

            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (int j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }
            var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 100);

            var results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {               
                var res = neuralNetwork.FeedForward(inputs[i]).Output;

                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 4);
                var actualValue = Math.Round(results[i], 4);
                //var areEqual = Equals(expected, actualValue);

                Console.WriteLine(expected + "  " + actualValue);

            }
        }
    }
}
