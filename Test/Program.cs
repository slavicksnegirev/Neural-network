using System;
using System.Collections.Generic;
using Neural_network;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
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
    }
}
