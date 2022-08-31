using System;
using System.Collections.Generic;
using Neural_network;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataSet = new List<Tuple<double, double[]>>
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
                 *                                            R  D  W  S
                 */                                           
                new Tuple<double, double[]> (0, new double[] {0, 0, 0, 0}),
                new Tuple<double, double[]> (0, new double[] {0, 0, 0, 1}),
                new Tuple<double, double[]> (1, new double[] {0, 0, 1, 0}),
                new Tuple<double, double[]> (0, new double[] {0, 0, 1, 1}),
                new Tuple<double, double[]> (0, new double[] {0, 1, 0, 0}),
                new Tuple<double, double[]> (0, new double[] {0, 1, 0, 1}),
                new Tuple<double, double[]> (1, new double[] {0, 1, 1, 0}),
                new Tuple<double, double[]> (0, new double[] {0, 1, 1, 1}),
                new Tuple<double, double[]> (1, new double[] {1, 0, 0, 0}),
                new Tuple<double, double[]> (1, new double[] {1, 0, 0, 1}),
                new Tuple<double, double[]> (1, new double[] {1, 0, 1, 0}),
                new Tuple<double, double[]> (1, new double[] {1, 0, 1, 1}),
                new Tuple<double, double[]> (1, new double[] {1, 1, 0, 0}),
                new Tuple<double, double[]> (0, new double[] {1, 1, 0, 1}),
                new Tuple<double, double[]> (1, new double[] {1, 1, 1, 0}),
                new Tuple<double, double[]> (1, new double[] {1, 1, 1, 1}),
            };

            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(dataSet, 100000);
            var results = new List<double>();

            foreach (var data in dataSet)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Output);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataSet[i].Item1, 4);
                var actualValue = Math.Round(results[i], 4);
                //var areEqual = Equals(expected, actualValue);

                Console.WriteLine(expected + "  " + actualValue);

            }
        }
    }
}
