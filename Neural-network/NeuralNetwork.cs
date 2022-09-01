using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural_network
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += BackPropagation(output, input);
                }
            }
            var result = error / epoch;

            return result;
        }

        private double[,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if (item < min)
                    {
                        min = item;
                    }
                    if (item > max)
                    {
                        max = item;
                    }
                }

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / (max - min);
                }
            }

            return result;
        }

        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                // The average value of the neuron signal
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                // Standart square deviation of a neuron
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow(inputs[row, column] - average, 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }

            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];

            for (int i = 0; i < columns; i++)
            {
                array[i] = matrix[row, i];
            }

            return array;
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();

            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);

            }

            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayer()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();

                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);

                }

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);

            }

            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            var actualValue = FeedForward(inputs).Output;
            var difference = actualValue - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var prevLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < prevLayer.NeuronCount; k++)
                    {
                        var prevNeuron = prevLayer.Neurons[k];
                        var error = prevNeuron.Weights[i] * prevNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);

                    }
                }
            }

            var result = difference * difference;

            return result;
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);

            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }
    }
}
