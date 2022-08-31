using System;
using System.Collections.Generic;

namespace Neural_network
{
    public class Neuron
    {
        public List<double> Inputs { get; }
        public List<double> Weights { get; }
        public NeuronType NeuronType { get; }
        public double Delta { get; private set; }
        public double Output { get; private set; }

        public Neuron(int inputCounts, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Inputs = new List<double>();
            Weights = new List<double>();

            InitWeightsRandomValue(inputCounts);
        }

        private void InitWeightsRandomValue(int inputCounts)
        {
            var random = new Random();

            for (int i = 0; i < inputCounts; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(random.NextDouble());
                }
                
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;

            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];

            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }

        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var input = Inputs[i];
                var weight = Weights[i];
                var newWeight = weight - input * Delta * learningRate;

                Weights[i] = newWeight;
            }
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));

            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);

            return result;
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
