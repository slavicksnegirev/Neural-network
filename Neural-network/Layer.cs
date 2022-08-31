﻿using System;
using System.Collections.Generic;

namespace Neural_network
{
    public class Layer
    {
        public NeuronType Type;
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            Type = type;
            Neurons = neurons;  
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();

            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }

            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
