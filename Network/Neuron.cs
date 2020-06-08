using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN_Emotions.Network
{
    public class Neuron
    {
        public double Input { get; set; }
        public double Output { get; set; }

        public double[] Weights { get; set; }

        public double Value { get; set; }

        public Neuron()
        {
            var rand = new Random();

            Weights = new double[486];
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = ((double)rand.Next(0, 20) / (double)100);
            }
        }

        public Neuron(int size)
        {
            var rand = new Random();

            Weights = new double[size];
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = ((double)rand.Next(0, 20) / (double)100);
            }
        }
    }
}
