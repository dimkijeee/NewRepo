using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN_Emotions.Network
{
    public class Core
    {
        public double[,] Weights { get; set; } = new double[5, 5];

        public Core()
        {
            var rand = new Random();

            for (int i = 0; i < 5; ++i)
            {
                for (int j = 0; j < 5; ++j)
                {
                    Weights[i, j] = ((double)rand.Next(0, 20) / (double)100);
                    if (rand.Next(0, 3) % 2 == 0)
                    {
                        Weights[i, j] *= -1;
                    }
                }
            }
        }
    }
}
