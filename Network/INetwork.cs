using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN_Emotions.Network
{
    interface INetwork
    {
        void Iterate();
        void Train();

        void BypassCore(double[,] map, Core core, double[,] resultMap, bool withRelu = true);
        void Subsample(double[,] map, double[,] resultMap);
        void ProcessConnectedLayer();

        double Activation(double value);

        Network Load();
        void Save();
    }
}
