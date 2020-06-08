using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace CNN_Emotions.Network 
{
    public class Network: INetwork
    {
        //-------- Input components --------//
        [JsonIgnore]
        public Bitmap Image { get; set; }

        public double[,] Input_Red { get; set; }
        public double[,] Input_Green { get; set; }
        public double[,] Input_Blue { get; set; }

        public const int Input_Size = 48;

        //-------- First conv layer --------//
        public Core Core_Red_11 { get; set; } = new Core();
        public Core Core_Red_12 { get; set; } = new Core();
        public double[,] Map_Red_11 { get; set; } = new double[Layer_1_Size, Layer_1_Size];
        public double[,] Map_Red_12 { get; set; } = new double[Layer_1_Size, Layer_1_Size];

        public Core Core_Green_11 { get; set; } = new Core();
        public Core Core_Green_12 { get; set; } = new Core();
        public double[,] Map_Green_11 { get; set; } = new double[Layer_1_Size, Layer_1_Size];
        public double[,] Map_Green_12 { get; set; } = new double[Layer_1_Size, Layer_1_Size];

        public Core Core_Blue_11 { get; set; } = new Core();
        public Core Core_Blue_12 { get; set; } = new Core();
        public double[,] Map_Blue_11 { get; set; } = new double[Layer_1_Size, Layer_1_Size];
        public double[,] Map_Blue_12 { get; set; } = new double[Layer_1_Size, Layer_1_Size];

        public const int Layer_1_Size = 44;

        //-------- Second subsample layer --------//
        public double[,] Map_Red_21 { get; set; } = new double[Layer_2_Size, Layer_2_Size];
        public double[,] Map_Red_22 { get; set; } = new double[Layer_2_Size, Layer_2_Size];

        public double[,] Map_Green_21 { get; set; } = new double[Layer_2_Size, Layer_2_Size];
        public double[,] Map_Green_22 { get; set; } = new double[Layer_2_Size, Layer_2_Size];

        public double[,] Map_Blue_21 { get; set; } = new double[Layer_2_Size, Layer_2_Size];
        public double[,] Map_Blue_22 { get; set; } = new double[Layer_2_Size, Layer_2_Size];

        public const int Layer_2_Size = 22;

        //--------Third conv layer--------//
        public Core Core_Red_31 { get; set; } = new Core();
        public Core Core_Red_32 { get; set; } = new Core();
        public double[,] Map_Red_31 { get; set; } = new double[Layer_3_Size, Layer_3_Size];
        public double[,] Map_Red_32 { get; set; } = new double[Layer_3_Size, Layer_3_Size];

        public Core Core_Green_31 { get; set; } = new Core();
        public Core Core_Green_32 { get; set; } = new Core();
        public double[,] Map_Green_31 { get; set; } = new double[Layer_3_Size, Layer_3_Size];
        public double[,] Map_Green_32 { get; set; } = new double[Layer_3_Size, Layer_3_Size];

        public Core Core_Blue_31 { get; set; } = new Core();
        public Core Core_Blue_32 { get; set; } = new Core();
        public double[,] Map_Blue_31 { get; set; } = new double[Layer_3_Size, Layer_3_Size];
        public double[,] Map_Blue_32 { get; set; } = new double[Layer_3_Size, Layer_3_Size];

        public const int Layer_3_Size = 18;

        //-------- Fourth subsample layer --------//
        public double[,] Map_Red_41 { get; set; } = new double[Layer_4_Size, Layer_4_Size];
        public double[,] Map_Red_42 { get; set; } = new double[Layer_4_Size, Layer_4_Size];

        public double[,] Map_Green_41 { get; set; } = new double[Layer_4_Size, Layer_4_Size];
        public double[,] Map_Green_42 { get; set; } = new double[Layer_4_Size, Layer_4_Size];

        public double[,] Map_Blue_41 { get; set; } = new double[Layer_4_Size, Layer_4_Size];
        public double[,] Map_Blue_42 { get; set; } = new double[Layer_4_Size, Layer_4_Size];

        public const int Layer_4_Size = 9;

        //--------Hidden layers--------//

        public Neuron[] InputNeurons = new Neuron[54];
        public Neuron[] HiddenNeurons = new Neuron[18];
        public Neuron[] ResultNeurons = new Neuron[6];

        public const int Layer_5_Size = 6;

        //--------Training variables--------//
        public double Error { get; set; }
        public double[] ResultDeltas { get; set; } = new double[Layer_5_Size];
        public double[] HiddenDeltas { get; set; } = new double[18];
        public double[] InputDeltas { get; set; } = new double[54];

        public double BiasValue = 0.5;
        public double ConectedLayerBias = 3;

        public double[,] Layer_4_Red11_Deltas { get; set; } = new double[9, 9];
        public double[,] Layer_4_Red12_Deltas { get; set; } = new double[9, 9];

        public double[,] Layer_4_Green11_Deltas { get; set; } = new double[9, 9];
        public double[,] Layer_4_Green12_Deltas { get; set; } = new double[9, 9];

        public double[,] Layer_4_Blue11_Deltas { get; set; } = new double[9, 9];
        public double[,] Layer_4_Blue12_Deltas { get; set; } = new double[9, 9];

        public double[,] Layer_3_Red11_Deltas { get; set; } = new double[18, 18];
        public double[,] Layer_3_Red12_Deltas { get; set; } = new double[18, 18];

        public double[,] Layer_3_Green11_Deltas { get; set; } = new double[18, 18];
        public double[,] Layer_3_Green12_Deltas { get; set; } = new double[18, 18];

        public double[,] Layer_3_Blue11_Deltas { get; set; } = new double[18, 18];
        public double[,] Layer_3_Blue12_Deltas { get; set; } = new double[18, 18];

        public double[,] Layer_2_Red11_Deltas { get; set; } = new double[22, 22];
        public double[,] Layer_2_Red12_Deltas { get; set; } = new double[22, 22];

        public double[,] Layer_2_Green11_Deltas { get; set; } = new double[22, 22];
        public double[,] Layer_2_Green12_Deltas { get; set; } = new double[22, 22];

        public double[,] Layer_2_Blue11_Deltas { get; set; } = new double[22, 22];
        public double[,] Layer_2_Blue12_Deltas { get; set; } = new double[22, 22];

        public double[] Expected { get; set; } = new double[Layer_5_Size] 
        {
            1, 0, 0, 0, 0, 0
        };

        public double LearningSpeed { get; set; } = 1.2;
        public double LearningSpeedForConv { get; set; } = 0.1;
        public double DeltasBias = 100;

        //--------Methods--------//
        public Network(Bitmap bitmap)
        {
            Image = bitmap;

            Input_Red = new double[48, 48];
            Input_Green = new double[48, 48];
            Input_Blue = new double[48, 48];
            
            if (Image != null)
            {
                for (int i = 0; i < 48; i++)
                {
                    for (int j = 0; j < 48; j++)
                    {
                        var color = Image.GetPixel(i, j);
                        Input_Red[i, j] = 1 - (double)color.R / (double)255;
                        Input_Green[i, j] = 1 - (double)color.G / (double)255;
                        Input_Blue[i, j] = 1 - (double)color.B / (double)255;
                    }
                }
            }

            for (int i = 0; i < HiddenNeurons.Length; ++i)
            {
                HiddenNeurons[i] = new Neuron(54);
            }

            for (int i = 0; i < ResultNeurons.Length; ++i)
            {
                ResultNeurons[i] = new Neuron(18);
            }

            for (int i = 0; i < InputNeurons.Length; ++i)
            {
                InputNeurons[i] = new Neuron(9);
            }
        }

        public void Init(Bitmap bitmap)
        {
            Image = bitmap;

            Input_Red = new double[48, 48];
            Input_Green = new double[48, 48];
            Input_Blue = new double[48, 48];

            for (int i = 0; i < 48; i++)
            {
                for (int j = 0; j < 48; j++)
                {
                    var color = Image.GetPixel(i, j);
                    Input_Red[i, j] = 1 - (double)color.R / (double)255;
                    Input_Green[i, j] = 1 - (double)color.G / (double)255;
                    Input_Blue[i, j] = 1 - (double)color.B / (double)255;
                }
            }
        }

        public void Refresh()
        {
            Map_Red_11 = new double[Layer_1_Size, Layer_1_Size];
            Map_Red_12 = new double[Layer_1_Size, Layer_1_Size];

            Map_Green_11 = new double[Layer_1_Size, Layer_1_Size];
            Map_Green_12 = new double[Layer_1_Size, Layer_1_Size];

            Map_Blue_11 = new double[Layer_1_Size, Layer_1_Size];
            Map_Blue_12 = new double[Layer_1_Size, Layer_1_Size];

            //-------- Second subsample layer --------//
            Map_Red_21 = new double[Layer_2_Size, Layer_2_Size];
            Map_Red_22 = new double[Layer_2_Size, Layer_2_Size];

            Map_Green_21 = new double[Layer_2_Size, Layer_2_Size];
            Map_Green_22 = new double[Layer_2_Size, Layer_2_Size];

            Map_Blue_21 = new double[Layer_2_Size, Layer_2_Size];
            Map_Blue_22 = new double[Layer_2_Size, Layer_2_Size];

            //--------Third conv layer--------//
            Map_Red_31 = new double[Layer_3_Size, Layer_3_Size];
            Map_Red_32 = new double[Layer_3_Size, Layer_3_Size];

            Map_Green_31 = new double[Layer_3_Size, Layer_3_Size];
            Map_Green_32  = new double[Layer_3_Size, Layer_3_Size];

            Map_Blue_31 = new double[Layer_3_Size, Layer_3_Size];
            Map_Blue_32 = new double[Layer_3_Size, Layer_3_Size];

            //-------- Fourth subsample layer --------//
            Map_Red_41 = new double[Layer_4_Size, Layer_4_Size];
            Map_Red_42 = new double[Layer_4_Size, Layer_4_Size];

            Map_Green_41 = new double[Layer_4_Size, Layer_4_Size];
            Map_Green_42 = new double[Layer_4_Size, Layer_4_Size];

            Map_Blue_41 = new double[Layer_4_Size, Layer_4_Size];
            Map_Blue_42 = new double[Layer_4_Size, Layer_4_Size];

            //--------Train params--------//
            ResultDeltas = new double[Layer_5_Size];
            HiddenDeltas = new double[18];
            InputDeltas = new double[54];

            Layer_4_Red11_Deltas = new double[9, 9];
            Layer_4_Red12_Deltas = new double[9, 9];

            Layer_4_Green11_Deltas = new double[9, 9];
            Layer_4_Green12_Deltas = new double[9, 9];

            Layer_4_Blue11_Deltas = new double[9, 9];
            Layer_4_Blue12_Deltas = new double[9, 9];

            Layer_3_Red11_Deltas = new double[18, 18];
            Layer_3_Red12_Deltas = new double[18, 18];

            Layer_3_Green11_Deltas = new double[18, 18];
            Layer_3_Green12_Deltas = new double[18, 18];
             
            Layer_3_Blue11_Deltas = new double[18, 18];
            Layer_3_Blue12_Deltas = new double[18, 18];

            Layer_2_Red11_Deltas = new double[22, 22];
            Layer_2_Red12_Deltas = new double[22, 22];

            Layer_2_Green11_Deltas = new double[22, 22];
            Layer_2_Green12_Deltas = new double[22, 22];

            Layer_2_Blue11_Deltas = new double[22, 22];
            Layer_2_Blue12_Deltas = new double[22, 22];
        }

        public void Iterate()
        {
            Refresh();

            //--------Process first layer--------//
            BypassCore(Input_Red, Core_Red_11, Map_Red_11);
            BypassCore(Input_Red, Core_Red_12, Map_Red_12);

            BypassCore(Input_Green, Core_Green_11, Map_Green_11);
            BypassCore(Input_Green, Core_Green_12, Map_Green_12);

            BypassCore(Input_Blue, Core_Blue_11, Map_Blue_11);
            BypassCore(Input_Blue, Core_Blue_12, Map_Blue_12);

            //--------Process second layer--------//
            Subsample(Map_Red_11, Map_Red_21);
            Subsample(Map_Red_12, Map_Red_22);

            Subsample(Map_Green_11, Map_Green_21);
            Subsample(Map_Green_12, Map_Green_22);

            Subsample(Map_Blue_11, Map_Blue_21);
            Subsample(Map_Blue_12, Map_Blue_22);

            //--------Process third layer--------//
            BypassCore(Map_Red_21, Core_Red_31, Map_Red_31);
            BypassCore(Map_Red_22, Core_Red_32, Map_Red_32);

            BypassCore(Map_Green_21, Core_Green_31, Map_Green_31);
            BypassCore(Map_Green_22, Core_Green_32, Map_Green_32);

            BypassCore(Map_Blue_21, Core_Blue_31, Map_Blue_31);
            BypassCore(Map_Blue_22, Core_Blue_32, Map_Blue_32);

            //--------Process fourth layer--------//
            Subsample(Map_Red_31, Map_Red_41);
            Subsample(Map_Red_32, Map_Red_42);

            Subsample(Map_Green_31, Map_Green_41);
            Subsample(Map_Green_32, Map_Green_42);

            Subsample(Map_Blue_31, Map_Blue_41);
            Subsample(Map_Blue_32, Map_Blue_42);

            //--------Process hidden layers--------//
            ProcessConnectedLayer();
        }

        public double Activation(double value)
        {
            return 1 / (1 + Math.Pow(2.71, -1 * value));
            //return (Math.Pow(Math.E, 2 * value) - 1)/(Math.Pow(Math.E, 2 * value) + 1);
        }

        public double ActivationDerivative(double value)
        {
            return Activation(value) * (1 - Activation(value));
            //return 1 - Activation(value) * Activation(value);
        }

        public double ReLU(double value)
        {
            return Math.Max(0, value);
        }

        public double ReLUActivated(double value)
        {
            if (value > 0)
            {
                return 1;
            }
            else
            {
                var rand = new Random();
                return (double) rand.Next(1, 5) / (double) 100;
            }
        }

        public void CalculateError()
        {
            double error = 0;

            for (int i = 0; i < ResultNeurons.Length; ++i)
            {
                error += Math.Pow(Expected[i] - ResultNeurons[i].Output, 2);
            }

            Error = error / ResultNeurons.Length;
        }

        public void CalculateDeltaForOutputs()
        {
            for (int i = 0; i < ResultNeurons.Length; ++i)
            {
                ResultDeltas[i] = (Expected[i] - ResultNeurons[i].Output) * ActivationDerivative(ResultNeurons[i].Input) * ConectedLayerBias;
            }
        }

        public void CalculateDeltaForHiddenAndUpdateWeights()
        {
            for (int i = 0; i < HiddenNeurons.Length; ++i)
            {
                for (int j = 0; j < ResultNeurons.Length; ++j)
                {
                    for (int k = 0; k < ResultNeurons[j].Weights.Length; ++k)
                    {
                        HiddenDeltas[i] += ResultNeurons[j].Weights[k] * ResultDeltas[j];
                    }
                }

                HiddenDeltas[i] *= ActivationDerivative(HiddenNeurons[i].Input);

                for (int j = 0; j < ResultNeurons.Length; j++)
                {
                    double gradient = HiddenNeurons[i].Output * ResultDeltas[j];
                    ResultNeurons[j].Weights[i] += LearningSpeed * gradient;
                }
            }
        }

        public void UpdateInputWeights()
        {
            for (int i = 0; i < InputNeurons.Length; ++i)
            {
                for (int j = 0; j < HiddenNeurons.Length; ++j)
                {
                    for (int k = 0; k < HiddenNeurons[j].Weights.Length; ++k)
                    {
                        InputDeltas[i] += HiddenNeurons[j].Weights[k] * HiddenDeltas[j];
                    }
                }

                InputDeltas[i] *= ActivationDerivative(InputNeurons[i].Input);

                for (int j = 0; j < HiddenNeurons.Length; j++)
                {
                    double gradient = InputNeurons[i].Output * HiddenDeltas[j];
                    HiddenNeurons[j].Weights[i] += LearningSpeed * gradient;
                }
            }
        }

        public void Train()
        {
            Iterate();
            CalculateError();
            CalculateDeltaForOutputs();
            CalculateDeltaForHiddenAndUpdateWeights();
            UpdateInputWeights();
            ChangeWeightsForConvLayers();
        }

        public void Train(Bitmap bitmap, double[] expected)
        {
            Init(bitmap);
            Expected = expected;
            Train();
        }

        public void ChangeWeightsForConvLayers()
        {
            //--------Update 4 layer-------//

            int layer = 0;
            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    for (int k = 0; k < InputNeurons[j].Weights.Length; ++k)
                    {
                        Layer_4_Red11_Deltas[i, j] += InputNeurons[j + layer * Layer_4_Size].Weights[k] * InputDeltas[j + layer * Layer_4_Size];
                    }

                    Layer_4_Red11_Deltas[i, j] *= ActivationDerivative(Map_Red_41[i, j]) * BiasValue;
                }
            }

            layer++; 
            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    for (int k = 0; k < InputNeurons[j].Weights.Length; ++k)
                    {
                        Layer_4_Red12_Deltas[i, j] += InputNeurons[j + layer * Layer_4_Size].Weights[k] * InputDeltas[j + layer * Layer_4_Size];
                    }

                    Layer_4_Red12_Deltas[i, j] *= ActivationDerivative(Map_Red_42[i, j]) * BiasValue;
                }
            }

            layer++;
            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    for (int k = 0; k < InputNeurons[j].Weights.Length; ++k)
                    {
                        Layer_4_Green11_Deltas[i, j] += InputNeurons[j + layer * Layer_4_Size].Weights[k] * InputDeltas[j + layer * Layer_4_Size];
                    }

                    Layer_4_Green11_Deltas[i, j] *= ActivationDerivative(Map_Green_41[i, j]) * BiasValue;
                }
            }

            layer++;
            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    for (int k = 0; k < InputNeurons[j].Weights.Length; ++k)
                    {
                        Layer_4_Green12_Deltas[i, j] += InputNeurons[j + layer * Layer_4_Size].Weights[k] * InputDeltas[j + layer * Layer_4_Size];
                    }

                    Layer_4_Green12_Deltas[i, j] *= ActivationDerivative(Map_Green_42[i, j]) * BiasValue;
                }
            }

            layer++;
            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    for (int k = 0; k < InputNeurons[j].Weights.Length; ++k)
                    {
                        Layer_4_Green11_Deltas[i, j] += InputNeurons[j + layer * Layer_4_Size].Weights[k] * InputDeltas[j + layer * Layer_4_Size];
                    }

                    Layer_4_Green11_Deltas[i, j] *= ActivationDerivative(Map_Blue_41[i, j]) * BiasValue;
                }
            }

            layer++;
            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    for (int k = 0; k < InputNeurons[j].Weights.Length; ++k)
                    {
                        Layer_4_Blue12_Deltas[i, j] += InputNeurons[j + layer * Layer_4_Size].Weights[k] * InputDeltas[j + layer * Layer_4_Size];
                    }

                    Layer_4_Blue12_Deltas[i, j] *= ActivationDerivative(Map_Blue_42[i, j]) * BiasValue;
                }
            }

            //--------Update 3 layer--------//
            UpdateDeltasFor_3_Layer();

            double[,] gradientsForCore_31 = new double[5, 5];
            Rotate(Layer_4_Red11_Deltas);
            BypassLearning(Map_Red_21, new Core { Weights = Layer_4_Red11_Deltas }, gradientsForCore_31);
            double[,] gradientsForCore_32 = new double[5, 5];
            Rotate(Layer_4_Red12_Deltas);
            BypassLearning(Map_Red_22, new Core { Weights = Layer_4_Red12_Deltas }, gradientsForCore_32);

            double[,] gradientsForCore_33 = new double[5, 5];
            Rotate(Layer_4_Green11_Deltas);
            BypassLearning(Map_Green_21, new Core { Weights = Layer_4_Green11_Deltas }, gradientsForCore_33);
            double[,] gradientsForCore_34 = new double[5, 5];
            Rotate(Layer_4_Green12_Deltas);
            BypassLearning(Map_Green_22, new Core { Weights = Layer_4_Green12_Deltas }, gradientsForCore_34);

            double[,] gradientsForCore_35 = new double[5, 5];
            Rotate(Layer_4_Blue11_Deltas);
            BypassLearning(Map_Blue_21, new Core { Weights = Layer_4_Blue11_Deltas }, gradientsForCore_35);
            double[,] gradientsForCore_36 = new double[5, 5];
            Rotate(Layer_4_Blue12_Deltas);
            BypassLearning(Map_Blue_22, new Core { Weights = Layer_4_Blue12_Deltas }, gradientsForCore_36);

            for (int i = 0; i < 5; ++i)
            {
                for (int j = 0; j < 5; ++j)
                {
                    Core_Red_31.Weights[i, j] += gradientsForCore_31[i, j] * LearningSpeedForConv;
                    Core_Red_32.Weights[i, j] += gradientsForCore_32[i, j] * LearningSpeedForConv;

                    Core_Green_31.Weights[i, j] += gradientsForCore_33[i, j] * LearningSpeedForConv;
                    Core_Green_32.Weights[i, j] += gradientsForCore_34[i, j] * LearningSpeedForConv;

                    Core_Blue_31.Weights[i, j] += gradientsForCore_35[i, j] * LearningSpeedForConv;
                    Core_Blue_32.Weights[i, j] += gradientsForCore_36[i, j] * LearningSpeedForConv;
                }
            }

            //--------Change for 2 layer--------//
            Rotate(Core_Red_31.Weights);
            Rotate(Core_Red_32.Weights);
            Rotate(Core_Green_31.Weights);
            Rotate(Core_Green_32.Weights);
            Rotate(Core_Blue_31.Weights);
            Rotate(Core_Blue_32.Weights);

            BypassCoreSecondLayer(Layer_3_Red11_Deltas, new Core { Weights = Core_Red_31.Weights }, Layer_2_Red11_Deltas);
            BypassCoreSecondLayer(Layer_3_Red12_Deltas, new Core { Weights = Core_Red_32.Weights }, Layer_2_Red12_Deltas);
            BypassCoreSecondLayer(Layer_3_Green11_Deltas, new Core { Weights = Core_Green_31.Weights }, Layer_2_Green11_Deltas);
            BypassCoreSecondLayer(Layer_3_Green12_Deltas, new Core { Weights = Core_Green_32.Weights }, Layer_2_Green12_Deltas);
            BypassCoreSecondLayer(Layer_3_Blue11_Deltas, new Core { Weights = Core_Blue_31.Weights }, Layer_2_Blue11_Deltas);
            BypassCoreSecondLayer(Layer_3_Blue12_Deltas, new Core { Weights = Core_Blue_32.Weights }, Layer_2_Blue12_Deltas);

            var newDeltas = new double[18, 18];
            Resize(Layer_2_Red11_Deltas, newDeltas);
            Layer_2_Red11_Deltas = newDeltas;
            Resize(Layer_2_Red12_Deltas, newDeltas);
            Layer_2_Red12_Deltas = newDeltas;

            Resize(Layer_2_Green11_Deltas, newDeltas);
            Layer_2_Green11_Deltas = newDeltas;
            Resize(Layer_2_Green12_Deltas, newDeltas);
            Layer_2_Green12_Deltas = newDeltas;

            Resize(Layer_2_Blue11_Deltas, newDeltas);
            Layer_2_Blue11_Deltas = newDeltas;
            Resize(Layer_2_Blue12_Deltas, newDeltas);
            Layer_2_Blue12_Deltas = newDeltas;

            Rotate(Core_Red_31.Weights);
            Rotate(Core_Red_32.Weights);
            Rotate(Core_Green_31.Weights);
            Rotate(Core_Green_32.Weights);
            Rotate(Core_Blue_31.Weights);
            Rotate(Core_Blue_32.Weights);

            //--------Change for first layer--------//
            double[,] gradientsForCore_11 = new double[5, 5];
            Rotate(Layer_2_Red11_Deltas);
            BypassLearning(Map_Red_11, new Core { Weights = Layer_2_Red11_Deltas }, gradientsForCore_11);
            double[,] gradientsForCore_12 = new double[5, 5];
            Rotate(Layer_2_Red12_Deltas);
            BypassLearning(Map_Red_12, new Core { Weights = Layer_2_Red12_Deltas }, gradientsForCore_12);

            double[,] gradientsForCore_13 = new double[5, 5];
            Rotate(Layer_2_Green11_Deltas);
            BypassLearning(Map_Green_11, new Core { Weights = Layer_2_Green11_Deltas }, gradientsForCore_13);
            double[,] gradientsForCore_14 = new double[5, 5];
            Rotate(Layer_2_Green12_Deltas);
            BypassLearning(Map_Green_12, new Core { Weights = Layer_2_Green12_Deltas }, gradientsForCore_14);

            double[,] gradientsForCore_15 = new double[5, 5];
            Rotate(Layer_2_Blue11_Deltas);
            BypassLearning(Map_Blue_11, new Core { Weights = Layer_2_Blue11_Deltas }, gradientsForCore_15);
            double[,] gradientsForCore_16 = new double[5, 5];
            Rotate(Layer_2_Blue12_Deltas);
            BypassLearning(Map_Blue_12, new Core { Weights = Layer_2_Blue12_Deltas }, gradientsForCore_16);

            for (int i = 0; i < 5; ++i)
            {
                for (int j = 0; j < 5; ++j)
                {
                    Core_Red_11.Weights[i, j] += gradientsForCore_11[i, j] * LearningSpeedForConv;
                    Core_Red_12.Weights[i, j] += gradientsForCore_12[i, j] * LearningSpeedForConv;

                    Core_Green_11.Weights[i, j] += gradientsForCore_13[i, j] * LearningSpeedForConv;
                    Core_Green_12.Weights[i, j] += gradientsForCore_14[i, j] * LearningSpeedForConv;

                    Core_Blue_11.Weights[i, j] += gradientsForCore_15[i, j] * LearningSpeedForConv;
                    Core_Blue_12.Weights[i, j] += gradientsForCore_16[i, j] * LearningSpeedForConv;
                }
            }
        }

        public void UpdateDeltasFor_3_Layer()
        {
            int p = 0, q = 0;
            for (int i = 0; i < Math.Sqrt(Map_Red_31.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(Map_Red_32.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(Map_Red_32[i, j], Map_Red_32[i, j + 1]), Math.Max(Map_Red_32[i + 1, j], Map_Red_32[i + 1, j + 1]));
                    q++;

                    if (max == Map_Red_32[i, j])
                    {
                        Layer_3_Red12_Deltas[i, j] = Layer_4_Red12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Red_32[i, j + 1])
                    {
                        Layer_3_Red12_Deltas[i, j + 1] = Layer_4_Red12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Red_32[i + 1, j])
                    {
                        Layer_3_Red12_Deltas[i + 1, j] = Layer_4_Red12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Red_32[i + 1, j + 1])
                    {
                        Layer_3_Red12_Deltas[i + 1, j + 1] = Layer_4_Red12_Deltas[i / 2, j / 2];
                    }
                }
                p++;
                q = 0;
            }

            p = 0; q = 0;
            for (int i = 0; i < Math.Sqrt(Map_Red_31.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(Map_Red_31.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(Map_Red_31[i, j], Map_Red_31[i, j + 1]), Math.Max(Map_Red_31[i + 1, j], Map_Red_31[i + 1, j + 1]));
                    q++;

                    if (max == Map_Red_31[i, j])
                    {
                        Layer_3_Red11_Deltas[i, j] = Layer_4_Red11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Red_31[i, j + 1])
                    {
                        Layer_3_Red11_Deltas[i, j + 1] = Layer_4_Red11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Red_31[i + 1, j])
                    {
                        Layer_3_Red11_Deltas[i + 1, j] = Layer_4_Red11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Red_31[i + 1, j + 1])
                    {
                        Layer_3_Red11_Deltas[i + 1, j + 1] = Layer_4_Red11_Deltas[i / 2, j / 2];
                    }
                }
                p++;
                q = 0;
            }

            p = 0; q = 0;
            for (int i = 0; i < Math.Sqrt(Map_Green_31.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(Map_Green_31.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(Map_Green_31[i, j], Map_Green_31[i, j + 1]), Math.Max(Map_Green_31[i + 1, j], Map_Green_31[i + 1, j + 1]));
                    q++;

                    if (max == Map_Green_31[i, j])
                    {
                        Layer_3_Green11_Deltas[i, j] = Layer_4_Green11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Green_31[i, j + 1])
                    {
                        Layer_3_Green11_Deltas[i, j + 1] = Layer_4_Green11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Green_31[i + 1, j])
                    {
                        Layer_3_Green11_Deltas[i + 1, j] = Layer_4_Green11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Green_31[i + 1, j + 1])
                    {
                        Layer_3_Green11_Deltas[i + 1, j + 1] = Layer_4_Green11_Deltas[i / 2, j / 2];
                    }
                }
                p++;
                q = 0;
            }

            p = 0; q = 0;
            for (int i = 0; i < Math.Sqrt(Map_Green_32.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(Map_Green_32.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(Map_Green_32[i, j], Map_Green_32[i, j + 1]), Math.Max(Map_Green_32[i + 1, j], Map_Green_32[i + 1, j + 1]));
                    q++;

                    if (max == Map_Green_32[i, j])
                    {
                        Layer_3_Green12_Deltas[i, j] = Layer_4_Green12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Green_32[i, j + 1])
                    {
                        Layer_3_Green12_Deltas[i, j + 1] = Layer_4_Green12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Green_32[i + 1, j])
                    {
                        Layer_3_Green12_Deltas[i + 1, j] = Layer_4_Green12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Green_32[i + 1, j + 1])
                    {
                        Layer_3_Green12_Deltas[i + 1, j + 1] = Layer_4_Green12_Deltas[i / 2, j / 2];
                    }
                }
                p++;
                q = 0;
            }

            p = 0; q = 0;
            for (int i = 0; i < Math.Sqrt(Map_Blue_31.Length) - 1; i = i + 2)
            {

                for (int j = 0; j < Math.Sqrt(Map_Blue_31.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(Map_Blue_31[i, j], Map_Blue_31[i, j + 1]), Math.Max(Map_Blue_31[i + 1, j], Map_Blue_31[i + 1, j + 1]));
                    q++;

                    if (max == Map_Blue_31[i, j])
                    {
                        Layer_3_Blue11_Deltas[i, j] = Layer_4_Blue11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Blue_31[i, j + 1])
                    {
                        Layer_3_Blue11_Deltas[i, j + 1] = Layer_4_Blue11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Blue_31[i + 1, j])
                    {
                        Layer_3_Blue11_Deltas[i + 1, j] = Layer_4_Blue11_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Blue_31[i + 1, j + 1])
                    {
                        Layer_3_Blue11_Deltas[i + 1, j + 1] = Layer_4_Blue11_Deltas[i / 2, j / 2];
                    }
                }
                p++;
                q = 0;
            }

            p = 0; q = 0;
            for (int i = 0; i < Math.Sqrt(Map_Blue_32.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(Map_Blue_32.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(Map_Blue_32[i, j], Map_Blue_32[i, j + 1]), Math.Max(Map_Blue_32[i + 1, j], Map_Blue_32[i + 1, j + 1]));
                    q++;

                    if (max == Map_Blue_32[i, j])
                    {
                        Layer_3_Blue12_Deltas[i, j] = Layer_4_Blue12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Blue_32[i, j + 1])
                    {
                        Layer_3_Blue12_Deltas[i, j + 1] = Layer_4_Blue12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Blue_32[i + 1, j])
                    {
                        Layer_3_Blue12_Deltas[i + 1, j] = Layer_4_Blue12_Deltas[i / 2, j / 2];
                    }

                    if (max == Map_Blue_32[i + 1, j + 1])
                    {
                        Layer_3_Blue12_Deltas[i + 1, j + 1] = Layer_4_Blue12_Deltas[i / 2, j / 2];
                    }
                }
                p++;
                q = 0;
            }
        }

        public void BypassLearning(double[,] map, Core core, double[,] resultMap)
        {
            var size = (int)Math.Sqrt(core.Weights.Length) / 2;

            for (int i = 0; i < 5; ++i)
            {
                for (int j = 0; j < 5; ++j)
                {
                    double result = 0;
                    for (int p = 0; p < Math.Sqrt(core.Weights.Length); ++p)
                    {
                        for (int q = 0; q < Math.Sqrt(core.Weights.Length); ++q)
                        {
                            result += map[i + p, j + q] * core.Weights[i, j];
                        }
                    }

                    resultMap[i, j] = result;
                }
            }
        }

        public void BypassCoreSecondLayer(double[,] map, Core core, double[,] resultMap)
        {
            int size = (int)Math.Sqrt(map.Length) + 4;
            double[,] newMap = new double[size, size];

            for (int i = 2; i < size - 2; ++i)
            {
                for (int j = 2; j < size - 2; ++j)
                {
                    newMap[i, j] = map[i-2, j-2];
                }
            }

            //process left side
            for (int i = 2; i < size - 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    newMap[i, j] = map[i - 2, j];
                }
            }

            //top
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 2; j < size - 2; ++j)
                {
                    newMap[i, j] = map[i, j - 2];
                }
            }

            //right
            for (int i = 2; i < size - 2; ++i)
            {
                for (int j = size - 3; j < size; ++j)
                {
                    newMap[i, j] = map[i - 2, j - 4];
                }
            }

            //bottom
            for (int i = size - 3; i < size; ++i)
            {
                for (int j = 2; j < size - 3; ++j)
                {
                    newMap[i, j] = map[i - 4, j - 2];
                }
            }

            newMap[0, 0] = map[0, 0];
            newMap[0, 1] = map[0, 0];
            newMap[1, 0] = map[0, 0];
            newMap[1, 1] = map[0, 0];

            newMap[0, size - 2] = map[0, size - 6];
            newMap[0, size - 1] = map[0, size - 5];
            newMap[1, size - 2] = map[0, size - 6];
            newMap[1, size - 1] = map[0, size - 5];

            newMap[size - 2, 0] = map[size - 6, 0];
            newMap[size - 2, 1] = map[size - 6, 1];
            newMap[size - 1, 0] = map[size - 5, 0];
            newMap[size - 1, 1] = map[size - 5, 1];

            newMap[size - 2, size - 2] = map[size - 6, size - 6];
            newMap[size - 2, size - 1] = map[size - 6, size - 5];
            newMap[size - 1, size - 2] = map[size - 5, size - 6];
            newMap[size - 1, size - 1] = map[size - 5, size - 5];

            map = newMap;
            BypassCore(map, core, resultMap, false);
        }

        public void BypassCore(double[,] map, Core core, double[,] resultMap, bool withRelu = true)
        {
            for (int i = 2; i < Math.Sqrt(map.Length) - 2; i++)
            {
                for (int j = 2; j < Math.Sqrt(map.Length) - 2; j++)
                {
                    double value =
                        map[i - 2, j - 2] * core.Weights[0, 0] +
                        map[i - 2, j - 1] * core.Weights[0, 1] +
                            map[i - 2, j] * core.Weights[0, 2] +
                        map[i - 2, j + 1] * core.Weights[0, 3] +
                        map[i - 2, j + 2] * core.Weights[0, 4] +

                        map[i - 1, j - 2] * core.Weights[1, 0] +
                        map[i - 1, j - 1] * core.Weights[1, 1] +
                            map[i - 1, j] * core.Weights[1, 2] +
                        map[i - 1, j + 1] * core.Weights[1, 3] +
                        map[i - 1, j + 2] * core.Weights[1, 4] +

                        map[i, j - 2] * core.Weights[2, 0] +
                        map[i, j - 1] * core.Weights[2, 1] +
                            map[i, j] * core.Weights[2, 2] +
                        map[i, j + 1] * core.Weights[2, 3] +
                        map[i, j + 2] * core.Weights[2, 4] +

                        map[i + 1, j - 2] * core.Weights[3, 0] +
                        map[i + 1, j - 1] * core.Weights[3, 1] +
                            map[i + 1, j] * core.Weights[3, 2] +
                        map[i + 1, j + 1] * core.Weights[3, 3] +
                        map[i + 1, j + 2] * core.Weights[3, 4] +

                        map[i + 2, j - 2] * core.Weights[4, 0] +
                        map[i + 2, j - 1] * core.Weights[4, 1] +
                            map[i + 2, j] * core.Weights[4, 2] +
                        map[i + 2, j + 1] * core.Weights[4, 3] +
                        map[i + 2, j + 2] * core.Weights[4, 4];

                        resultMap[i - 2, j - 2] = withRelu ? ReLU(value) : value;
                }
            }
        }

        public void SubsampleLearning(double[,] map, double[,] deltasPrev, double[,] deltasNext)
        {
            for (int i = 0; i < Math.Sqrt(map.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(map.Length) - 1; j = j + 2)
                {
                    double max = 
                        Math.Max(Math.Max(map[i, j], map[i, j + 1]), Math.Max(map[i + 1, j], map[i + 1, j + 1]));

                    if (max == map[i, j])
                        deltasNext[i, j] = deltasPrev[i / 2, j / 2];
                    if (max == map[i, j + 1])
                        deltasNext[i, j + 1] = deltasPrev[i / 2, j / 2];
                    if (max == map[i + 1, j])
                        deltasNext[i + 1, j] = deltasPrev[i / 2, j / 2];
                    if (max == map[i + 1, j + 1])
                        deltasNext[i + 1, j + 1] = deltasPrev[i / 2, j / 2];
                }
            }
        }

        public void Subsample(double[,] map, double[,] resultMap)
        {
            int p = 0, q = 0;

            for (int i = 0; i < Math.Sqrt(map.Length) - 1; i = i + 2)
            {
                for (int j = 0; j < Math.Sqrt(map.Length) - 1; j = j + 2)
                {
                    resultMap[p, q] = ReLU(
                        Math.Max(Math.Max(map[i, j], map[i, j + 1]), Math.Max(map[i+ 1, j], map[i + 1, j + 1])));
                    q++;
                }
                p++;
                q = 0;
            }
        }

        public void ProcessConnectedLayer()
        {
            int section = 0;
            double value = 0;

            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    value += Map_Red_41[i, j] * InputNeurons[Layer_4_Size * section + i].Weights[j];
                }

                InputNeurons[Layer_4_Size * section + i].Input = value;
                InputNeurons[Layer_4_Size * section + i].Output = Activation(value);

                value = 0;
            }
            section++;

            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    value += Map_Red_42[i, j] * InputNeurons[Layer_4_Size * section + i].Weights[j];
                }

                InputNeurons[Layer_4_Size * section + i].Input = value;
                InputNeurons[Layer_4_Size * section + i].Output = Activation(value);

                value = 0;
            }
            section++;

            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    value += Map_Green_41[i, j] * InputNeurons[Layer_4_Size * section + i].Weights[j];
                }

                InputNeurons[Layer_4_Size * section + i].Input = value;
                InputNeurons[Layer_4_Size * section + i].Output = Activation(value);

                value = 0;
            }
            section++;

            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    value += Map_Green_42[i, j] * InputNeurons[Layer_4_Size * section + i].Weights[j];
                }

                InputNeurons[Layer_4_Size * section + i].Input = value;
                InputNeurons[Layer_4_Size * section + i].Output = Activation(value);

                value = 0;
            }
            section++;

            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    value += Map_Blue_41[i, j] * InputNeurons[Layer_4_Size * section + i].Weights[j];
                }

                InputNeurons[Layer_4_Size * section + i].Input = value;
                InputNeurons[Layer_4_Size * section + i].Output = Activation(value);

                value = 0;
            }
            section++;

            for (int i = 0; i < Layer_4_Size; ++i)
            {
                for (int j = 0; j < Layer_4_Size; ++j)
                {
                    value += Map_Blue_42[i, j] * InputNeurons[Layer_4_Size * section + i].Weights[j];
                }

                InputNeurons[Layer_4_Size * section + i].Input = value;
                InputNeurons[Layer_4_Size * section + i].Output = Activation(value);

                value = 0;
            }
            section++;

            int p = 0;
            for (int i = 0; i < HiddenNeurons.Length; ++i)
            {
                p = 0;
                HiddenNeurons[i].Input = 0;

                foreach (var n in InputNeurons)
                {
                    HiddenNeurons[i].Input += n.Output * HiddenNeurons[i].Weights[p];
                    p++;
                }

                HiddenNeurons[i].Output = Activation(HiddenNeurons[i].Input);
            }

            for (int i = 0; i < ResultNeurons.Length; ++i)
            {
                p = 0;
                ResultNeurons[i].Input = 0;

                foreach (var n in HiddenNeurons)
                {
                    ResultNeurons[i].Input += n.Output * ResultNeurons[i].Weights[p];
                    p++;
                }

                foreach (var n in ResultNeurons)
                {
                    //n.Input += BiasValue;
                    n.Output = Activation(n.Input);
                }
            }
        }

        public Network Load()
        {
            var serialized = File.ReadAllText(@"C:\Users\Dimon\Desktop\LNU\repos\CNN_Emotions\Backup\CNN.txt");
            return JsonConvert.DeserializeObject<Network>(serialized);
        }

        public void Save()
        {
            var serialized = JsonConvert.SerializeObject(this);
            File.WriteAllText(@"C:\Users\dimkijeee\source\repos\CNN_Emotions\Backup\CNN.txt", serialized);
        }

        private void Rotate90(double[,] mas, int n)
        {
            double tmp;
            for (int i = 0; i < n / 2; i++)
            {
                for (int j = i; j < n - 1 - i; j++)
                {
                    tmp = mas[i, j];
                    mas[i, j] = mas[n - j - 1, i];
                    mas[n - j - 1, i] = mas[n - i - 1, n - j - 1];
                    mas[n - i - 1, n - j - 1] = mas[j, n - i - 1];
                    mas[j, n - i - 1] = tmp;
                }
            }
        }

        private void Rotate(double [,] map)
        {
            Rotate90(map, (int)Math.Sqrt(map.Length));
            Rotate90(map, (int)Math.Sqrt(map.Length));
        }

        private void Resize(double [,] map, double[,] newMap)
        {
            for (int i = 0; i < Math.Sqrt(newMap.Length); ++i)
            {
                for (int j = 0; j < Math.Sqrt(newMap.Length); ++j)
                {
                    newMap[i, j] = map[i, j];
                }
            }
        }
    }
}
