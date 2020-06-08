using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CNN_Emotions
{
    public partial class Form1 : Form
    {
        public Network.Network Network { get; set; }

        public Form1()
        {
            InitializeComponent();
            AllocConsole();

            Network = new Network.Network(new Bitmap(@"C:\Users\Dimon\Desktop\LNU\Дипломна\Photo.jpg"));
        }
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool AllocConsole();

        private void label12_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            fileDialog.Filter = "Image Files(*.jpg; *.jpeg; *.gif; *.bmp)|*.jpg; *.jpeg; *.gif; *.bmp";

            if (fileDialog.ShowDialog() == DialogResult.OK)
            {
                // display image in picture box
                pictureBox9.SizeMode = PictureBoxSizeMode.StretchImage;
                pictureBox9.Image = new Bitmap(fileDialog.FileName);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var source = new Bitmap(pictureBox9.Image);
            var bitmap = new Bitmap(48, 48);
            ScaleBitmap(bitmap, source);

            Network.Init(bitmap);
            Network.Iterate();

            DrawAllResult(Network);
            ShowResult(Network);
        }

        private void ScaleBitmap(Bitmap dest, Bitmap src)
        {
            Rectangle srcRect = new Rectangle();
            Rectangle destRect = new Rectangle();

            destRect.Width = dest.Width;
            destRect.Height = dest.Height;
            using (Graphics g = Graphics.FromImage(dest))
            {
                Brush b = new SolidBrush(Color.White);
                g.FillRectangle(b, destRect);
                srcRect.Width = src.Width;
                srcRect.Height = src.Height;
                float sourceAspect = (float)src.Width / (float)src.Height;
                float destAspect = (float)dest.Width / (float)dest.Height;
                if (sourceAspect > destAspect)
                {
                    // wider than high heep the width and scale the height
                    destRect.Width = dest.Width;
                    destRect.Height = (int)((float)dest.Width / sourceAspect);
                    destRect.X = 0;
                    destRect.Y = (dest.Height - destRect.Height) / 2;
                }
                else
                {
                    // higher than wide – keep the height and scale the width
                    destRect.Height = dest.Height;
                    destRect.Width = (int)((float)dest.Height * sourceAspect);
                    destRect.X = (dest.Width - destRect.Width) / 2;
                    destRect.Y = 0;
                }
                g.DrawImage(src, destRect, srcRect, System.Drawing.GraphicsUnit.Pixel);
            }
        }

        private void ShowResult(Network.Network network)
        {
            textBox1.Text = Math.Round(network.ResultNeurons[0].Output, 3).ToString();
            textBox2.Text = Math.Round(network.ResultNeurons[1].Output, 3).ToString();
            textBox3.Text = Math.Round(network.ResultNeurons[2].Output, 3).ToString();
            textBox4.Text = Math.Round(network.ResultNeurons[3].Output, 3).ToString();
            textBox5.Text = Math.Round(network.ResultNeurons[4].Output, 3).ToString();
            textBox6.Text = Math.Round(network.ResultNeurons[5].Output, 3).ToString();
        }

        private void DrawAllResult(Network.Network network)
        {
            var redbmp = new Bitmap(44, 44);
            var greenbmp = new Bitmap(44, 44);
            var bluebmp = new Bitmap(44, 44);

            for (int i = 0; i < 44; ++i)
            {
                for (int j = 0; j < 44; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Input_Red[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Input_Green[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Input_Blue[i, j])) % 255));
                }
            }

            pictureBox8.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox8.Image = redbmp;
            pictureBox7.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox7.Image = greenbmp;
            pictureBox10.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox10.Image = bluebmp;

            redbmp = new Bitmap(44, 44);
            greenbmp = new Bitmap(44, 44);
            bluebmp = new Bitmap(44, 44);

            for (int i = 0; i < 44; ++i)
            {
                for (int j = 0; j < 44; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_11[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_11[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_11[i, j])) % 255));
                }
            }

            pictureBox1.Image = redbmp;
            pictureBox3.Image = greenbmp;
            pictureBox5.Image = bluebmp;

            redbmp = new Bitmap(44, 44);
            greenbmp = new Bitmap(44, 44);
            bluebmp = new Bitmap(44, 44);

            for (int i = 0; i < 44; ++i)
            {
                for (int j = 0; j < 44; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_12[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_12[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_12[i, j])) % 255));
                }
            }

            pictureBox2.Image = redbmp;
            pictureBox4.Image = greenbmp;
            pictureBox6.Image = bluebmp;

            redbmp = new Bitmap(22, 22);
            greenbmp = new Bitmap(22, 22);
            bluebmp = new Bitmap(22, 22);

            for (int i = 0; i < 22; ++i)
            {
                for (int j = 0; j < 22; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_21[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_21[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_21[i, j])) % 255));
                }
            }

            pictureBox11.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox13.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox15.SizeMode = PictureBoxSizeMode.StretchImage;

            pictureBox11.Image = redbmp;
            pictureBox13.Image = greenbmp;
            pictureBox15.Image = bluebmp;

            redbmp = new Bitmap(22, 22);
            greenbmp = new Bitmap(22, 22);
            bluebmp = new Bitmap(22, 22);

            for (int i = 0; i < 22; ++i)
            {
                for (int j = 0; j < 22; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_22[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_22[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_22[i, j])) % 255));
                }
            }

            pictureBox12.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox14.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox16.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox12.Image = redbmp;
            pictureBox14.Image = greenbmp;
            pictureBox16.Image = bluebmp;

            redbmp = new Bitmap(18, 18);
            greenbmp = new Bitmap(18, 18);
            bluebmp = new Bitmap(18, 18);

            for (int i = 0; i < 18; ++i)
            {
                for (int j = 0; j < 18; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_31[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_31[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_31[i, j])) % 255));
                }
            }

            pictureBox17.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox19.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox21.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox17.Image = redbmp;
            pictureBox19.Image = greenbmp;
            pictureBox21.Image = bluebmp;

            redbmp = new Bitmap(18, 18);
            greenbmp = new Bitmap(18, 18);
            bluebmp = new Bitmap(18, 18);


            for (int i = 0; i < 18; ++i)
            {
                for (int j = 0; j < 18; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_32[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_32[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_32[i, j])) % 255));
                }
            }

            pictureBox18.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox20.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox22.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox18.Image = redbmp;
            pictureBox20.Image = greenbmp;
            pictureBox22.Image = bluebmp;

            redbmp = new Bitmap(9, 9);
            greenbmp = new Bitmap(9, 9);
            bluebmp = new Bitmap(9, 9);

            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_41[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_41[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_41[i, j])) % 255));
                }
            }

            pictureBox23.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox25.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox27.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox23.Image = redbmp;
            pictureBox25.Image = greenbmp;
            pictureBox27.Image = bluebmp;

            redbmp = new Bitmap(9, 9);
            greenbmp = new Bitmap(9, 9);
            bluebmp = new Bitmap(9, 9);

            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    redbmp.SetPixel(i, j, Color.FromArgb((int)(120 * Math.Abs(network.Map_Red_42[i, j])) % 255, 0, 0));
                    greenbmp.SetPixel(i, j, Color.FromArgb(0, (int)(120 * Math.Abs(network.Map_Green_42[i, j])) % 255, 0));
                    bluebmp.SetPixel(i, j, Color.FromArgb(0, 0, (int)(120 * Math.Abs(network.Map_Blue_42[i, j])) % 255));
                }
            }

            pictureBox24.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox26.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox28.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox24.Image = redbmp;
            pictureBox26.Image = greenbmp;
            pictureBox28.Image = bluebmp;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            var source1 = new Bitmap(@"C:\Users\dimkijeee\Desktop\Дипломна\Obama-laugh.jpg");
            var source2 = new Bitmap(@"C:\Users\dimkijeee\Desktop\Дипломна\Angry.jpeg");
            var bitmap1 = new Bitmap(48, 48);
            var bitmap2 = new Bitmap(48, 48);

            ScaleBitmap(bitmap1, source1);
            ScaleBitmap(bitmap2, source2);

            var laughExpected = new double[6] { 1, 0, 0, 0, 0, 0 };
            var angryExpected = new double[6] { 0, 0, 0, 0, 1, 0 };

            Network.Init(new Bitmap(48, 48));

            for (int i = 0; i < 50; ++i)
            {
                if (i % 2 == 0)
                {
                    Network.Train(bitmap1, laughExpected);
                }
                else
                {
                    Network.Train(bitmap2, angryExpected);
                }

                if (i % 25 == 0)
                {
                    Console.WriteLine($"Progress: {((double)i / (double)500) * 100}%");
                }
            }

            Console.Write("Result: | ");
            foreach(var p in Network.ResultNeurons)
            {
                Console.Write($"{p.Output} | ");
            }

            Console.WriteLine("Green core 1-1");
            for (int i = 0; i < 5; ++i)
            {
                for (int j = 0; j < 5; ++j)
                {
                    Console.Write($"{Math.Round(Network.Core_Green_11.Weights[i, j], 4)} ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            var source = pictureBox9.Image;
            var bitmap = new Bitmap(48, 48);

            ScaleBitmap(bitmap, new Bitmap(source));
            Network.Init(bitmap);
            
            for (int i = 0; i < 50; ++i)
                Network.Train();

            DrawAllResult(Network);
            ShowResult(Network);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            Network.Save();
        }

        private void button6_Click(object sender, EventArgs e)
        {
            var network = Network.Load();
            Network = network;
        }

        private void label6_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label7_Click(object sender, EventArgs e)
        {

        }
    }
}
