using System;
using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;

namespace image_indexer
{
    class Program
    {
        static void Main(string[] args)
        {
            //Load train data
            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_shape: new Shape(2)));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));
        }
    }
}
