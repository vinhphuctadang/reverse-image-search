// using static Tensorflow.Binding;
// using static Tensorflow.KerasApi;
using System;
using Python.Runtime;
using System.Collections.Generic;
using System.IO;

namespace image_indexer
{
    class Program
    {   
        static string modelDir;
        static dynamic image_processor;// = Py.Import("tensorflow.keras.preprocessing.image");
        static dynamic kerasApplications; // = Py.Import("tensorflow.keras.applications");
        static dynamic mobileNet;// kerasApplications.MobileNetV2;
        static dynamic tensorflowVgg16; // Py.Import("tensorflow.keras.applications.vgg16");
        static dynamic preprocessInput; // tensorflowVgg16.preprocess_input;
        static dynamic numpy;
        static dynamic annoy;
        static dynamic model;
        // matplotlib
        static dynamic plt;

        // current directory

        static string currentDir;

        static void Init() {
            // import modules
            image_processor = Py.Import("tensorflow.keras.preprocessing.image");
            kerasApplications = Py.Import("tensorflow.keras.applications");
            mobileNet = kerasApplications.MobileNetV2;
            tensorflowVgg16 = Py.Import("tensorflow.keras.applications.vgg16");
            preprocessInput = tensorflowVgg16.preprocess_input;
            numpy = Py.Import("numpy");
            annoy = Py.Import("annoy");
            plt = Py.Import("matplotlib.pyplot");


            // get current directory
            currentDir = Directory.GetCurrentDirectory();

            // load model
            modelDir = currentDir + @"\..\model\mobilenetv2_notop";
            Console.WriteLine("model dir:" + modelDir);

            model = mobileNet(
                include_top: false,
                weights: modelDir
            );
        }
        // path is referred to local path
        static dynamic load_image(dynamic image_processor, string path){
            return image_processor.load_img(path, Py.kw("target_size", new int[] {224, 224}));
        }
        static List<dynamic> ImportData(string path) {
            
            var files = Directory.GetFiles(path);
            // create image list
            var imageList = new List<dynamic>();

            foreach(var file in files) {
                dynamic image = load_image(image_processor, file);
                // just add image to list
                imageList.Add(image);
            }
            return imageList;
        }
        static List<dynamic> Featurize(List<dynamic> imageList){
            var result = new List<dynamic>();
            
            // preprocess imageList:
            var processedImages = new List<dynamic>();
            foreach(var image in imageList) {

                // convert image to array, omit headers and preprocess
                processedImages.Add(preprocessInput(image_processor.img_to_array(image)));
            }

            // to numpy array
            var arr = numpy.array(processedImages);

            // call model predict and flatten value to 1-dim array
            var prediction = model.predict(arr);
            foreach(var x in prediction) {
                result.Add(x.flatten());
            }
            return result;
        }

        static void showHits(List<dynamic> imageList, dynamic srcImage, dynamic hits){
            var fig = plt.figure(); //figsize: new int[] {9, 10});
            var top_k = 8;
            var i = 1;

            fig.add_subplot(3, 3, 1);
            // show original image
            plt.title("query image");
            plt.imshow(srcImage);
            
            foreach(var hit in hits) {
                if (++i > top_k + 1) break;
            
                fig.add_subplot(3, 3, i);
                plt.title("match " + hit);
                plt.imshow(imageList[int.Parse(hit.ToString())]);
            }
            plt.show();
        }
        static void Main(string[] args)
        {
            // // config path for ready-to-use python and dependencies            
            // var pythonPath = @"..\python36;..\python36\Scripts;";
            // var pythonHome = @"..\python36";
            // var pythonLib = @"..\python36\Lib";
            
            // // if PYTHON path field not in path then:
            // if (!Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Machine).Contains(pythonPath)) {
            //     string path = pythonPath + Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Machine);
            //     Environment.SetEnvironmentVariable("PATH", path, EnvironmentVariableTarget.Process);
            // }
            // Environment.SetEnvironmentVariable("PYTHONHOME", pythonHome, EnvironmentVariableTarget.Process);
            // Environment.SetEnvironmentVariable("PYTHONPATH ", pythonLib, EnvironmentVariableTarget.Process);            
            using (Py.GIL())
            {
                Init();
                var path = currentDir + @"\test-data\";                              

                // load image to list
                var imageList = ImportData(path);

                // featurize data:
                var featurizedData = Featurize(imageList);
                
                // build index
                dynamic index = annoy.AnnoyIndex(62720, "angular");

                int ID = 0;
                foreach(var imageFeature in featurizedData) {
                    index.add_item(ID++, imageFeature);
                }

                // build 6 trees
                index.build(6);

                // search for image
                var imageToSearch = featurizedData[0];

                var hits = index.get_nns_by_vector(imageToSearch, 5, search_k: 3, include_distances: false);
                Console.WriteLine("hits: " + String.Join(", ", hits));

                showHits(imageList, imageList[0], hits);
            }
        }
    }
}
