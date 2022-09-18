# AWAD Android Application

<img src="https://github.com/wasdwasd0105/awad_tflite_android/raw/master/pics/AWAD_demo.png"/>

AWAD_Tflite is an application that can remove artifact wareform using pre-trained ML model. It uses TF-lite for the ML interface and can run on Android mobile platform.

This application is made by Kivy, and it is an open-source framework that can directly run python programs on Android Phones. The benefit of the framework is that it is compatible with most of the prototype code written in python, and some components, such as the data loader, can be natively run on Android phones with slight modification. 

Like most applications, this app is divided into front and back end. The front is a set of UI components that users can interact with all of the input and output operations. The front end can allow users to select the available case and model to run, display the predicted graph result, and change the graph’s shape and length. Also, it will save the user’s input options in the memory for the back end.

The back end is the place to handle case data and return the predicted result. First, it will load the user-selected case data into the data loader, the data loader will divide and sort the data into the appropriate format to fit the ML model. Then the ML program will be run to get the prediction. The result will be stored, and after finishing predicting all the parts, the predicted result will be transferred to the front end, and the front-end UI components will plot the graph based on the results.

<img src="https://github.com/wasdwasd0105/awad_tflite_android/raw/master/pics/AWAD_structure.jpg"/>


## Tensorflow lite and model convert

In the program, it uses TensorFlow Lite as the ML provider. The TensorFlow Lite is designed and optimized for mobile platforms such as Android phones, and it is compatible with the Kivy framework. 

The model is trained with Pytorch, and to run the model on the TensorFlow Lite runtime, we need to convert the model from Pytorch to TensorFlow Lite. 

The overall procedure is Pytorch -> ONNX -> TensorFlow -> TensorFlow Lite. First, we need to convert PyTorch to ONNX format. ONNX, Open Neural Network Exchange, is “an open format built to represent machine learning models.” It can convert different kinds of ML models into one uniform standard one. Second, export the ONNX format to TensorFlow. Third, convert TensorFlow to TensorFlow lite. TensorFlow can be directly converted to TensorFlow lite by the built-in TFLiteConverter on the TensorFlow library.

<img src="https://github.com/wasdwasd0105/awad_tflite_android/raw/master/pics/AWAD_model_convert.jpg"/>

Compiling from sorce
---------------
1. download soource code
    ```
    git clone https://github.com/wasdwasd0105/awad_tflite_android
    ```

2. apply patch for ndk to support Fortran
    ```
    https://github.com/mzakharo/android-gfortran
    ```

3. install python package
    ```
    pip3 install buildozer kivy kivymd scipy pandas numpy tflite-runtime kivy_garden.graph pillow
    ```
4. check dependency
   https://buildozer.readthedocs.io/en/latest/installation.html#targeting-android

5. compile and run
    ```
    buildozer android deploy run logcat
    ```
    
Installation
---------------

1. download from release page or compile the code
2. transmit the apk package and allow sideload
   https://www.howtogeek.com/313433/how-to-sideload-apps-on-android/


Dependency
---------------

- [Kivy](https://github.com/kivy/kivy): The Open Source Python App development Framework.
- [Python for Android](https://github.com/kivy/python-for-android): toolchain
  for building and packaging Python applications for Android.
- [Buildozer](https://github.com/kivy/buildozer): generic Python packager
  for Android and iOS.
- [Kivymd](https://github.com/kivymd/KivyMD): KivyMD is a collection of Material Design compliant widgets for use with Kivy.
- [scipy](https://github.com/scipy/scipy): open-source software for mathematics, science, and engineering.
- [pandas](https://pandas.pydata.org/): data analysis and manipulation tool

- [numpy](https://numpy.org/): The fundamental package for scientific computing with Python
- [tflite-runtime](https://www.tensorflow.org/lite): running machine learning models on mobile and embedded devices
- [Garden](https://github.com/kivy-garden): widgets and libraries created and
  maintained by users.
- [pillow](https://github.com/python-pillow/Pillow): The Python Imaging Library adds image processing capabilities to your Python interpreter.





