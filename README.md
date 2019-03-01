# MNIST
Practice with TensorFlow and Keras. Initial solution to MNIST problem. Many areas for improvement 

My program is built with python3.6. Start virtual environment from inside project directory: source venv/bin/activate. Install dependencies: pip install -r requirements.txt. Once dependencies have been installed, you can run it like this: 

To train: ```python main.py```
To predict: ```python main.py --predict Handwritten-digit-2.png```

Training will save the architecture of the neural network as a JSON file and the weights at the end of training as a .h5 file. Predict loads these values from a default './model_architecture.json', and './model_weights.h5' locations. An area of improvement here could be a dynamic source and multiple saving and loading mechanisms. The input provided had extraneous text, so other examples were used. In the future, I would implement a region-of-interest algorithm so that the inputs do not have to be as clean.


## Input -> Flatten -> Dense -> Dropout -> Dense -> output

My neural network is a keras sequential model which is a simple linear layer stack. I have four layers. This setup was chosen because it is a common layer stack and minimizes overfitting. Flatten is applied first in order to flatten input to a 28-d vector so that when Dense is called (the next layer) logits and labels have the same first dimension. Then a Dense layer of size n*n (784 neurons) with relu activation because relu because I usually start with a relu activation (fewer vanishing gradient problems). Then a dropout layer in the middle of the two major dense layers to avoid overfitting. I’m pretty sure the way keras implements this is with the same number of neurons as the previous layer and a certain percentage (in this case 20%) activated. One more Dense layer with the size of the expected number of classes (10 neurons) and a softmax activation for simple probability resolution. I’m constantly reaching 98% accuracy. With more time, I would loosen constraints on input type and implement online learning capabilities. My loss function is sparse categorical cross entropy because our targets are integers, I optimized with an adam optimizer and my metric was accuracy because of the project requirements.
