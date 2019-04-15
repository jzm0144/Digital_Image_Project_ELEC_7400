#%load_ext autoreload
#%autoreload 2

import warnings
warnings.simplefilter('ignore')

# Only for osX
import matplotlib
matplotlib.use('TkAgg')
#%matplotlib inline  

import imp
import numpy as np
import os
import ipdb

import keras
import keras.backend
import keras.models
from keras.models import load_model

from keras import backend as K
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import Adam

import innvestigate
import innvestigate.utils as iutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from importlib import util

# Use utility libraries to focus on relevant iNNvestigate routines.
#eutils = imp.load_source("utils", "../utils.py")
spec = util.spec_from_file_location("utils", "utils.py")
eutils = util.module_from_spec(spec)
spec.loader.exec_module(eutils)
#mnistutils = imp.load_source("utils_mnist", "../utils_mnist.py")
spec = util.spec_from_file_location("utils_mnist", "utils_mnist.py")
mnistutils = util.module_from_spec(spec)
spec.loader.exec_module(mnistutils)


def test_model(model, data, batch_size=128, epochs=20):
    num_classes = 10

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])
    score = model.evaluate(x_test, y_test, verbose=0)
    return score
########### Data ##################################
# Load the dataset and keep some images from the test set for the analysis.

# Load data
# returns x_train, y_train, x_test, y_test as numpy.ndarray
data_not_preprocessed = mnistutils.fetch_data()

# Create preprocessing functions
input_range = [-1, 1]
preprocess, revert_preprocessing = mnistutils.create_preprocessing_f(data_not_preprocessed[0], input_range)

# Preprocess data
data = (
    preprocess(data_not_preprocessed[0]), data_not_preprocessed[1],
    preprocess(data_not_preprocessed[2]), data_not_preprocessed[3]
)

num_classes = len(np.unique(data[1]))
label_to_class_name = [str(i) for i in range(num_classes)]

########## Model #####################################
# The next part trains and evaluates a CNN.

# Create & train model
if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)
'''
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, name='conv1'),
    keras.layers.Conv2D(64, (3, 3), activation="relu", name='conv2'),
    keras.layers.MaxPooling2D((2, 2), name='pool1'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu", name='dense1'),
    keras.layers.Dense(10, activation="softmax", name='dense2'),
])
# Train only first time and save the model
scores = mnistutils.train_model(model, data, batch_size=128, epochs=20)
model.save("model.h5")
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
print("Model trained and saved")
ipdb.set_trace()
'''
# Reuse the model
model = load_model('model.h5') 
scores = test_model(model, data, batch_size=128, epochs=20)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

########## Analysis ##################################
'''
Next, we will set up a list of analysis methods by preparing tuples containing 
the methods' string identifiers used by 
innvestigate.analyzer.create_analyzer(...),
some optional parameters, a post processing choice for visualizing the computed
analysis and a title for the figure to render. Analyzers can be deactivated by 
simply commenting the corresponding lines, or added by creating a new tuple as 
below.

For a full list of methods refer to the dictionary 
investigate.analyzer.analyzers
'''
# Scale to [0, 1] range for plotting.
def input_postprocessing(X):
    return revert_preprocessing(X) / 255

noise_scale = (input_range[1]-input_range[0]) * 0.1
ri = input_range[0]  # reference input


# Configure analysis methods and properties
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

    # Show input
    ("input",                 {},                       input_postprocessing,      "Input"),
    # Function
    ("input_t_gradient",      {},                       mnistutils.heatmap,        "Input * Gradient"),
 	#("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP"),
]

'''
The main loop below will now instantiate the analyzer objects based on the 
loaded/trained model and the analyzers' parameterizations above.
'''

# Create model without trailing softmax
model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

# Create analyzers.
analyzers = []
for method in methods:
    analyzer = innvestigate.create_analyzer(
        method[0],                     # analysis method identifier
        model_wo_softmax,              # model without softmax output
        neuron_selection_mode="index", # We want to select the output neuron to analyze.
        **method[1])                   # optional analysis parameters

    # Some analyzers require training.
    analyzer.fit(data[0], batch_size=256, verbose=1)
    analyzers.append(analyzer)

# Now we analyze each image with different analyzers on all 
# output neurons
n = 10
test_images = list(zip(data[2][:n], data[3][:n]))

for image_nr, (x, y) in enumerate(test_images):
    # Add batch axis.
    x = x[None, :, :, :]

    analysis = np.zeros([5, len(analyzers), 28, 28, 3])
    text = []

    for ii, output_neuron in enumerate([1, 3, 5, 6, 7]): #range(num_classes):
        # Predict final activations, probabilites, and label.
        presm = model_wo_softmax.predict_on_batch(x)[0]
        prob = model.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(("%s" % label_to_class_name[y],    # ground truth label
                     "%.4f" % presm[output_neuron],    # pre-softmax logits
                     "%.4f" % prob[output_neuron],     # probabilistic softmax output  
                     "%s" % label_to_class_name[output_neuron]
                    ))

        for aidx, analyzer in enumerate(analyzers):
            # Analyze.
            a = analyzer.analyze(x, neuron_selection=output_neuron)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = mnistutils.postprocess(a)
            # Apply analysis postprocessing, e.g., creating a heatmap.
            a = methods[aidx][2](a)
            # Store the analysis.
            analysis[ii, aidx] = a[0]

    print("-"*80)
    print("Image nr. {}: ".format(image_nr))
    # Prepare the grid as rectengular list
    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('Label: {}'.format(label[i]), 
    	                '\nClamped neuron: {}'.format(pred[i])) 
                        for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 
    	                 'Prob: {}'.format(prob[i])) 
                         for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]

    # Plot the analysis
    #file_name = os.environ.get("PLOTFILENAME", None)
    #if file_name is not None:
    #    file_name = ".".join(file_name.split(".")[:-1])+("_%i" % output_neuron)+file_name.split(".")[-1]
    file_name = str(image_nr)+'.png'
    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels, file_name=file_name)

'''
Each column shows the visualized results for different analyzers and each row
shows the analyses w.r.t. to one input sample. To the left of each row, the 
ground truth label label and the predicted label pred are show. To the right, 
the model's probabilistic (softmax) output is shown as prob and the logit 
output just before the terminating softmax layer as logit. Note that all 
analyses have been performed based on the logit output (layer).
'''