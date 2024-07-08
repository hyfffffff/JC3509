import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

"""
    Common functions used throughout the program
"""
def softmax(x):
    # Apply the softmax function to the input array x.
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def to_one_hot(y, num_classes=3):
    # Convert an array of labels (y) into a one-hot encoded matrix.
    # num_classes is the total number of categories for the one-hot encoding.
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y - 1] = 1
    return one_hot

def from_one_hot(one_hot_matrix):
    # Retrieve the original class labels from a one-hot encoded matrix.
    # The class labels are adjusted by adding 1 because classes start from 1.
    classes = np.argmax(one_hot_matrix, axis=1) + 1  
    return classes


"""
Task 1 - Data Preparation
"""

def preprocess_data(file_name, show_data_flag=True):
    """
    Preprocess the data from a CSV file.
    
    This function reads data from a CSV file into a pandas DataFrame,
    splits the data into training and testing sets,
    converts the sets to numpy arrays, and applies normalization.
    
    :param file_path: The path to the CSV file containing the data.
    :return: A tuple containing the normalized training and testing data as numpy arrays.
    """

    df = pd.read_csv(file_name, sep=',')

    # To verify the file's correct loading, check the resulting pandas DataFrame for any columns 
    # classified as 'object'. The presence of 'object' type columns suggests illegal or unexpected
    # characters in the original file, necessitating data cleanup. Additionally, ensure the row 
    # count in the DataFrame aligns with the file's line count to confirm complete data import.

    # Check the data types of each column in the DataFrame
    column_types = df.dtypes

    # Remove leading and trailing spaces from all column names
    df.columns = df.columns.str.strip()

    # Check if any column is of type 'object'
    if (column_types == 'object').any():
        print("There are columns with data type 'object'.")
    else:
        # If no column of type 'object' exists, print the number of rows in the DataFrame
        print(f"No columns of type 'object' found. The DataFrame has {len(df)} rows.")

    # Visualize the data to confirm it has been correctly loaded.
    if show_data_flag:
        showdata(df,[])

    # Obtain the feature names of the data."
    features_name = df.columns.tolist()[1:]

    X = df.iloc[:,1:]    # X contains the features
    y = df.iloc[:,0]     # y contains the labels

    # Split the feature data and labels into training and validation parts
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Convert all training and validation data to numpy arrays for subsequent training and validation
    X_train, X_test, y_train, y_test = df_X_train.to_numpy(), df_X_test.to_numpy(), df_y_train.to_numpy(), df_y_test.to_numpy()

    # Calculate the L2 norm of the training features along each feature axis
    norms = np.linalg.norm(X_train, axis=0)

    # Normalize training and test feature data by the calculated norms
    X_train, X_test = X_train/norms, X_test/norms

    # Convert training and test labels to one-hot encoded format
    y_train, y_test = to_one_hot(y_train), to_one_hot(y_test)

    return X_train, X_test, y_train, y_test, features_name, df

def showdata(df, sortedindex):
    """
    This function visualizes data in a DataFrame df as a series of histograms. 
    It takes two parameters: df, the DataFrame containing the data, 
    and sortedindex, a list specifying the order in which columns should be plotted. 
    """

    # If sortedindex is empty, generate a default index list based on the number of columns (excluding the first one)
    if len(sortedindex)==0:
        colcount = len(df.columns) - 1
        sortedindex = list(range(colcount))

    # Create a figure with a specified size
    plt.figure(figsize=(15,9))
    sns.set_style('whitegrid')  # Set the Seaborn style to 'whitegrid'
    sns.set_palette('deep')     # Set the Seaborn color theme to 'deep' for a wide range of hues

    # Loop through each column (starting from the second) to create a histogram for each
    for i, column in enumerate(df.columns[1:]):
        plt.subplot(3,5,i+1)
        # Plot the histogram using seaborn. 'hue' categorizes data, 'kde' adds a density curve, and 'bins' specifies the number of bins
        sns.histplot(data=df, x=df.columns[sortedindex[i]+1], hue='Producer', kde=True, bins=5)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

"""
Task 2 - Model Construction
Using object-oriented programming, the neural network is structured around four core classes: 
`BasicLayer`, `ReLULayer`, `SoftmaxCrossEntropyLoss`, and `FullyConnectedANN`. The `BasicLayer`
is designed for linear transformations, making it versatile for use as input, hidden, or output
layers. The `ReLULayer` introduces non-linearity with the ReLU function. The `SoftmaxCrossEntropyLoss` 
manages loss calculations, and the `FullyConnectedANN` brings these elements together, enabling 
essential network functionalities like forward/backward passes and gradient descent for training 
and validation.
"""

class BasicLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases with appropriate shapes, using He initialization for weights
        self.W = np.random.randn(input_size, output_size) * (2/input_size) ** 0.5
        self.b = np.random.rand(output_size)

        self.input = None  # Placeholder for storing input data
        self.gradient_W = None  # Placeholder for weight gradients
        self.gradient_b = None  # Placeholder for bias gradients

    def forward(self, input_data):
        # Forward pass: compute weighted sum of inputs + bias
        self.input = input_data
        return np.dot(input_data, self.W) + self.b

    def backward(self, output_gradient):
        # Backward pass: compute gradients of loss w.r.t weights and biases
        self.gradient_W = np.dot(self.input.T, output_gradient)
        self.gradient_b = np.sum(output_gradient, axis=0)
        return np.dot(output_gradient, self.W.T) # Return gradient with respect to inputs for the next layer
    
class ReLULayer:
    def __init__(self):
        self.input = None # Placeholder for storing input data
        
    def forward(self, input_data):
         # Apply ReLU activation: max(0, x)
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        # Gradient of ReLU: pass gradient where input > 0
        return output_gradient * (self.input > 0)


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.softmax_output = None  # Placeholder for softmax output
        self.y_true = None   # Placeholder for true labels

    def forward(self, logits, y_true):
        # Compute softmax output and cross-entropy loss
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        
        self.y_true = y_true
        epsilon = 1e-15   # Prevent log(0)
        clipped_output = np.clip(self.softmax_output, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(clipped_output)) / logits.shape[0]
        return loss

    def backward(self):
        # Compute gradient of loss w.r.t. softmax input (simplified due to softmax-crossentropy combination)
        d_logits = self.softmax_output - self.y_true
        return d_logits

class FullyConnectedANN:
    def __init__(self, layers, loss_function=SoftmaxCrossEntropyLoss(), learning_rate=5e-3, coeff=0.01, reg_type='None'):
        self.layers = layers    # List of layers in the neural network
        self.loss_function = loss_function    # Loss function to use
        self.learning_rate = learning_rate    # Learning rate for gradient descent
        self.coeff = coeff    # Regularization coefficient
        self.reg_type = reg_type   # Type of regularization: None, L1, or L2

    def forward(self, X):
        # Perform the forward pass through all layers
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_pred, y_true):
        # Compute the base loss using the loss function
        basic_loss = self.loss_function.forward(y_pred, y_true)
        
        # Initialize regularization loss
        reg_loss = 0
        # Compute regularization loss if applicable
        if self.reg_type == 'L2':
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    reg_loss += 0.5 * self.coeff * np.sum(layer.W ** 2)
        elif self.reg_type == 'L1':
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    reg_loss += self.coeff * np.sum(np.abs(layer.W))
        
        # Total loss = base loss + regularization loss
        total_loss = basic_loss + reg_loss
        return total_loss

    def backward(self):
        # Perform the backward pass to compute gradients
        loss_list = []   # Store gradients for analysis
        d_loss = self.loss_function.backward()   # Start with gradient from the loss function
        loss_list.append(d_loss)
        for layer in reversed(self.layers):     # Iterate backwards through layers
            d_loss = layer.backward(d_loss)
            loss_list.append(d_loss)
        
        # Apply gradient updates for regularization
        if self.reg_type == 'L2':
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    layer.gradient_W += self.coeff * layer.W
        elif self.reg_type == 'L1':
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    layer.gradient_W += self.coeff * np.sign(layer.W)

        dX = loss_list[-2].dot(self.layers[0].W.T)
        return dX
        

    def gradient_descent(self):
        # Iterate through each layer in the neural network
        for layer in self.layers:
            # Check if the layer has weights ('W'), indicating it's a layer whose parameters can be updated
            if hasattr(layer, 'W'):
                # Update the layer's weights ('W') by subtracting the product of the learning rate and the weight gradient
                layer.W -= self.learning_rate * layer.gradient_W
                # Update the layer's biases ('b') by subtracting the product of the learning rate and the bias gradient
                layer.b -= self.learning_rate * layer.gradient_b

def Producer_model(layersizes, learning_rate, coeff=1, reg_type='None'):
    """
    Generates a fully connected neural network model based on specified parameters.

    :param layersizes: A list of tuples, where each tuple represents the input size and output size of a layer in the network.
    :param learning_rate: The learning rate for the gradient descent optimization algorithm.
    :param coeff: Coefficient for regularization. 
    :param reg_type: Type of regularization to apply. Options are 'None', 'L1', or 'L2'. Default is 'None'.

    :return: An instance of a fully connected artificial neural network model.
    """

    # Initialize an empty list to hold the layers of the network
    layers = []
    # Iterate through the layer sizes (except the last one) to create BasicLayer and ReLULayer pairs
    for layer_parameter in layersizes[:-1]:
        # Add a BasicLayer with the specified input and output sizes
        layers.append(BasicLayer(layer_parameter[0], layer_parameter[1]))
        # Add a ReLULayer for non-linear activation after each BasicLayer
        layers.append(ReLULayer())
    # The last layer: only add a BasicLayer without a ReLULayer following it
    layer_parameter = layersizes[-1]
    layers.append(BasicLayer(layer_parameter[0], layer_parameter[1]))

    # Create the fully connected neural network model with the specified layers, learning rate, regularization coefficient, and regularization type
    model = FullyConnectedANN(layers, learning_rate=learning_rate, coeff=coeff, reg_type=reg_type)
    return model

"""
Task 3 - Model Training
In this task, the neural network built in Task 2 is trained and validated with pre-processed data. 
The process involves training the model on a subset of the data, using mini-batches and gradient 
descent to optimize parameters, and validating its performance on a separate data subset. 
Regularization to prevent overfitting is achieved through methods integrated within the network's 
structure from Task 2, enhancing the model's generalization to new data.
"""
    
def load_mini_batches(X, Y, batch_size=10):
    """
    Splits data into mini-batches for training. Generates mini-batches using an iterator. Helps manage large datasets by
    dividing them into smaller batches, improving memory use and training iteration.
    """
    row_count = X.shape[0]  # Total number of samples in the dataset

    for start in range(0, row_count, batch_size): 
        end = min(start + batch_size, row_count)   # Ensure the batch does not exceed dataset size
        yield X[start:end], Y[start:end]    # Yield consecutive mini-batches

def train_producer(model, X_train, y_train, epoch, batch_size=10):
    """
    Trains the model on the training dataset.
    """

    # Initialize variables for tracking training progress
    loss_history = []     # Stores loss for each batch
    accuracy_history = []  # Stores accuracy for each batch
    batch_idx = 0  # Batch counter
    accumulated_grads = np.zeros((X_train.shape[1]))  # Accumulator for gradients 
    correct_sum = 0    # Sum of correctly predicted samples
    train_loss = 0     # Total training loss

    for x0, y0 in load_mini_batches(X_train, y_train, batch_size):

        logits = model.forward(x0)   # Forward pass
        loss = model.compute_loss(logits, y0)   # Compute loss

        dX = model.backward()   # Backward pass (calculates gradients)
        accumulated_grads += np.sum(dX, axis=0)   # Update weights
        model.gradient_descent()    # Update weights

        # Convert predictions to labels and calculate accuracy
        Y_hat_softmax = softmax(logits)
        pred = from_one_hot(Y_hat_softmax)
        correct = np.sum(pred==from_one_hot(y0))
        correct_sum += correct

        # Update histories for loss and accuracy
        loss_history.append(loss)
        accuracy_history.append(correct / len(x0))

        # Accumulate the cumulative loss for this epoch
        train_loss += loss * len(x0)
        
        # Print progress for every `interval` batches
        batch_idx += 1
        interval = max(1, len(X_train) // len(x0) // 10)
        if batch_idx % interval == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss:0.2e} batch_acc={correct/len(x0):0.3f}"
            )

    return train_loss / len(X_train), correct_sum / len(X_train), loss_history, accuracy_history, accumulated_grads

def validate(model, X_test, y_test, batch_size=10):
    """
    Validates the model on a separate test dataset.
    """
    test_loss = 0   # Total loss on test data
    correct_sum = 0  # Sum of correctly predicted samples
    loss_history = []  # Stores loss for each batch
    accuracy_history = []  # Stores accuracy for each batch

    for x0, y0 in load_mini_batches(X_test, y_test, batch_size):

        logits = model.forward(x0)  # Forward pass

        loss = model.compute_loss(logits, y0)    # Compute loss

        # Accumulate the cumulative loss for this epoch
        test_loss += loss * len(x0)

        # Convert predictions to labels and calculate accuracy
        Y_hat_softmax = softmax(logits)
        pred = from_one_hot(Y_hat_softmax)
        correct = np.sum(pred==from_one_hot(y0))
        correct_sum += correct

        # Update histories for loss and accuracy
        loss_history.append(loss)
        accuracy_history.append(correct / len(x0))

    # Calculate and print test loss and accuracy
    test_loss /= len(X_test)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct_sum,
            len(X_test),
            100.0 * correct_sum / len(X_test),
        )
    )
    return test_loss, correct_sum / len(X_test), loss_history, accuracy_history

"""
Task 4 - Evaluation
This task rigorously evaluates the trained neural network by analyzing loss and accuracy 
through epoch-wise and batch-wise visualizations. The epoch-wise plots provide insights 
into the overall learning trends, while the batch-wise plots reveal fluctuations and deviations 
within the training process.
"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def present_result(model, X_test, y_test):
    """
    Evaluates model performance using key metrics: accuracy, precision, recall, F1 score, 
    and plots confusion matrix, ROC curves. 
    """

    # Forward pass through the model to get logits
    logits = model.forward(X_test)
    # Apply softmax to logits to get predicted probabilities
    y_pred = softmax(logits)

    # Generate and print classification report
    report = classification_report(from_one_hot(y_test), from_one_hot(y_pred))
    print(report)

    # Compute and display the confusion matrix
    matrix = confusion_matrix(from_one_hot(y_test), from_one_hot(y_pred))


    plt.figure(figsize=(10, 7))

    # Display confusion matrix using a heatmap
    plt.subplot(1,2,1)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted Producer')
    plt.ylabel('True Producer')
    plt.title('Confusion Matrix')

    plt.subplot(1,2,2)

    # Dictionaries to hold FPR, TPR, and AUC values for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate FPR, TPR, and AUC for each class
    n_classes = 3
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Producer Prediction')
    plt.legend(loc="lower right")
    plt.show()




def plot_model_performance(train_batch_loss, val_batch_loss, train_batch_accuracy, val_batch_accuracy,train_loss, val_loss, train_accurary, val_accuracy, num_epochs, model_info):
    """
    This function plots the performance of a model in terms of loss and accuracy for both training and validation datasets.
    """

    def plot_one_picture(train, val, x_label, y_label, num_epochs):
        """
        Inner function to plot a single metric (like loss or accuracy) for both training and validation datasets.
        """
        n_train = len(train)    # Number of training data points
        usealpha = False        # Flag to use alpha blending for the plot lines

        # Adjust the x-axis scale if the number of training points is different from the number of epochs
        if n_train!=num_epochs:
            t_train = num_epochs * np.arange(n_train) / n_train
            usealpha = True
        else:
            t_train = np.arange(n_train)

        n_val = len(val)    # Number of validation data points
        # Adjust the x-axis scale for validation data if necessary
        if n_val!=num_epochs:
            t_val = num_epochs * np.arange(n_val)/ n_val
        else:
            t_val = np.arange(n_val)

        # Plot training and validation data
        if usealpha:
            plt.plot(t_train, train, label="Train", alpha=0.5)
            plt.plot(t_val, val, label="Val", alpha=0.5)
        else:
            plt.plot(t_train, train, label="Train")
            plt.plot(t_val, val, label="Val")

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)


    # Calculate the number of training batches per epoch
    batch_count = len(train_batch_loss) // num_epochs

    # Calculate the volatility (max-min difference) of training loss for each epoch
    train_loss_volatility = [max(train_batch_loss[i:i+batch_count]) - min(train_batch_loss[i:i+batch_count]) for i in range(0, len(train_batch_loss), batch_count)]

    # Calculate the volatility (max-min difference) of training accuracy for each epoch
    train_accuracy_volatility = [max(train_batch_accuracy[i:i+batch_count]) - min(train_batch_accuracy[i:i+batch_count]) for i in range(0, len(train_batch_accuracy), batch_count)]

    # Recalculate the number of validation batches per epoch as it might differ from training batches
    batch_count = len(val_batch_loss) // num_epochs

    # Calculate the volatility (max-min difference) of validation loss for each epoch
    val_loss_volatility = [max(val_batch_loss[i:i+batch_count]) - min(val_batch_loss[i:i+batch_count]) for i in range(0, len(val_batch_loss), batch_count)]

    # Calculate the volatility (max-min difference) of validation accuracy for each epoch
    val_accuracy_volatility = [max(val_batch_accuracy[i:i+batch_count]) - min(val_batch_accuracy[i:i+batch_count]) for i in range(0, len(val_batch_accuracy), batch_count)]


    plt.figure(figsize=(15,9))   # Set the overall figure size

    plt.suptitle(model_info, fontsize=14, y=0.95)   # Add a super title to the figure with model information

    # Plot batch-wise loss for training and validation
    plt.subplot(2,3,1)
    plot_one_picture(train_batch_loss, val_batch_loss, 'epoch', 'batch loss', num_epochs)

    # Plot epoch-wise accuracy for training and validation
    plt.subplot(2,3,2)
    plot_one_picture(train_loss, val_loss, 'epoch', 'loss', num_epochs)

    # Plot loss volatility for training and validation
    plt.subplot(2,3,3)
    plot_one_picture(train_loss_volatility, val_loss_volatility, 'epoch', 'loss volatility', num_epochs)
    
    # Plot batch-wise accuracy for training and validation
    plt.subplot(2,3,4)
    plot_one_picture(train_batch_accuracy, val_batch_accuracy, 'epoch', 'accuracy', num_epochs)

    plt.subplot(2,3,5)
    plot_one_picture(train_accurary, val_accuracy, 'epoch', 'accuracy', num_epochs)

    # Plot accuracy volatility for training and validation
    plt.subplot(2,3,6)
    plot_one_picture(train_accuracy_volatility, val_accuracy_volatility, 'epoch', 'accuracy volatility', num_epochs)


def run_producer_training(num_epochs, lr, batch_size, layersizes, reg_type="None", coeff=1, plot_performace=True):
    """
    This program trains a neural network in two modes: The first displays training outcomes through loss and 
    accuracy graphs, aiding in model performance assessment. The second returns the input layer's parameters 
    and the first layer's gradients after training, focusing on identifying the most influential features. 
    This dual-mode approach allows for both a broad evaluation of the model's learning progress and a targeted 
    analysis of feature importance.
    """

    # Preprocess data: Load and split data, returning training and test sets, feature names, and the dataframe.
    X_train, X_test, y_train, y_test, features_name, df = preprocess_data('Assessment1_Dataset.csv', plot_performace)

    # Initialize the model with specified parameters
    model = Producer_model(layersizes, learning_rate=lr, coeff=coeff,reg_type=reg_type)

    # Initialize lists to store training and validation loss and accuracy, for both epoch-wise and batch-wise tracking
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    train_batch_loss_history = []
    train_batch_acc_history = []
    val_batch_loss_history = []
    val_batch_acc_history = []

    # Initialize an array to accumulate gradients, useful for analyzing feature importance.
    input_size = layersizes[0][0]
    accumulated_grads = np.zeros((input_size))

    # Loop through each epoch to train and validate the model.
    for epoch in range(1, num_epochs + 1):
        # Train the model for one epoch and collect training metrics and gradients.
        train_loss, train_acc, train_batch_loss, train_batch_acc, grads = train_producer(model, X_train, y_train, epoch, batch_size)
        # Update histories with current epoch's results.
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        train_batch_loss_history.extend(train_batch_loss)
        train_batch_acc_history.extend(train_batch_acc)
        # Accumulate gradients for feature importance analysis.
        accumulated_grads += grads

        # Validate the model using the test set and update validation histories.
        val_loss, val_acc, val_batch_loss, val_batch_acc = validate(model, X_test, y_test, batch_size)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_batch_loss_history.extend(val_batch_loss)
        val_batch_acc_history.extend(val_batch_acc)


    # After training, analyze and print the first layer's weights to understand initial feature importance.
    weights_first_layer = model.layers[0].W
    print(weights_first_layer)

    # Sort features based on the sum of the absolute values of weights, indicating their importance.
    importance = np.sum(weights_first_layer,axis=1)
    sorted_indices = np.argsort(importance)[::-1]

    # Sort gradients to identify which features had the most influence during training.
    sorted_grads = np.argsort(accumulated_grads)[::-1]

    # Print the indices of the most important features and gradients, and their corresponding names.
    print('Most important features:', sorted_indices)
    print([features_name[i] for i in sorted_indices])

    print('Most important grads:', sorted_grads)
    print([features_name[i] for i in sorted_grads])

    # If not plotting, return the sorted indices for features and gradients.
    if not plot_performace:
        return sorted_indices, sorted_grads
    
    present_result(model, X_test, y_test)
    
    # Construct a string summarizing the model's configuration.
    model_info = f"Learning rate {lr}, batch size {batch_size}, number of layers {len(layersizes)}"
    if reg_type!="None":    # Add regularization information if applicable.
        if reg_type=="L2":
            reg_name = "Ridge"
        else:
            reg_name = "Lasso"

        model_info +=  f", {reg_name} with with regularization coefficient \u03BB = {coeff}."
   
    # Plot the model's performance over time using the collected metrics.
    plot_model_performance(train_batch_loss_history, val_batch_loss_history, train_batch_acc_history, val_batch_acc_history, train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs, model_info)

    plt.show()

# Entry point of the script.
if __name__ == '__main__':
    # Define learning rate, batch size, number of epochs, and regularization coefficient.
    lr = 1e-2
    batch_size = 10
    num_epochs = 250
    coeff = 0.008

    # Flag to determine the mode of operation.
    find_factors = False
    
    # Define the architecture of the neural network: layer sizes from input to output.
    layersizes = [(13,128),(128,128), (128,3)]

    # Mode 1: Train the model and display performance metrics if not in feature analysis mode.
    if not find_factors:
        lr = 1e-2
        batch_size = 10
        num_epochs = 300
        coeff = 0
        layersizes = [(13,128),(128,128), (128,3)]
        run_producer_training(num_epochs, lr, batch_size, layersizes, reg_type="None", coeff=coeff)
        coeff = 0.08
        run_producer_training(num_epochs, lr, batch_size, layersizes, reg_type="L2", coeff=coeff)
        coeff = 0.008
        run_producer_training(num_epochs, lr, batch_size, layersizes, reg_type="L1", coeff=coeff)
    else:
        # Mode 2: Analyze feature importance by collecting input layer parameters and first layer gradients over multiple runs.
        features, grads = [], []
        # Run the training process multiple times to accumulate data on feature importance.
        for i in range(300):
            epoch_features, epoch_grads = run_producer_training(num_epochs, lr, batch_size, layersizes, reg_type="L1", coeff=coeff, plot_performace=False)
            features.append(epoch_features)
            grads.append(epoch_grads)

        # Prepare a matrix to aggregate gradient information for heatmap visualization.
        heatmap_matrix = np.zeros((13, 13))  # Assuming 13 features for simplification.
        score = np.arange(13, 0, -1)
        features_score = np.zeros((13,))

        # Aggregate gradient information across all runs.
        for grad in grads:
            for position, number in enumerate(grad):
                heatmap_matrix[number, position] += 1
                features_score[number] += score[position]

        # Display the features in descending order of their importance based on scores
        X_train, X_test, y_train, y_test, features_name, df = preprocess_data('Assessment1_Dataset.csv', False)
        showdata(df, np.argsort(-features_score))

        # Plot a heatmap to visualize feature importance based on gradients
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_matrix, annot=True, fmt='g', cmap='viridis')
        plt.title('Feature Position Heatmap')
        plt.xlabel('Position in list')
        plt.ylabel('Number')

        plt.show()
