---
title: "Bridging the Divide: Logistic Regression as a Fundamental Component of Neural Networks part-2"
description: "The article explores the fusion of logistic regression and neural networks in solving complex problems, using practical code examples."
pubDate: "Nov 4 2023"
heroImage: "/blog-2.webp"
tags: ["Deep Learning","Logistic Regression"]
---

So far, we have discussed that logistic regression is a fundamental algorithm for binary classification, simplifying complex data into clear “yes” or “no” decisions. In contrast, neural networks, with their intricate architecture, excel at handling complex tasks by recognizing intricate patterns within data. However, the true revelation lies in the integration of logistic regression within neural networks. While neural networks decode complex features, logistic regression at the output layer simplifies decision-making by assigning probabilities and making binary choices, creating a powerful synergy between simplicity and complexity in solving a wide array of problems.

## Incorporating Code with Explanations
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

Here, we load the dataset, which contains images of cats and non-cats, along with their corresponding labels. The dataset is split into training and test sets.

    # Example of a picture
    index = 25
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

This code snippet displays an example image from the dataset and prints its corresponding label. It helps us visualize the data and understand what the model will be classifying.

    # Extracting dataset dimensions
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

In these lines, we extract important dimensions of the dataset, such as the number of training and test examples (m_train and m_test) and the number of pixels in each image's width and height (num_px).

    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

Here, we reshape the training and test datasets into a flat format, where each example is represented as a 1D vector. This transformation is essential for inputting the data into the logistic regression model.

    # Implementing the sigmoid function
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        s = 1 / (1 + np.exp(-z))
        
        return s

In machine learning and neural networks, the sigmoid function is vital. It converts input values (in this case, the linear combination of weights and input data) into values between 0 and 1, which can be interpreted as probabilities. This function plays a crucial role in logistic regression and neural networks for binary classification.

    # Initializing weights and bias with zeros
    def initialize_with_zeros(dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias) of type float
        """
        
        w = np.zeros((dim, 1))
        b = 0.0

        return w, b

Before we start training a logistic regression model, we need to initialize the parameters. In this function, we initialize the weights w as a vector of zeros and the bias b as zero. These will be updated during training.

    # Propagation: Forward and Backward
    def propagate(w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        grads -- dictionary containing the gradients of the weights and bias
                (dw -- gradient of the loss with respect to w, thus same shape as w)
                (db -- gradient of the loss with respect to b, thus same shape as b)
        cost -- negative log-likelihood cost for logistic regression
        """
        
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        A = sigmoid(np.dot(w.T, X) + b)
        
        cost = -1/m * (np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1/m * np.dot(X, (A-Y).T)
        db = 1/m * np.sum(A-Y)
        
        cost = np.squeeze(cost)
        
        grads = {"dw": dw, "db": db}
        
        return grads, cost

This code implements the forward and backward propagation steps. Forward propagation calculates the activation A and the cost, while backward propagation computes the gradients of the cost function with respect to the weights dw and the bias db.

    # Optimization: Updating parameters with gradient descent
    def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        
        w = copy.deepcopy(w)
        b = copy.deepcopy(b)
        costs = []
        
        for i in range(num_iterations):
            grads, cost = propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            if i % 100 == 0:
                costs.append(cost)
            
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        
        return params, grads, costs

Here, we perform the optimization using gradient descent. We iterate through the training process, updating the weights and bias based on the computed gradients. This iterative process minimizes the cost function, helping our model make better predictions.

    # Making predictions
    def predict(w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        A = sigmoid(np.dot(w.T, X) + b)
        
        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
        
        return Y_prediction

After training the model, we use it to make predictions. The predict function takes the learned weights and bias to predict whether an input example belongs to class 0 or class 1, based on the sigmoid output.

    # Building the logistic regression model
    def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to True to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """
        
        w, b = initialize_with_zeros(X_train.shape[0])
        params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        w = params["w"]
        b = params["b"]
        
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
        
        if print_cost:
            train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
            test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
            print("train accuracy: {} %".format(train_accuracy))
            print("test accuracy: {} %".format(test_accuracy))
        
        d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train, "w": w, "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}
        
        return d

Finally, the model function orchestrates the entire process. It initializes parameters, optimizes them through gradient descent, and computes the accuracy of the model. This function serves as the entry point to train and evaluate a logistic regression model.

whether you’re taking your first steps into the world of AI or you’re a seasoned practitioner, logistic regression is a must-know technique that will undoubtedly be a part of your machine learning journey.