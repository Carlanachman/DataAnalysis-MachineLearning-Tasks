import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        def sigmoid(z):
          return 1/(1+np.exp(-z))

        def cost_func( s, y):
          m = y.shape[0]
          return -1/m *np.sum(y* np.log(s + 1e-15) +(1-y)* np.log(1 -s + 1.e-15))

        np.random.seed(self.random_state)
        m, n = X.shape

        X_biais = np.hstack([np.ones((m, 1)), X])

        self.theta = np.random.randn(n+1)
        self.thetas.append(self.theta.copy())

        for _ in range(self.n_iter):
          z = X_biais @ self.theta
          s = sigmoid(z) ##

          #cost
          cost = cost_func(s, y) ##
          self.Js.append(cost)

          #gradient
          gradient = (1/m) * X_biais.T @ (s-y)

          self.theta -= self.eta * gradient
          self.thetas.append(self.theta.copy())


          if len(self.Js) > 1 and abs(self.Js[-2]- self.Js[-1]) < self.eps:
            break


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        def sigmoid( z):
          return 1/(1+np.exp(-z))


        m =X.shape[0]
        X_biais = np.hstack([np.ones((m, 1)), X])
        probs = sigmoid(X_biais @ self.theta)
        preds = (probs >= 0.5).astype(int)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
      a = i*fold_size
      b = (i+1)* fold_size if i< (folds -1) else X.shape[0]

      X_val = X_shuffled[a:b]
      y_val = y_shuffled[a:b]

      X_train = np.concatenate((X_shuffled[:a], X_shuffled[b:]), axis =0)
      y_train = np.concatenate((y_shuffled[:a], y_shuffled[b:]), axis =0)

      #new instnce of the classifier
      model = algo
      model.fit(X_train, y_train)
      y_pred = model.predict(X_val)
      acc = np.mean(y_pred == y_val)
      accuracies.append(acc)

    cv_accuracy = np.mean(accuracies)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    c = 1 / (sigma* np.sqrt(2* np.pi))
    e = -0.5 * ((data - mu) / sigma) **2
    p = (c * np.exp(e)).flatten()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n = data.shape[0]
        self.weights = np.full(self.k, 1/self.k)
        self.mus =np.random.choice(data.flatten(), self.k)
        self.sigmas = np.full(self.k, np.std(data))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        N = data.shape[0]
        self.responsibilities = np.zeros((N, self.k))

        for i in range(self.k):
          pdf = norm_pdf(data, self.mus[i], self.sigmas[i])
          self.responsibilities[:,i] = self.weights[i] * pdf

        sum_resp = np.sum(self.responsibilities, axis=1, keepdims=True) + 1e-15
        self.responsibilities /= sum_resp
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        N = data.shape[0]

        for i in range(self.k):
          resp = self.responsibilities[:,i]
          total = np.sum(resp)

          
          self.mus[i] =np.sum(resp* data.flatten()) / total
          self.sigmas[i] = np.sqrt(np.sum(resp * (data.flatten()- self.mus[i])**2) / total)
          self.weights[i] = total / N

          

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        def cost_func2(data):
          total = np.zeros(data.shape[0])

          for i in range(self.k):
            pdf = norm_pdf(data, self.mus[i], self.sigmas[i])
            total += self.weights[i] * pdf
          log_lik = np.log(total +1e-15)
          return -np.sum(log_lik)

        self.costs = []
        self.init_params(data)

        for i in range(self.n_iter):
          self.expectation(data)
          self.maximization(data)
          cost = cost_func2(data)
          self.costs.append(cost)

          if i>0 and abs(self.costs[-2] - self.costs[-1] ) < self.eps:
            break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    k = len(weights)
    total = np.zeros(data.shape[0])

    for i in range(k):
      total += weights[i] * norm_pdf(data, mus[i], sigmas[i])

    pdf = total  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    
    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.class_params = None
        self.classes = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        np.random.seed(self.random_state)
        
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        self.prior = {}
        for class_label in self.classes:
            self.prior[class_label] = np.sum(y == class_label) / len(y)
        
        self.class_params = {}
        
        for class_label in self.classes:
            self.class_params[class_label] = {}
            
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            for feature_idx in range(n_features):
                feature_data = X_class[:, feature_idx].reshape(-1, 1)
                
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(feature_data)
                
                weights, mus, sigmas = em.get_dist_params()
                self.class_params[class_label][feature_idx] = {
                    'weights': weights,
                    'mus': mus,
                    'sigmas': sigmas
                }        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_samples, n_features = X.shape
        predictions = []
        
        for sample_idx in range(n_samples):
            class_posteriors = {}
            
            for class_label in self.classes:
                posterior = self.prior[class_label]
                
                for feature_idx in range(n_features):
                    feature_value = X[sample_idx, feature_idx].reshape(-1, 1)
                    
                    weights = self.class_params[class_label][feature_idx]['weights']
                    mus = self.class_params[class_label][feature_idx]['mus']
                    sigmas = self.class_params[class_label][feature_idx]['sigmas']
                    
                    likelihood = gmm_pdf(feature_value, weights, mus, sigmas)
                    posterior *= likelihood[0]  
                
                class_posteriors[class_label] = posterior
            
            predicted_class = max(class_posteriors, key=class_posteriors.get)
            predictions.append(predicted_class)
        
        preds = np.array(predictions)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps, visualize_func=None):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps, random_state=1)
    lor_model.fit(x_train, y_train)
    
    lor_train_predictions = lor_model.predict(x_train)
    lor_test_predictions = lor_model.predict(x_test)

    lor_train_acc = np.mean(lor_train_predictions == y_train)
    lor_test_acc = np.mean(lor_test_predictions == y_test)
    
    visualize_func(x_train, y_train, lor_model, title='Logistic Regression Decision Boundary')

    
    plt.plot(range(len(lor_model.Js)), lor_model.Js)
    plt.title('Logistic Regression: Cost vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()


    bayes_model = NaiveBayesGaussian(k=k, random_state=1)
    bayes_model.fit(x_train, y_train)
    
    bayes_train_predictions = bayes_model.predict(x_train)
    bayes_test_predictions = bayes_model.predict(x_test)
    
    bayes_train_acc = np.mean(bayes_train_predictions == y_train)
    bayes_test_acc = np.mean(bayes_test_predictions == y_test)

    visualize_func(x_train, y_train, bayes_model, title='Naive Bayes Decision Boundary')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    np.random.seed(42)
    n_samples_per_class = 1000
    
    
    mean_0_a = [0, 0, 0]
    mean_1_a = [2, 2, 2]
    cov_a = [[1, 0, 0],    
             [0, 1, 0], 
             [0, 0, 1]]
    
   
    class_0_a = multivariate_normal.rvs(mean_0_a, cov_a, n_samples_per_class)
    class_1_a = multivariate_normal.rvs(mean_1_a, cov_a, n_samples_per_class)
    
    dataset_a_features = np.vstack([class_0_a, class_1_a])
    dataset_a_labels = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    

    mean_0_b = [1, 1, 1]
    mean_1_b = [0, 0, 0]
    cov_b = [
    [1, 0.99, 0.99],
    [0.99, 1, 0.99],
    [0.99, 0.99, 1]
  ]
    
    class_0_b = multivariate_normal.rvs(mean_0_b, cov_b, n_samples_per_class)
    class_1_b = multivariate_normal.rvs(mean_1_b, cov_b, n_samples_per_class)
    
    dataset_b_features = np.vstack([class_0_b, class_1_b])
    dataset_b_labels = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    idx_a = np.random.permutation(len(dataset_a_features))
    dataset_a_features = dataset_a_features[idx_a]
    dataset_a_labels = dataset_a_labels[idx_a].astype(int)
    
    idx_b = np.random.permutation(len(dataset_b_features))
    dataset_b_features = dataset_b_features[idx_b]
    dataset_b_labels = dataset_b_labels[idx_b].astype(int)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }