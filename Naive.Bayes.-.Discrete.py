
# coding: utf-8

# In[141]:

import numpy


# Convert string categorical features to number of categories

# In[142]:

def numerize_features(a):
    distinct = numpy.unique(a)
    column = numpy.zeros(len(a))
    distinct = distinct.tolist()
    #print loaded[:, 1] == distinct[1]
    for i in range(len(a)):
        column[i] = distinct.index(a[i])
    return column.astype(int)


# Preprocessing of the data given the url. Returns the X matrix with categorical features and corresponding labels in y

# In[143]:

def preprocessing(url):
    loaded = numpy.genfromtxt(url, delimiter = ';', dtype = str, skip_header = 1)
    loaded = numpy.random.permutation(loaded)
    X = numpy.column_stack((numerize_features(loaded[:, 1]), numerize_features(loaded[:, 2]), numerize_features(loaded[:, 3]), 
                        numerize_features(loaded[:, 4]), numerize_features(loaded[:, 6]), numerize_features(loaded[:, 7]), 
                        numerize_features(loaded[:, 6]), numerize_features(loaded[:, 9]), numerize_features(loaded[:, 10]), 
                        numerize_features(loaded[:, 15])))  
    y = numerize_features(loaded[:, 16])
    #return (X, y)
    return (X[0:22000, :], y[0:22000], X[22000:, :], y[22000:])


# In[144]:

#X, y = preprocessing("E:\Lecs\IIIT\SMAI\Assignments\Assignment 4\Bank Dataset\\bank-full.csv")
(X, y, XTest, yTest) = preprocessing("E:\Lecs\IIIT\SMAI\Assignments\Assignment 4\Bank Dataset\\bank-full.csv")


# Calculate prior probabilities for all classes

# In[146]:

prior = numpy.bincount(y)/float(len(y))


# Calculate the likelihood ratio for each feature given a class

# In[147]:

def calculate_likelihood(f, w):
    count = numpy.zeros((len(numpy.unique(f)), len(numpy.unique(w))))
    for i, j in zip(f, w):
        count[i][j] = count[i][j] + 1
        #print count
    individual_class = numpy.bincount(w)
    for i in count:
        i[0] = i[0]/float(individual_class[0])
        i[1] = i[1]/float(individual_class[1])
    return numpy.log(count)


# create the likelihood table

# In[148]:

count = numpy.zeros((len(numpy.unique(X[:, 0])), len(numpy.unique(y))))
likelihood = list()
for i in range(len(X[1, :])):
    likelihood.append(calculate_likelihood(X[:, i], y))




# Calculating the posterior probabilities

# In[149]:

def calculate_posterior(prior, likelihood, f, w):
    posterior = numpy.log(1)
    #print f
    for i in range(len(f)):
        #print likelihood[i][f[i]][w]
        posterior = posterior + likelihood[i][f[i]][w]
    return posterior + prior[w]


# The final predict function

# In[150]:

def Naive_Bayes_predict(prior, likelihood, f, w):
    posteriors = numpy.zeros((len(w)))
    for i in range(len(w)):
        posteriors[i] = calculate_posterior(prior, likelihood, f, i)
    return numpy.argmax(posteriors)


# Calculating the accuract and the confusion matrix

# In[151]:

def accuracy_and_confusion(XTest, yTest):
    count = 0
    confusion_matrix = numpy.zeros((len(numpy.unique(y)), len(numpy.unique(y))))
    for i in range(len(yTest)):
        predict = Naive_Bayes_predict(prior, likelihood, XTest[i, :], numpy.unique(y))
        if predict == yTest[i]:
            count = count + 1
        confusion_matrix[predict][yTest[i]] += 1
    return count/float(len(yTest)), confusion_matrix


# In[153]:

accuracy, confusion_matrix =  accuracy_and_confusion(XTest, yTest)


# In[ ]:



