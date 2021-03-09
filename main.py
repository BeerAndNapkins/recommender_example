from sklearn.datasets import _samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def data_generation():
    # Generate data
    X, y = _samples_generator.make_classification(n_samples=150, n_features=25, n_classes=3, n_informative=6,
                                                  n_redundant=0, random_state=7)
    # Select top K features
    k_best_selector = SelectKBest(f_regression, k=9)
    # Initialize Extremely Random Forests classifer
    classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)
    # Construct the pipeline
    processor_pipeline = Pipeline([('selector', k_best_selector), ('erf', classifier)])
    # Set the parameters
    processor_pipeline.set_params(selector__k=7, erf__n_estimators=30)
    # Training the pipeline
    processor_pipeline.fit(X, y)
    # Predict outputs for the input data
    output = processor_pipeline.predict(X)
    print("\nPredicted output:\n", output)
    # Print scores
    print("\nScore:", processor_pipeline.score(X, y))
    # Print the features chosen by the pipeline selector
    status = processor_pipeline.named_steps['selector'].get_support()
    # Extract and print indices of selected features
    selected = [i for i, x in enumerate(status) if x]
    print("\nIndices of selected features:", ', '.join([str(x) for x in selected]))

def extract_neighbors():
    #Input data
    X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], [7.3, 2.1],
               [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9], [5.7, 3.5], [6.1, 4.3],
               [5.1, 2.2], [6.2, 1.1]])
    #Number of nearest neighbors
    k = 5
    #Test datapoint
    test_datapoint = [4.3, 2.7]
    #Plot input data
    plt.figure()
    plt.title('Input data')
    plt.scatter(X[:,0], X[:,1], marker='o', s=75, color='black')
    #Build K nearest Neighbors model
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = knn_model.kneighbors([test_datapoint])
    #Print the 'k' nearest neighbors
    print("\nK Nearest Neighbors:")
    for rank, index in enumerate(indices[0][:k], start=1):
        print(str(rank) + " ==> ", X[index])
    #Visualize the nearest neighbors along with the test datapoint
    plt.figure()
    plt.title('Neartest neighbors')
    plt.scatter(X[:,0], X[:, 1], marker='o', s=75, color='k')
    plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1], marker='o', s=250, color='k', facecolors='none')
    plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', s=75, color='k')
    plt.show()

def knn_classifier():
    #Load input data
    input_file = 'data.txt'
    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    #Plot input data
    plt.figure()
    plt.title('Input data')
    marker_shapes = 'v^os'
    mapper = [marker_shapes[i] for i in y]
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=75, edgecolor='black', facecolors='none')
    #Define nearest neighbors
    num_neighbors = 12
    step_size = 0.01
    #Create a K Nearest Neighbor classifier model
    classifier = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
    #Train the K Nearest Neighbors model
    classifier.fit(X, y)
    #Create the mesh to plot the boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    #Evaluate the classifier on all the points in the grid
    output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    #Visualize the predicted output
    output = output.reshape(x_values.shape)
    plt.figure()
    plt.pcolormesh(x_values, y_values, output, cmap=cm.Paired)
    #Overlay the training points on the map
    for i in range(X.shape[0]):
        plt.scatter(X[i,0], X[i, 1], marker=mapper[i], s=50, edgecolors='black', facecolors='none')

    plt.xlim(x_values.min(),x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.title('K Nearest Neighbors classifier model boundaries')
    #Test input datapoint
    test_datapoint = [5.1, 3.6]
    plt.figure()
    plt.title('Test datapoint')
    for i in range(X.shape[0]):
        plt.scatter(X[i,0], X[i, 1], marker=mapper[i], s=75, edgecolors='black', facecolors='none')
    plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidth=6, s=200, facecolors='black')

    #Extract the K nearest neighbors
    _, indices = classifier.kneighbors([test_datapoint])
    indices = indices.astype(np.int)[0]

    #Plot k nearest neighbors
    plt.figure()
    plt.title('K Nearest Neighbors')
    for i in indices:
        plt.scatter(X[i, 0], X[i,1], marker=mapper[y[i]], linewidth=3, s=100, facecolors='black')
    plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidth=6, s=200, facecolors='black')
    for i in range(X.shape[0]):
        plt.scatter(X[i,0], X[i, 1], marker=mapper[i], s=75, edgecolors='black', facecolors='none')
    print("Predicted output:", classifier.predict([test_datapoint])[0])
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #data_generation()
    #extract_neighbors()
    knn_classifier()


