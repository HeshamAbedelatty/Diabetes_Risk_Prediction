from tkinter import *
from tkinter import filedialog
from tkinter import *
from tkinter import filedialog, ttk
from tabulate import tabulate  # Import tabulate library
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_entry.delete(0, END)
    file_entry.insert(END, file_path)
def preprocess_data():
    file_path = file_entry.get()
    percentage = float(percentage_entry.get())
    k = int(TrainingEntry.get()) / 100

    # Calculate the total number of rows in the file
    total_rows = sum(1 for _ in open(file_path))

    # Calculate the number of rows to read
    nrows_to_read = int(total_rows * (percentage / 100))

    data = pd.read_csv(file_path, nrows=nrows_to_read)

    # Encode categorical variables
    data['gender'] = data['gender'].astype('category').cat.codes
    data['smoking_history'] = data['smoking_history'].astype('category').cat.codes

    # labels: gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes
    # Drop the 'bmi' column , 'blood_glucose_level', 'HbA1c_level', 'age'
    data = data.drop(columns=['bmi'])

    return data, k
# /////////////////////////////////////////Naive bayes Algorithm/////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////
# Function to calculate probability mass function (PMF) for discrete features
def calculate_pmf(x, feature_values):
    count = np.sum(feature_values == x)
    total = len(feature_values)
    return count / total


# Function to train Naive Bayes model with PMF
def train_naive_bayes_with_pmf(X_train, y_train):
    class_probs = {}
    pmfs = {}

    # Calculate class probabilities
    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    for class_label, count in class_counts.items():
        class_probs[class_label] = count / total_samples

    # Calculate PMF for each feature and class
    for class_label in class_probs.keys():
        class_data = X_train[y_train == class_label]
        pmfs[class_label] = {}
        for feature in X_train.columns:
            pmfs[class_label][feature] = {}
            feature_values = class_data[feature]
            unique_values = feature_values.unique()
            for value in unique_values:
                pmfs[class_label][feature][value] = calculate_pmf(value, feature_values)

    return class_probs, pmfs


# Function to make predictions using Naive Bayes model with PMF
def predict_naive_bayes_with_pmf(data_test, class_probs, pmfs):
    predictions = []
    for _, row in data_test.iterrows():
        class_scores = {}
        for ClassLabel, class_prob in class_probs.items():
            class_likelihood = 1
            for Feature, value in row.items():
                if Feature in pmfs[ClassLabel]:
                    if value in pmfs[ClassLabel][Feature]:
                        class_likelihood *= pmfs[ClassLabel][Feature][value]
                    else:
                        class_likelihood *= 0  # Smoothing for unseen values
            class_scores[ClassLabel] = class_prob * class_likelihood
        predicted_class = max(class_scores, key=class_scores.get)
        predictions.append(predicted_class)
    data_test['predicted'] = predictions
    return predictions, data_test


def naive_bayes_classifier():
    # Split dataset into training set and testing set
    data, k = preprocess_data()
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=1 - k, random_state=42)

    # Train Naive Bayes model with PMF
    class_probs_pmf, pmfs = train_naive_bayes_with_pmf(X_train.drop(columns=['diabetes']), y_train)

    # Make predictions using Naive Bayes model with PMF
    predictions_pmf, newData = predict_naive_bayes_with_pmf(X_test, class_probs_pmf, pmfs)

    # Print data_test as a table
    naive_data_text.delete(1.0, END)  # Clear previous content
    naive_data_text.insert(END, tabulate(newData, headers='keys', tablefmt='grid'))

    # Calculate accuracy
    accuracy_pmf = np.mean(predictions_pmf == y_test)
    # Display clusters
    naiveAccuracy_text.config(state=NORMAL)
    naiveAccuracy_text.delete(1.0, END)
    naiveAccuracy_text.insert(END, f"Accuracy:\n {accuracy_pmf}\n")
    # print("Accuracy (with PMF):", accuracy_pmf)

# # /////////////////////////////////////////Decision Tree Algorithm/////////////////////////////////////////////////

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def information_gain(data, feature_name):
    total_entropy = entropy(data.iloc[:, -1])
    values, counts = np.unique(data[feature_name], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(data[data[feature_name] == values[i]].iloc[:, -1]) for i in range(len(values)))
    information_gain = total_entropy - weighted_entropy
    return information_gain

def split_data(data, feature_name, value):
    return data[data[feature_name] == value]

# Step 5: Build the decision tree recursively
def build_tree(data):
    labels = data.iloc[:, -1]
    if len(np.unique(labels)) == 1:
        return np.unique(labels)[0]
    if data.shape[1] == 1 or len(data) == 0:
        return np.unique(labels)[np.argmax(np.unique(labels, return_counts=True)[1])]
    best_feature = max(data.columns[:-1], key=lambda x: information_gain(data, x))
    tree = {best_feature: {}}
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value].drop(columns=[best_feature])
        subtree = build_tree(sub_data)
        tree[best_feature][value] = subtree
    return tree

# Step 6: Make predictions using the trained decision tree
def predict(tree, sample):
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        value = sample[feature]
        if value in tree[feature]:
            subtree = tree[feature][value]
            return predict(subtree, sample)
    return tree

# Step 7: Evaluate the accuracy of the model
def accuracy(tree, data):
    correct = 0
    predictions = []
    for index, row in data.iterrows():
        value = predict(tree, row)
        if not isinstance(value, dict):
            predictions.append(value)
            if value == row.iloc[-1]:
                correct += 1
        else:
            # Drop the row from data DataFrame
            data = data.drop(index)
    data.reset_index(drop=True, inplace=True)
    # for _, row in data.iterrows():
    #     value = predict(tree, row)
    #     if not isinstance(value, dict):
    #         predictions.append(value)
    #     else:
    #         predictions.append(list(value.keys())[0])
    #     if value == row.iloc[-1]:
    #         correct += 1
    data['predicted'] = predictions
    # Print data_test as a table
    DT_data_text.delete(1.0, END)  # Clear previous content
    DT_data_text.insert(END, tabulate(data, headers='keys', tablefmt='grid'))
    return correct / len(data)

# Step 8: Split the data into training and testing sets
def DTsplit(data, test_size=0.25):
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data

def decision_tree_classifier():
    # Preprocess the data
    data, k = preprocess_data()
    train_data, test_data = DTsplit(data, test_size=1 - k)
    # Build the decision tree using the training data
    tree = build_tree(train_data)

    # Calculate accuracy using the testing data
    accur = accuracy(tree, test_data)
    # Display clusters
    decisionAccuracy_text.config(state=NORMAL)
    decisionAccuracy_text.delete(1.0, END)
    decisionAccuracy_text.insert(END, f"Accuracy:\n {accur}\n")
    # print("Accuracy:", accuracy)


# /////////////////////////////////////////====GuI====/////////////////////////////////////////////////
root = Tk()
root.title("Naive Bayes Classifier And Decision Tree Classifier")

# Add scrollbar to root window
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

# Create a Canvas widget to hold all the other widgets
canvas = Canvas(root, yscrollcommand=scrollbar.set)
canvas.pack(side=LEFT, fill=BOTH, expand=True)

# Configure the scrollbar to scroll the canvas
scrollbar.config(command=canvas.yview)

# Create a frame inside the canvas to hold all the widgets
frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor=NW)

file_label = Label(frame, text="Select File:")
file_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

file_entry = Entry(frame, width=50)
file_entry.grid(row=0, column=1, padx=5, pady=5)

file_button = Button(frame, text="Browse", command=select_file)
file_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

percentage_label = Label(frame, text="Percentage of Data to Read:")
percentage_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

percentage_entry = Entry(frame, width=10)
percentage_entry.grid(row=1, column=1, padx=4, pady=5)
percentage_entry.insert(END, "100")

TrainingLabel = Label(frame, text="Number of Training Set:")
TrainingLabel.grid(row=2, column=0, padx=5, pady=5, sticky="w")

TrainingEntry = Entry(frame, width=10)
TrainingEntry.grid(row=2, column=1, padx=5, pady=5)
TrainingEntry.insert(END, "75")

start_button = Button(frame, text="Run Algorithm", command=naive_bayes_classifier)
start_button.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

# Add treeview to display the data in function predict_naive_bayes_with_pmf before (return predictions)

naive_dataframe = Frame(frame)
naive_dataframe.grid(row=5, column=0, columnspan=3, padx=2, pady=2)

# Add a Text widget to display centroids
naive_data_text = Text(naive_dataframe, width=160, height=20, wrap=WORD)
naive_data_text.pack(side=LEFT, fill=Y)

# Add a Scrollbar to scroll through the centroids Text widget
naiveAccuracy_scrollbar = Scrollbar(naive_dataframe, orient=VERTICAL, command=naive_data_text.yview)
naiveAccuracy_scrollbar.pack(side=RIGHT, fill=Y)

# Link the Scrollbar to the centroids Text widget
naive_data_text.config(yscrollcommand=naiveAccuracy_scrollbar.set)

# Add treeview to display the data in function predict_naive_bayes_with_pmf before (return predictions)
TrainingLabel = Label(frame, text="Accuracy:")
TrainingLabel.grid(row=6, column=0, padx=5, pady=5, sticky="w")

naiveAccuracy_frame = Frame(frame)
naiveAccuracy_frame.grid(row=7, column=0, columnspan=3, padx=0, pady=2, sticky="w")

# Add a Text widget to display centroids
naiveAccuracy_text = Text(naiveAccuracy_frame, width=50, height=3, wrap=WORD)
naiveAccuracy_text.pack(side=LEFT, fill=Y)

button = Button(frame, text="Decision Tree Classifier", command=decision_tree_classifier)
button.grid(row=8, column=0, columnspan=3, padx=5, pady=5)

# Add treeview to display the data in function predict_naive_bayes_with_pmf before (return predictions)
DT_dataframe = Frame(frame)
DT_dataframe.grid(row=9, column=0, columnspan=3, padx=2, pady=2)

# Add a Text widget to display centroids
DT_data_text = Text(DT_dataframe, width=160, height=20, wrap=WORD)
DT_data_text.pack(side=LEFT, fill=Y)

# Add a Scrollbar to scroll through the centroids Text widget
DTAccuracy_scrollbar = Scrollbar(DT_dataframe, orient=VERTICAL, command=DT_data_text.yview)
DTAccuracy_scrollbar.pack(side=RIGHT, fill=Y)

# Link the Scrollbar to the centroids Text widget
DT_data_text.config(yscrollcommand=DTAccuracy_scrollbar.set)

# Add treeview to display the data in function predict_naive_bayes_with_pmf before (return predictions)
TrainingLabel = Label(frame, text="Accuracy:")
TrainingLabel.grid(row=10, column=0, padx=5, pady=5, sticky="w")

decisionAccuracy_frame = Frame(frame)
decisionAccuracy_frame.grid(row=11, column=0, columnspan=3, padx=0, pady=2, sticky="w")

# Add a Text widget to display centroids
decisionAccuracy_text = Text(decisionAccuracy_frame, width=50, height=3, wrap=WORD)
decisionAccuracy_text.pack(side=LEFT, fill=Y)

cong = Label(frame, text="Finally \n Congratulations !!!!!")
cong.grid(row=10, column=1, padx=5, pady=5, sticky="w")

# Set minimum size for root window based on the size of its content
root.update_idletasks()  # Update widgets to get correct sizes
root.minsize(root.winfo_width(), root.winfo_height())


# Configure the canvas scrolling region
canvas.config(scrollregion=canvas.bbox("all"))
root.mainloop()




# root = Tk()
# root.title("Naive bayes classifier And Decision Tree Classifier")
#
# file_label = Label(root, text="Select File:")
# file_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
#
# file_entry = Entry(root, width=50)
# file_entry.grid(row=0, column=0, padx=5, pady=5)
#
# file_button = Button(root, text="Browse", command=select_file)
# file_button.grid(row=0, column=0, padx=5, pady=5, sticky="e")
#
# percentage_label = Label(root, text="Percentage of Data to Read:")
# percentage_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
#
# percentage_entry = Entry(root, width=10)
# percentage_entry.grid(row=1, column=0, padx=4, pady=5)
# percentage_entry.insert(END, "100")
#
# TrainingLabel = Label(root, text="Number of Training Set:")
# TrainingLabel.grid(row=2, column=0, padx=5, pady=5, sticky="w")
#
# TrainingEntry = Entry(root, width=10)
# TrainingEntry.grid(row=2, column=0, padx=5, pady=5)
# TrainingEntry.insert(END, "75")
#
# start_button = Button(root, text="Run Algorithm", command=naive_bayes_classifier)
# start_button.grid(row=4, column=0, padx=5, pady=5)
#
# # add treeview to display the data in fuction predict_naive_bayes_with_pmf before (return predictions)
#
# naive_dataframe = Frame(root)
# naive_dataframe.grid(row=5, column=0, padx=2, pady=2)
#
# # Add a Text widget to display centroids
# naive_data_text = Text(naive_dataframe, width=150, height=20, wrap=WORD)
# naive_data_text.pack(side=LEFT, fill=Y)
#
# # Add a Scrollbar to scroll through the centroids Text widget
# naiveAccuracy_scrollbar = Scrollbar(naive_dataframe, orient=VERTICAL, command=naive_data_text.yview)
# naiveAccuracy_scrollbar.pack(side=RIGHT, fill=Y)
#
# # Link the Scrollbar to the centroids Text widget
# naive_data_text.config(yscrollcommand=naiveAccuracy_scrollbar.set)
#
# # add treeview to display the data in fuction predict_naive_bayes_with_pmf before (return predictions)
#
# naiveAccuracy_frame = Frame(root)
# naiveAccuracy_frame.grid(row=6, column=0, padx=0, pady=2)
#
# # Add a Text widget to display centroids
# naiveAccuracy_text = Text(naiveAccuracy_frame, width=30, height=1, wrap=WORD)
# naiveAccuracy_text.pack(side=LEFT, fill=Y)
#
# # Add a Scrollbar to scroll through the centroids Text widget
# naiveAccuracy_scrollbar = Scrollbar(naiveAccuracy_frame, orient=VERTICAL, command=naiveAccuracy_text.yview)
# naiveAccuracy_scrollbar.pack(side=RIGHT, fill=Y)
#
# # Link the Scrollbar to the centroids Text widget
# naiveAccuracy_text.config(yscrollcommand=naiveAccuracy_scrollbar.set)
#
# root.mainloop()
