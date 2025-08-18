# Perceptron Lab Exercise: Sorting Books with a Robot Librarian

## Introduction
In this lab, you’ll explore the **Perceptron**, a simple machine learning model that learns to classify data into two categories, like sorting books into **fiction (+1)** or **non-fiction (-1)**. Think of the Perceptron as a **robot librarian** who learns to sort books based on features like size and color. You’ll run code, visualize results, and answer questions to understand how it works.

### Analogy: The Robot Librarian
- **Setup (`__init__`)**: The librarian chooses how fast to learn (`eta`) and how many times to practice (`n_iter`).
- **Training (`fit`)**: They look at books (data), guess based on features (e.g., pages, color), and adjust their guessing rules (weights) if wrong.
- **Prediction (`net_input`, `predict`)**: For a new book, they calculate a score and decide fiction or non-fiction.
- **Docstring**: A guidebook explaining the librarian’s tools and results.

## Goals
- Run the Perceptron code to classify a simple dataset.
- Visualize the decision boundary and learning progress.
- Experiment with different settings (`eta`, `n_iter`).
- Answer questions to understand the Perceptron’s behavior.

## Prerequisites
- Basic Python knowledge (lists, loops, functions).
- Install Python libraries: `numpy`, `matplotlib` (run `pip install numpy matplotlib` in a terminal).
- A Python environment (e.g., Jupyter Notebook, VS Code, or an IDE).

## Lab Setup
You’ll use the following Perceptron code, which implements a binary classifier:

```python
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data."""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

## Exercise 1: Running the Perceptron
You’ll use a simple dataset to train the Perceptron and predict whether new books are fiction (+1) or non-fiction (-1).

### Dataset
The dataset represents 3 books with features `[size, color]`:
- `X = np.array([[2, 3], [1, 1], [4, 5]])` (size, color for 3 books).
- `y = np.array([1, -1, 1])` (fiction=+1, non-fiction=-1).
- Interpretation:
  - Book 1: `[2, 3]` → fiction (+1)
  - Book 2: `[1, 1]` → non-fiction (-1)
  - Book 3: `[4, 5]` → fiction (+1)

### Task
1. Copy the Perceptron code into a Python environment (e.g., Jupyter Notebook).
2. Run the following code to train the Perceptron and make a prediction:

```python
import numpy as np
X = np.array([[2, 3], [1, 1], [4, 5]])  # Features: size, color
y = np.array([1, -1, 1])  # Labels: fiction (+1), non-fiction (-1)
model = Perceptron(eta=0.1, n_iter=10)
model.fit(X, y)
print("Prediction for new book [3, 2]:", model.predict(np.array([3, 2])))
print("Errors per epoch:", model.errors_)
```

3. **Record the output**. It should look something like:
   ```
   Prediction for new book [3, 2]: -1
   Errors per epoch: [2, 1, 2, 1, 1, 1, 0, 0, 0, 0]
   ```

### Questions
1. What does the prediction `-1` mean for the book `[3, 2]`? (Hint: Think about the labels in `y`.)
2. Look at the errors list. How many total errors did the Perceptron make across all 10 epochs? (Sum the numbers.)
3. Why do the errors drop to 0 by epoch 7? What does this tell you about the data?

## Exercise 2: Visualizing Learning Progress
The errors list (`model.errors_`) shows how many books the Perceptron got wrong in each epoch. Let’s visualize this to see how the librarian learns.

### Task
1. Add the following code after training the Perceptron to plot the errors:

```python
import matplotlib.pyplot as plt
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Progress')
plt.grid(True)
plt.show()
```

2. Run the code and observe the plot.
3. **Record observations**:
   - How do the errors change over time?
   - When does the Perceptron stop making mistakes?

### Questions
1. Why do the errors fluctuate (e.g., 2, 1, 2, 1) before reaching 0?
2. What does it mean when errors reach 0? (Hint: Think about the librarian’s sorting rule.)

## Exercise 3: Visualizing the Decision Boundary
The Perceptron learns a line to separate fiction (+1) from non-fiction (-1) books in 2D space (size vs. color). Let’s visualize this decision boundary.

### Task
1. Add the following code to plot the data points and decision boundary:

```python
import matplotlib.pyplot as plt

# Plot data points
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Fiction (+1)')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Non-fiction (-1)')
plt.scatter([3], [2], color='green', marker='*', s=200, label='New book [3, 2]')

# Plot decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdBu')
plt.xlabel('Size')
plt.ylabel('Color')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
```

2. Run the code and observe the plot:
   - Blue circles: Fiction books (`+1`).
   - Red crosses: Non-fiction books (`-1`).
   - Green star: New book `[3, 2]`.
   - Shaded regions: Decision boundary (blue for `+1`, red for `-1`).

### Questions
1. Where is the new book `[3, 2]` located relative to the decision boundary? Does this explain the `-1` prediction?
2. How does the decision boundary separate the fiction and non-fiction books?
3. If you move the new book to `[4, 4]`, what prediction would you expect? Why?

## Exercise 4: Experimenting with Parameters
The librarian’s learning speed (`eta`) and practice rounds (`n_iter`) affect how they learn. Let’s experiment!

### Task
1. Modify the Perceptron parameters and rerun the code from Exercise 1:
   - Try `eta=0.01` (slower learning) and `n_iter=20`.
   - Try `eta=0.5` (faster learning) and `n_iter=5`.
2. For each setting, record:
   - The prediction for `[3, 2]`.
   - The errors list.
3. Plot the errors for each setting (use the code from Exercise 2).

### Questions
1. How does changing `eta` affect the errors list? (Compare the speed of learning.)
2. How does changing `n_iter` affect the results? Did fewer epochs still reach 0 errors?
3. Did the prediction for `[3, 2]` change with different settings? Why or why not?

## Exercise 5: Trying a New Dataset
Let’s use a subset of the **Iris dataset** (available in scikit-learn) to classify two types of flowers (Setosa vs. Versicolor) based on petal length and width.

### Task
1. Install scikit-learn: `pip install scikit-learn`.
2. Run the following code to load and train on the Iris dataset:

```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:100, [2, 3]]  # Petal length, petal width (first 100 samples: Setosa and Versicolor)
y = iris.target[:100]  # 0 for Setosa, 1 for Versicolor
y = np.where(y == 0, -1, 1)  # Convert to -1 and 1 for Perceptron
model = Perceptron(eta=0.1, n_iter=10)
model.fit(X, y)
print("Prediction for new flower [4.0, 1.0]:", model.predict(np.array([[4.0, 1.0]])))
print("Errors per epoch:", model.errors_)
```

3. Plot the errors and decision boundary (adapt the code from Exercises 2 and 3).
4. **Record** the prediction and errors.

### Questions
1. What does the prediction mean (Setosa or Versicolor)? (Check the Iris dataset labels.)
2. Does the errors list reach 0? Why or why not? (Hint: Is the Iris data linearly separable for these classes?)
3. How does the decision boundary look on the Iris data compared to the book dataset?

## Bonus Challenge
- Try adding a new book to the original dataset (e.g., `[3, 4]` with label `+1`) and retrain the Perceptron.
- Does the prediction for `[3, 2]` change? Why?
- Experiment with `random_state` (e.g., 42, 100). How does it affect the errors and prediction?

## Teaching Tips for Students
- **Visualize**: The plots in Exercises 2 and 3 show how the librarian learns and sorts books. Blue/red regions in the decision boundary plot show the fiction/non-fiction areas.
- **Simplify Math**: The `net_input` method is like adding up a score for a book (e.g., `size * weight_size + color * weight_color + bias`). The `predict` method checks if the score is positive (fiction) or negative (non-fiction).
- **Show Progress**: The errors plot shows the librarian getting better at sorting over time.
- **Hands-On**: Experimenting with `eta` and `n_iter` helps you see how the librarian’s learning speed and practice time affect their performance.

## Submission
- Submit your code, plots, and answers to the questions.
- Include a brief paragraph explaining what you learned about the Perceptron and the robot librarian analogy.

## Conclusion
This lab helps you understand the Perceptron by acting as a robot librarian sorting books. You’ve seen how it learns weights to classify data, visualized its decision boundary, and experimented with settings. The Perceptron is a simple but powerful model, forming the basis of modern neural networks!
