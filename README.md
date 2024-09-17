
---

# 🌸 Iris Species Classification with Machine Learning 🌸

Dive into the fascinating world of flowers and data science with this project that classifies **Iris flower species** based on their physical attributes. Using popular machine learning algorithms like **Decision Tree** and **K-Nearest Neighbors (KNN)**, we’ve created a simple yet powerful model that boasts **perfect accuracy**! 🌟

## 📂 Dataset Overview

The famous **Iris dataset** contains 150 samples, each described by the following attributes:
- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**
- **Species** (Setosa, Versicolor, Virginica)

This dataset is a great starting point for machine learning enthusiasts, offering a clear, balanced classification problem.

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**:
   - **Standardization**: Features were scaled using **StandardScaler** to ensure that the models learn effectively.
   - **Data Splitting**: The data was split into **80% training** and **20% testing** for evaluating the models.

2. **Model Selection**:
   - **Decision Tree Classifier**: This tree-based model was trained to predict the species based on the four features.
   - **K-Nearest Neighbors (KNN)**: Another simple, yet effective model was trained to predict flower species by comparing to the nearest neighbors in the dataset.

3. **Training and Accuracy**:
   - Both models achieved a **perfect accuracy score of 100%** on the test set. This means the models predicted the species correctly for every sample! 🎯

## 🔍 Model Evaluation: Confusion Matrix

The confusion matrix below showcases how the Decision Tree model performed on the test set:

![image](https://github.com/user-attachments/assets/f164acf5-ce5a-407f-a891-5bdad63d9a23)

- **Setosa**, **Versicolor**, and **Virginica** were all classified correctly, demonstrating the strength of the model.

## 🏆 Results at a Glance

- **Decision Tree Accuracy**: `100%`
- **K-Nearest Neighbors Accuracy**: `100%`

These results show how well simple models like Decision Trees and KNN can perform on clean, well-structured datasets like the Iris dataset.

## 📊 Visualization: Confusion Matrix

The confusion matrix above gives a clear picture of the model's predictions. Here's a breakdown:
- **Perfect classification**: All samples were predicted correctly for each species (Setosa, Versicolor, and Virginica).
  
This is visualized as an easy-to-read heatmap for intuitive understanding.

## 🚀 Going Beyond

Interested in taking this further? Here are a few ways to enhance the project:
- **Model Comparison**: Try other classifiers such as **Random Forests**, **Support Vector Machines (SVM)**, or even **Logistic Regression** to see how they compare.
- **Cross-Validation**: Implement **cross-validation** to get a more robust understanding of your model's performance across different subsets of the data.
- **Web App Deployment**: Deploy the model using **Flask** or **Streamlit** and create a simple web interface for real-time flower classification.

## 🧰 Tools and Libraries

- **Jupyter Notebook** 📓
- **scikit-learn**: For model training and evaluation
- **pandas** & **numpy**: For data manipulation
- **matplotlib** & **seaborn**: For visualizations

## ⚙️ How to Run the Project

To try this project for yourself in **Jupyter Notebook**, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SupremeEvilGod/iris-classification.git
   cd iris-classification
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook iris_classification.ipynb
   ```

4. **Run the cells in sequence** to preprocess the data, train the model, and evaluate it.

## 📸 Screenshots

Here's a peek into the model’s performance:
![image](https://github.com/user-attachments/assets/f164acf5-ce5a-407f-a891-5bdad63d9a23)


## 📈 Performance Metrics

- **Decision Tree Classifier**: `Accuracy: 100%`
- **K-Nearest Neighbors (KNN)**: `Accuracy: 100%`

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
