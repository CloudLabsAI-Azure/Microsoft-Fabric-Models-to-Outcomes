# Advanced Analytics: From Models to Outcomes with Microsoft Fabric - Lab 1




## Contents

- Introduction
- Task 1 - Getting Started with Microsoft Fabric: Activate Trial and Set Up a Workspace
- Task 2 - Data Science in Fabric
    - Task 2.1 - Data Science in Fabric (Notebook Setup)
    - Task 2.2 - Data Science in Fabric (Continued)


## Introduction

In this lab, you will begin an end-to-end Data Science workflow using Microsoft Fabric. The goal is to build a predictive model that determines whether bank customers are likely to churn—i.e., stop doing business with the bank. You'll walk through each phase of the workflow, from installing necessary libraries to training machine learning models and visualizing results in Power BI.

By the end of this lab, you’ll have a solid foundation for performing data science tasks in Microsoft Fabric, using tools like Data Wrangler, scikit-learn, LightGBM, MLflow, and built-in visualization features.

<img src=images/5wt984t4.jpg />

## Task 1: Getting Started with Microsoft Fabric: Activate Trial and Set Up a Workspace

1. In the top-right corner of Power BI, click the **User icon**, then select **Free trial** from the menu.

    ![alt text](image.png)

1. On Activate your 60-day free Fabric trial capacity dialog opens. Select **Activate**.

    ![alt text](image-1.png)

1. On Successfully upgraded to Microsoft Fabric dialog opens. Select **Fabric Home Page**.

    ![alt text](image-2.png)

1. On Welcome to the Fabric view dialog opens, click **Cancel**.

    ![alt text](image-3.png)

1. You will be navigated to the **Microsoft** **Fabric Home page**.

    ![alt text](image-4.png)

1. On the **Microsoft Fabric Home** page, click **+ New workspace** to create a new workspace.

    <img src=images/oihiw0yi.jpg />

    > **Note:** If a task flows preview feature notification appears, click **Got it** to proceed. 

    <img src=images/f16yvpx9.jpg />

3. Enter the workspace name **ws-<inject key="Deployment ID" enableCopy="false"/>*** and click/expand **Advanced** to assign the license mode for Fabric workload.

    <img src=images/guxmyzhx.jpg />

4. Choose **Fabric capacity** and select the available Capacity from the dropdown list. If no Fabric Capacity license is available.

    <img src=images/7p2lxw7o.jpg />
    
5. For the **Semantic model storage format**, select **Large Semantic model storage format**, then click **Apply**.

    > **Note:** Power BI semantic models use a highly compressed in-memory cache to deliver fast query performance and responsive user interaction. In Premium capacities, you can go beyond the default size limits by enabling the **Large semantic model storage format**. When this setting is enabled, the model size is governed by the Premium capacity or any maximum size defined by the administrator.

## Task 2:  Data Science in Fabric (Notebook Setup)


### Open the built-in notebook

The sample **Customer churn** notebook accompanies this lab manual.

To open the manual's built-in sample notebook in the Data Science experience:

1. Click **Workloads** in the left navigation menu and select **Data Science**.

    <img src=images/l8a9mlo3.jpg />

2. Select **Explore a Sample**.
    
    <img src=images/e3f6plyh.jpg />

3. Select the **Customer churn** sample, from the default **End-to-end workflows (Python)** tab.

    <img src=images/d15m9004.jpg />
    
4. After successful creation of sample notebook, you may explore the tour by clicking **Show me** or **Skip tour**.

    <img src=images/fb1dw2t5.jpg />

5. [Attach a lakehouse to the notebook](https://learn.microsoft.com/en-us/fabric/data-science/tutorial-data-science-prepare-system#attach-a-lakehouse-to-the-notebooks) before you start running code. Click **Data sources** and Select the option **Lakehouses** in the popup menu within the sample notebook. 

    <img src=images/76wv6e23.jpg />

6. In the **Add Lakehouse** popup, select **New Lakehouse** and click **Add**.

    <img src=images/nx5yibf9.jpg />

7. Enter the **New lakehouse** name as **CustomersLH** or a name of your choice and click **Create** (no need to select Lakehouse Schemas checkbox (Public Preview)).

    <img src=images/4mkel884.jpg />

8. Now, your sample notebook is ready for execution.

    <img src=images/jlybvgt3.jpg />

- [ ]  Tick this box to indicate the successful import of the sample notebook and completion of lakehouse creation.

===

# Exercise 1 - Data Science in Fabric (Continued)

This lab consists of **five different sections**/exercises and here is an overview. You are currently in **Exercise 1: Part2 - Data Science in Fabric (Continued)** exercise.

- **[Configure Workspace](#configure-workspace)**
- **[Exercise 1 - Data Science in Fabric](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part1 - Data Science in Fabric (Notebook Setup)](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part2 - Data Science in Fabric (Continued)](#exercise-1---data-science-in-fabric-continued)**
- **[Exercise 2 - Client Tools (Optional)](##exercise-2---client-tools)**
- **[Exercise 3 - Copilot Data Science Experience (Optional)](##exercise-3---copilot-data-science-experience-enhanced-customer-churn-sample-with-copilot)**
- **[Exercise 4 - Power BI Visualization + PBI Copilot (Optional)](##exercise-4---power-bi-visualization--pbi-copilot)**
- **[Thank You Note and Feedback Request!](#feedback-your-feedback-is-valuable)**

Let's dive into this exercise step by step to discover what **Fabric offers in terms of Data Science capabilities**, including **Notebook**, **Experiment**, **ML model**, and more.

<img src=images/dd3u2buf.jpg />

💡***Quick Tip*!**💡</br>
***After completing** **Exercise 1 - Data Science in Fabric (Notebook Setup)**, you can click "**Run All**" ▶️ under **Home** menu to **execute all cells and preview the notebook results** before going through the step-by-step walkthrough. This helps you get an overview of the final outputs and can make troubleshooting easier.*

## Step 1: Install custom libraries (Ctrl + Enter or Press Run cell icon next to the cell <img src=images/i4uc7ogh.jpg />).

For machine learning model development or ad-hoc data analysis, you might need to quickly install a custom library for your Apache Spark session. You have two options to install libraries.

- Use the inline installation capabilities (`%pip` or `%conda`) of your notebook to install a library, in your current notebook only.
- Alternatively, you can create a Fabric environment, install libraries from public sources or upload custom libraries to it, and then your workspace admin can attach the environment as the default for the workspace. All the libraries in the environment will then become available for use in any notebooks and Spark job definitions in the workspace. For more information on environments, see [create, configure, and use an environment in Microsoft Fabric](https://aka.ms/fabric/create-environment).

For this tutorial, use `%pip install` to install the `imblearn` library in your notebook.

 Note

The PySpark kernel restarts after `%pip install` runs. Install the needed libraries before you run any other cells.

PythonCopy

```
# Use pip to install libraries
%pip install imblearn
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#step-2-load-the-data)

- [ ]  Tick this box to indicate Step 1 is complete. 

## Step 2: Load the data (Ctrl + Enter or Press Run cell icon next to the cell <img src=images/i4uc7ogh.jpg />).

The dataset in _churn.csv_ contains the churn status of 10,000 customers, along with 14 attributes that include:

- Credit score
- Geographical location (Germany, France, Spain)
- Gender (male, female)
- Age
- Tenure (number of years the person was a customer at that bank)
- Account balance
- Estimated salary
- Number of products that a customer purchased through the bank
- Credit card status (whether or not a customer has a credit card)
- Active member status (whether or not the person is an active bank customer)

The dataset also includes row number, customer ID, and customer surname columns. Values in these columns shouldn't influence a customer's decision to leave the bank.

A customer bank account closure event defines the churn for that customer. The dataset `Exited` column refers to the customer's abandonment. Since we have little context about these attributes, we don't need background information about the dataset. We want to understand how these attributes contribute to the `Exited` status.

Out of those 10,000 customers, only 2037 customers (roughly 20%) left the bank. Because of the class imbalance ratio, we recommend generation of synthetic data. Confusion matrix accuracy might not have relevance for imbalanced classification. We might want to measure the accuracy using the Area Under the Precision-Recall Curve (AUPRC).

- This table shows a preview of the `churn.csv` data:

Expand table

|CustomerID|Surname|CreditScore|Geography|Gender|Age|Tenure|Balance|NumOfProducts|HasCrCard|IsActiveMember|EstimatedSalary|Exited|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|15634602|Hargrave|619|France|Female|42|2|0.00|1|1|1|101348.88|1|
|15647311|Hill|608|Spain|Female|41|1|83807.86|1|0|1|112542.58|0|

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#download-the-dataset-and-upload-to-the-lakehouse)

### Download the dataset and upload to the lakehouse

Define these parameters, so that you can use this notebook with different datasets:

PythonCopy

```
IS_CUSTOM_DATA = False  # If TRUE, the dataset has to be uploaded manually

IS_SAMPLE = False  # If TRUE, use only SAMPLE_ROWS of data for training; otherwise, use all data
SAMPLE_ROWS = 5000  # If IS_SAMPLE is True, use only this number of rows for training

DATA_ROOT = "/lakehouse/default"
DATA_FOLDER = "Files/churn"  # Folder with data files
DATA_FILE = "churn.csv"  # Data file name
```

This code downloads a publicly available version of the dataset, and then stores that dataset in a Fabric lakehouse:

 Important

[Add a lakehouse](https://aka.ms/fabric/addlakehouse) to the notebook before you run it. Failure to do so will result in an error.

PythonCopy

```
import os, requests
if not IS_CUSTOM_DATA:
# With an Azure Synapse Analytics blob, this can be done in one line

# Download demo data files into the lakehouse if they don't exist
    remote_url = "https://synapseaisolutionsa.blob.core.windows.net/public/bankcustomerchurn"
    file_list = ["churn.csv"]
    download_path = "/lakehouse/default/Files/churn/raw"

    if not os.path.exists("/lakehouse/default"):
        raise FileNotFoundError(
            "Default lakehouse not found, please add a lakehouse and restart the session."
        )
    os.makedirs(download_path, exist_ok=True)
    for fname in file_list:
        if not os.path.exists(f"{download_path}/{fname}"):
            r = requests.get(f"{remote_url}/{fname}", timeout=30)
            with open(f"{download_path}/{fname}", "wb") as f:
                f.write(r.content)
    print("Downloaded demo data files into lakehouse.")
```

Start recording the time needed to run the notebook:

PythonCopy

```
# Record the notebook running time
import time

ts = time.time()
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#read-raw-data-from-the-lakehouse)

### Read raw data from the lakehouse

Explore the downloaded raw data in Lakehouse under **Files** (click the refresh icon in the Lakehouse **ellipsis (...) menu** within the Files section).

<img src=2t191vi7.jpg](images/2t191vi7.jpg />

This code reads raw data from the **Files** section of the lakehouse, and adds more columns for different date parts. Creation of the partitioned delta table uses this information.

PythonCopy

```
df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("Files/churn/raw/churn.csv")
    .cache()
)
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#create-a-pandas-dataframe-from-the-dataset)

### Create a pandas DataFrame from the dataset

This code converts the Spark DataFrame to a pandas DataFrame, for easier processing and visualization:

PythonCopy

```
df = df.toPandas()
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#step-3-perform-exploratory-data-analysis)

- [ ]  Tick this box to indicate Step 2 is complete. 

## Step 3: Perform exploratory data analysis (Ctrl + Enter or Press Run cell icon next to the cell <img src=i4uc7ogh.jpg](images/i4uc7ogh.jpg />).

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#display-raw-data)

### Display raw data

Explore the raw data with `display`, calculate some basic statistics, and show chart views. You must first import the required libraries for data visualization - for example, [seaborn](https://seaborn.pydata.org/). Seaborn is a Python data visualization library, and it provides a high-level interface to build visuals on dataframes and arrays.

PythonCopy

```
import seaborn as sns
sns.set_theme(style="whitegrid", palette="tab10", rc = {'figure.figsize':(9,6)})
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc, rcParams
import numpy as np
import pandas as pd
import itertools
```

PythonCopy

```
display(df, summary=True)
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#use-data-wrangler-to-perform-initial-data-cleaning)

### Use Data Wrangler to perform initial data cleaning

Launch Data Wrangler directly from the notebook to explore and transform pandas dataframes. At the notebook ribbon **Home** tab *(Data Wrangler is on the right of the Run all button)*, use the Data Wrangler dropdown prompt to browse the activated pandas DataFrames available for editing. Select the DataFrame you want to open in Data Wrangler.

 Note

Data Wrangler cannot be opened while the notebook kernel is busy. The cell execution must finish before you launch Data Wrangler. [Learn more about Data Wrangler](https://aka.ms/fabric/datawrangler).

<img src=images/8yif5ozt.jpg />

After the Data Wrangler launches, a descriptive overview of the data panel is generated, as shown in the following images. The overview includes information about the dimension of the DataFrame, any missing values, etc. You can use Data Wrangler to generate the script to drop the rows with missing values, the duplicate rows and the columns with specific names. Then, you can copy the script into a cell. The next cell shows that copied script.

![Screenshot that shows the Data Wrangler menu.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/menu-data-wrangler.png)

<img src=images/orqeibbd.jpg />

PythonCopy

```
# Code generated by Data Wrangler for pandas DataFrame
def clean_data(df):    
    # Drop duplicate rows across all columns    
    df = df.drop_duplicates()    
    # Drop duplicate rows in columns: 'RowNumber', 'CustomerId'    
    df = df.drop_duplicates(subset=['RowNumber', 'CustomerId'])    
    # Drop duplicate rows in columns: 'RowNumber', 'CustomerId'    
    df = df.drop_duplicates(subset=['RowNumber', 'CustomerId'])    
    return df

df_clean = clean_data(df.copy())
df_clean.head()

```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#determine-attributes)

### Determine attributes

This code determines the categorical, numerical, and target attributes:

PythonCopy

```
# Determine the dependent (target) attribute
dependent_variable_name = "Exited"
print(dependent_variable_name)
# Determine the categorical attributes
categorical_variables = [col for col in df_clean.columns if col in "O"
                        or df_clean[col].nunique() <=5
                        and col not in "Exited"]
print(categorical_variables)
# Determine the numerical attributes
numeric_variables = [col for col in df_clean.columns if df_clean[col].dtype != "object"
                        and df_clean[col].nunique() >5]
print(numeric_variables)
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#show-the-five-number-summary)

### Show the five-number summary

Use box plots to show the five-number summary

- the minimum score
- first quartile
- median
- third quartile
- maximum score

for the numerical attributes.

PythonCopy

```
df_num_cols = df_clean[numeric_variables]
sns.set(font_scale = 0.7) 
fig, axes = plt.subplots(nrows = 2, ncols = 3, gridspec_kw =  dict(hspace=0.3), figsize = (17,8))
fig.tight_layout()
for ax,col in zip(axes.flatten(), df_num_cols.columns):
    sns.boxplot(x = df_num_cols[col], color='green', ax = ax)
# fig.suptitle('visualize and compare the distribution and central tendency of numerical attributes', color = 'k', fontsize = 12)
fig.delaxes(axes[1,2])
```

![Screenshot that shows a notebook display of the box plot for numerical attributes.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/box-plots.jpg)

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#show-the-distribution-of-exited-and-non-exited-customers)

### Show the distribution of exited and non-exited customers

Show the distribution of exited versus non-exited customers, across the categorical attributes:

PythonCopy

```
attr_list = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Tenure']
fig, axarr = plt.subplots(2, 3, figsize=(15, 4))
for ind, item in enumerate (attr_list):
    sns.countplot(x = item, hue = 'Exited', data = df_clean, ax = axarr[ind%2][ind//2])
fig.subplots_adjust(hspace=0.7)
```

![Screenshot that shows a notebook display of the distribution of exited versus non-exited customers.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/bar-charts.jpg)

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#show-the-distribution-of-numerical-attributes)

### Show the distribution of numerical attributes

Use a histogram to show the frequency distribution of numerical attributes:

PythonCopy

```
columns = df_num_cols.columns[: len(df_num_cols.columns)]
fig = plt.figure()
fig.set_size_inches(18, 8)
length = len(columns)
for i,j in itertools.zip_longest(columns, range(length)):
    plt.subplot((length // 2), 3, j+1)
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    df_num_cols[i].hist(bins = 20, edgecolor = 'black')
    plt.title(i)
# fig = fig.suptitle('distribution of numerical attributes', color = 'r' ,fontsize = 14)
plt.show()

```

![Screenshot that shows a notebook display of numerical attributes.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/numerical-attributes.jpg)

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#perform-feature-engineering)

### Perform feature engineering

This feature engineering generates new attributes based on the current attributes:

PythonCopy

```
df_clean["NewTenure"] = df_clean["Tenure"]/df_clean["Age"]
df_clean["NewCreditsScore"] = pd.qcut(df_clean['CreditScore'], 6, labels = [1, 2, 3, 4, 5, 6])
df_clean["NewAgeScore"] = pd.qcut(df_clean['Age'], 8, labels = [1, 2, 3, 4, 5, 6, 7, 8])
df_clean["NewBalanceScore"] = pd.qcut(df_clean['Balance'].rank(method="first"), 5, labels = [1, 2, 3, 4, 5])
df_clean["NewEstSalaryScore"] = pd.qcut(df_clean['EstimatedSalary'], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#use-data-wrangler-to-perform-one-hot-encoding)

### Use Data Wrangler to perform one-hot encoding

With the same steps to launch Data Wrangler, as discussed earlier, use the Data Wrangler to perform one-hot encoding. This cell shows the copied generated script for one-hot encoding:

![Screenshot that shows one-hot encoding in Data Wrangler.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/1-hot-encoding-data-wrangler.png)

<img src=images/cspw8248.jpg />

PythonCopy

```
# Code generated by Data Wrangler for pandas DataFrame
import pandas as pd

def clean_data(df):    
    # One-hot encode columns: 'Geography', 'Gender'    
    for column in ['Geography', 'Gender']:        
        insert_loc = df.columns.get_loc(column)        
        df = pd.concat([df.iloc[:,:insert_loc], pd.get_dummies(df.loc[:, [column]]), df.iloc[:,insert_loc+1:]], axis=1)    
        return df

df_clean = clean_data(df_clean.copy())
df_clean.head()
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#create-a-delta-table-to-generate-the-power-bi-report)

### Create a delta table to generate the Power BI report

PythonCopy

```
table_name = "df_clean"
# Create a PySpark DataFrame from pandas
sparkDF=spark.createDataFrame(df_clean) 
sparkDF.write.mode("overwrite").format("delta").save(f"Tables/{table_name}")
print(f"Spark DataFrame saved to delta table: {table_name}")
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#summary-of-observations-from-the-exploratory-data-analysis)

### Summary of observations from the exploratory data analysis

- Most of the customers are from France. Spain has the lowest churn rate, compared to France and Germany.
- Most customers have credit cards
- Some customers are both over the age of 60 and have credit scores below 400. However, they can't be considered as outliers
- Very few customers have more than two bank products
- Inactive customers have a higher churn rate
- Gender and tenure years have little impact on a customer's decision to close a bank account

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#step-4-perform-model-training-and-tracking)

- [ ]  Tick this box to indicate Step 3 is complete. 

## Step 4: Perform model training and tracking (Ctrl + Enter or Press Run cell icon next to the cell <img src=images/i4uc7ogh.jpg />).

With the data in place, you can now define the model. Apply random forest and LightGBM models in this notebook.

Use the scikit-learn and LightGBM libraries to implement the models, with a few lines of code. Additionally, use MLfLow and Fabric Autologging to track the experiments.

This code sample loads the delta table from the lakehouse. You can use other delta tables that themselves use the lakehouse as the source.

PythonCopy

```
SEED = 12345
df_clean = spark.read.format("delta").load("Tables/df_clean").toPandas()
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#generate-an-experiment-for-tracking-and-logging-the-models-by-using-mlflow)

### Generate an experiment for tracking and logging the models by using MLflow

This section shows how to generate an experiment, and it specifies the model and training parameters and the scoring metrics. Additionally, it shows how to train the models, log them, and save the trained models for later use.

PythonCopy

```
import mlflow

# Set up the experiment name
EXPERIMENT_NAME = "sample-bank-churn-experiment"  # MLflow experiment name
```

Autologging automatically captures both the input parameter values and the output metrics of a machine learning model, as that model is trained. This information is then logged to your workspace, where the MLflow APIs or the corresponding experiment in your workspace can access and visualize it.

When complete, your experiment will resemble this image. Instructions for accessing the experiment are provided in the following section, **View the experiment artifact to track model performance**.:

![Screenshot that shows the experiment page for the bank churn experiment.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/experiment-runs.png)

All the experiments with their respective names are logged, and you can track their parameters and performance metrics. To learn more about autologging, see [Autologging in Microsoft Fabric](https://aka.ms/fabric-autologging).

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#set-experiment-and-autologging-specifications)

### Set experiment and autologging specifications

PythonCopy

```
mlflow.set_experiment(EXPERIMENT_NAME) # Use a date stamp to append to the experiment
mlflow.autolog(exclusive=False)
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#import-scikit-learn-and-lightgbm)

### Import scikit-learn and LightGBM

PythonCopy

```
# Import the required libraries for model training
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score, roc_auc_score, classification_report
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#prepare-training-and-test-datasets)

### Prepare training and test datasets

PythonCopy

```
y = df_clean["Exited"]
X = df_clean.drop("Exited",axis=1)
# Train/test separation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#apply-smote-to-the-training-data)

### Apply SMOTE to the training data

Imbalanced classification has a problem, because it has too few examples of the minority class for a model to effectively learn the decision boundary. To handle this, Synthetic Minority Oversampling Technique (SMOTE) is the most widely used technique to synthesize new samples for the minority class. Access SMOTE with the `imblearn` library that you installed in step 1.

Apply SMOTE only to the training dataset. You must leave the test dataset in its original imbalanced distribution, to get a valid approximation of model performance on the original data. This experiment represents the situation in production.

PythonCopy

```
from collections import Counter
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=SEED)
X_res, y_res = sm.fit_resample(X_train, y_train)
new_train = pd.concat([X_res, y_res], axis=1)
```

For more information, see [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#) and [From random over-sampling to SMOTE and ADASYN](https://imbalanced-learn.org/stable/over_sampling.html#smote-adasyn). The imbalanced-learn website hosts these resources.

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#train-the-model)

### Train the model

Use Random Forest to train the model, with a maximum depth of four, and with four features:

PythonCopy

```
mlflow.sklearn.autolog(registered_model_name='rfc1_sm')  # Register the trained model with autologging
rfc1_sm = RandomForestClassifier(max_depth=4, max_features=4, min_samples_split=3, random_state=1) # Pass hyperparameters
with mlflow.start_run(run_name="rfc1_sm") as run:
    rfc1_sm_run_id = run.info.run_id # Capture run_id for model prediction later
    print("run_id: {}; status: {}".format(rfc1_sm_run_id, run.info.status))
    # rfc1.fit(X_train,y_train) # Imbalanced training data
    rfc1_sm.fit(X_res, y_res.ravel()) # Balanced training data
    rfc1_sm.score(X_test, y_test)
    y_pred = rfc1_sm.predict(X_test)
    cr_rfc1_sm = classification_report(y_test, y_pred)
    cm_rfc1_sm = confusion_matrix(y_test, y_pred)
    roc_auc_rfc1_sm = roc_auc_score(y_res, rfc1_sm.predict_proba(X_res)[:, 1])
```

Use Random Forest to train the model, with a maximum depth of eight, and with six features:

PythonCopy

```
mlflow.sklearn.autolog(registered_model_name='rfc2_sm')  # Register the trained model with autologging
rfc2_sm = RandomForestClassifier(max_depth=8, max_features=6, min_samples_split=3, random_state=1) # Pass hyperparameters
with mlflow.start_run(run_name="rfc2_sm") as run:
    rfc2_sm_run_id = run.info.run_id # Capture run_id for model prediction later
    print("run_id: {}; status: {}".format(rfc2_sm_run_id, run.info.status))
    # rfc2.fit(X_train,y_train) # Imbalanced training data
    rfc2_sm.fit(X_res, y_res.ravel()) # Balanced training data
    rfc2_sm.score(X_test, y_test)
    y_pred = rfc2_sm.predict(X_test)
    cr_rfc2_sm = classification_report(y_test, y_pred)
    cm_rfc2_sm = confusion_matrix(y_test, y_pred)
    roc_auc_rfc2_sm = roc_auc_score(y_res, rfc2_sm.predict_proba(X_res)[:, 1])
```

Train the model with LightGBM:

PythonCopy

```
# lgbm_model
mlflow.lightgbm.autolog(registered_model_name='lgbm_sm')  # Register the trained model with autologging
lgbm_sm_model = LGBMClassifier(learning_rate = 0.07, 
                        max_delta_step = 2, 
                        n_estimators = 100,
                        max_depth = 10, 
                        eval_metric = "logloss", 
                        objective='binary', 
                        random_state=42)

with mlflow.start_run(run_name="lgbm_sm") as run:
    lgbm1_sm_run_id = run.info.run_id # Capture run_id for model prediction later
    # lgbm_sm_model.fit(X_train,y_train) # Imbalanced training data
    lgbm_sm_model.fit(X_res, y_res.ravel()) # Balanced training data
    y_pred = lgbm_sm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cr_lgbm_sm = classification_report(y_test, y_pred)
    cm_lgbm_sm = confusion_matrix(y_test, y_pred)
    roc_auc_lgbm_sm = roc_auc_score(y_res, lgbm_sm_model.predict_proba(X_res)[:, 1])
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#view-the-experiment-artifact-to-track-model-performance)

### View the experiment artifact to track model performance

The experiment runs are automatically saved in the experiment artifact. You can find that artifact in the workspace. An artifact name is based on the name used to set the experiment. All of the trained models, their runs, performance metrics and model parameters are logged on the experiment page.

To view your experiments:

1. On the left panel, select your workspace **ws-models2outcomes-##** to explore the run artifacts such as **Experiment** and **ML model**, as well as Notebook, Lakehouse, Semantic model, and SQL analytics endpoint.

<img src=images/fcg3z290.jpg />

2. Find and select the experiment name, in this case, **sample-bank-churn-experiment**.

<img src=images/z1mb8by4.jpg />

3. Compare run details in a list view. Click **View run list** under **Compare runs** to Select certain runs to visually compare run metrics. 

<img src=images/7znqi74n.jpg />

4. Select all runs under **Run list** menu.

<img src=images/0756z1nk.jpg />


- [ ]  Tick this box to indicate Step 4 is complete. 

## Step 5: Evaluate and save the final machine learning model (Ctrl + Enter or Press Run cell icon next to the cell <img src=images/i4uc7ogh.jpg />).

Open the saved experiment from the workspace to select and save the best model:

PythonCopy

```
# Define run_uri to fetch the model
# MLflow client: mlflow.model.url, list model
load_model_rfc1_sm = mlflow.sklearn.load_model(f"runs:/{rfc1_sm_run_id}/model")
load_model_rfc2_sm = mlflow.sklearn.load_model(f"runs:/{rfc2_sm_run_id}/model")
load_model_lgbm1_sm = mlflow.lightgbm.load_model(f"runs:/{lgbm1_sm_run_id}/model")
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#assess-the-performance-of-the-saved-models-on-the-test-dataset)

### Assess the performance of the saved models on the test dataset

PythonCopy

```
ypred_rfc1_sm = load_model_rfc1_sm.predict(X_test) # Random forest with maximum depth of 4 and 4 features
ypred_rfc2_sm = load_model_rfc2_sm.predict(X_test) # Random forest with maximum depth of 8 and 6 features
ypred_lgbm1_sm = load_model_lgbm1_sm.predict(X_test) # LightGBM
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#show-truefalse-positivesnegatives-by-using-a-confusion-matrix)

### Show true/false positives/negatives by using a confusion matrix

To evaluate the accuracy of the classification, build a script that plots the confusion matrix. You can also plot a confusion matrix using SynapseML tools, as shown in the [Fraud Detection sample](https://aka.ms/samples/frauddectection).

PythonCopy

```
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)
    plt.figure(figsize=(4,4))
    plt.rcParams.update({'font.size': 10})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, color="blue")
    plt.yticks(tick_marks, classes, color="blue")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

Create a confusion matrix for the random forest classifier, with a maximum depth of four, with four features:

PythonCopy

```
cfm = confusion_matrix(y_test, y_pred=ypred_rfc1_sm)
plot_confusion_matrix(cfm, classes=['Non Churn','Churn'],
                      title='Random Forest with max depth of 4')
tn, fp, fn, tp = cfm.ravel()
```

![Screenshot that shows a notebook display of a confusion matrix for random forest with a maximum depth of four.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/confusion-random-forest-depth-4.jpg)

Create a confusion matrix for the random forest classifier with maximum depth of eight, with six features:

PythonCopy

```
cfm = confusion_matrix(y_test, y_pred=ypred_rfc2_sm)
plot_confusion_matrix(cfm, classes=['Non Churn','Churn'],
                      title='Random Forest with max depth of 8')
tn, fp, fn, tp = cfm.ravel()
```

![Screenshot that shows a notebook display of a confusion matrix for random forest with a maximum depth of eight.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/confusion-random-forest-depth-8.jpg)

Create a confusion matrix for LightGBM:

PythonCopy

```
cfm = confusion_matrix(y_test, y_pred=ypred_lgbm1_sm)
plot_confusion_matrix(cfm, classes=['Non Churn','Churn'],
                      title='LightGBM')
tn, fp, fn, tp = cfm.ravel()
```

![Screenshot that shows a notebook display of a confusion matrix for LightGBM.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/confusion-lgbm.jpg)

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#save-results-for-power-bi)

### Save results for Power BI

Save the delta frame to the lakehouse, to move the model prediction results to a Power BI visualization.

PythonCopy

```
df_pred = X_test.copy()
df_pred['y_test'] = y_test
df_pred['ypred_rfc1_sm'] = ypred_rfc1_sm
df_pred['ypred_rfc2_sm'] =ypred_rfc2_sm
df_pred['ypred_lgbm1_sm'] = ypred_lgbm1_sm
table_name = "df_pred_results"
sparkDF=spark.createDataFrame(df_pred)
sparkDF.write.mode("overwrite").format("delta").option("overwriteSchema", "true").save(f"Tables/{table_name}")
print(f"Spark DataFrame saved to delta table: {table_name}")
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#step-6-access-visualizations-in-power-bi)

- [ ]  Tick this box to indicate Step 5 is complete. 

## Step 6: Access visualizations in Power BI

**Note**: If you are encountering error **Cannot load model** it might be the chance that the **Power BI Pro/PPU/Trial** license is not activated for the user. Recheck whether you have activate the trail to get Power BI authering experience. This was covered in the **Configure Workspace** earlier section under **Step 2** (9. Click on User Profile in the top right corner, then select Free trial from the user profile menu, and finally click Activate in the popup window).

Access your saved table in Power BI:

1. On the left, select **OneLake data hub**

    <img src=images/nrkl6asa.jpg />

2. Select the lakehouse that you added to this notebook

    <img src=images/2zsyr6ac.jpg />

3. In the **Open this Lakehouse** section, select **Open**

    <img src=images/yp8jiinf.jpg />

4. *Optionally*, explore using **SQL analytics endpoints** if you're familiar with T-SQL.

    <img src=images/7egtfd29.jpg />

    Select **New SQL Query** for a specific tables and **Select TOP 100**.

    <img src=images/c7od0oeo.jpg />

    Click **Run** to view the results in SQL editor.

    <img src=images/b5pqxi96.jpg />

    Click **New semantic model** from **Reporting**.

    <img src=images/mkc6of3h.jpg />

5. On the ribbon, select **New semantic model**. Check `df_pred_results` under `dbo` and `Tables`, then give it a name **ChurnDS** and select **Confirm** to create a new Power BI semantic model linked to the predictions.

    <img src=images/hjmgbhl4.jpg />

    <img src=images/60mgfejj.jpg />

6. *Optionally*, you may go back to the **Workspace** to use the **default semantic model** that comes with every lakehouse. 

    <img src=images/5zhr2sq0.jpg />

7. Go back to the **Workspace** to access the **Semantic model** you created. Use the **more options (...)** to **Create report ** with the semantic model, which will open the Power BI report **authoring page**.

    <img src=images/yx2omoso.jpg />

    <img src=images/964iga0o.jpg />

**💡Quick Tip!💡**</br>
*Enable **Copilot** from the ribbon to get content suggestions for a **new report page** and have it create one for you!!*


The following screenshot shows some example visualizations. The data panel shows the delta tables and columns to select from a table. After selection of appropriate category (x) and value (y) axis, you can choose the filters and functions - for example, sum or average of the table column.

**Note:** *If you are encountering error **Cannot load model** it might be the chance that the **Power BI Pro/PPU/Trial** license is not activated for the user. Recheck whether you have activate the trail to get Power BI authering experience. This was covered in the **Configure Workspace** earlier section under **Step 2** (9. Click on User Profile in the top right corner, then select Free trial from the user profile menu, and finally click Activate in the popup window).*

In this screenshot, the illustrated example describes the analysis of the saved prediction results in Power BI:

![Screenshot that shows a Power BI dashboard example.](https://learn.microsoft.com/en-us/fabric/data-science/media/tutorial-bank-churn/power-bi-dashboard.png)

However, for a real customer churn use-case, the user might need a more thorough set of requirements of the visualizations to create, based on subject matter expertise, and what the firm and business analytics team and firm have standardized as metrics.

The Power BI report shows that customers who use more than two of the bank products have a higher churn rate. However, few customers had more than two products. (See the plot in the bottom left panel.) The bank should collect more data, but should also investigate other features that correlate with more products.

Bank customers in Germany have a higher churn rate compared to customers in France and Spain. (See the plot in the bottom right panel). Based on the report results, an investigation into the factors that encouraged customers to leave might help.

There are more middle-aged customers (between 25 and 45). Customers between 45 and 60 tend to exit more.

Finally, customers with lower credit scores would most likely leave the bank for other financial institutions. The bank should explore ways to encourage customers with lower credit scores and account balances to stay with the bank.

PythonCopy

```
# Determine the entire runtime
print(f"Full run cost {int(time.time() - ts)} seconds.")
```

[](https://learn.microsoft.com/en-us/fabric/data-science/customer-churn#related-content)

- [ ]  Tick this box to indicate Step 6 is complete. 

**Congratulations**, you have successfully completed **Exercise 1 - Data Science in Fabric**!

## Related content

- [Machine learning model in Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/data-science/machine-learning-model)
- [Train machine learning models](https://learn.microsoft.com/en-us/fabric/data-science/model-training-overview)
- [Machine learning experiments in Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/data-science/machine-learning-experiment)

===
# Exercise 2 - Client tools

This lab consists of **five different sections**/exercises and here is an overview. You are currently in **Exercise 2 - Client Tools (Optional)** exercise.

- **[Configure Workspace](#configure-workspace)**
- **[Exercise 1 - Data Science in Fabric](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part1 - Data Science in Fabric (Notebook Setup)](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part2 - Data Science in Fabric (Continued)](#exercise-1---data-science-in-fabric-continued)**
- **[Exercise 2 - Client Tools (Optional)](#exercise-2---client-tools)**
- **[Exercise 3 - Copilot Data Science Experience (Optional)](#exercise-3---copilot-data-science-experience-enhanced-customer-churn-sample-with-copilot)**
- **[Exercise 4 - Power BI Visualization + PBI Copilot (Optional)](#exercise-4---power-bi-visualization--pbi-copilot)**
- **[Thank You Note and Feedback Request!](#feedback-your-feedback-is-valuable)**

There are few client tools such as OneLake File Explorer, Power BI Desktop and VS Code for your ecxploration. 

- [OneLake File Explorer](#onelake-file-explorer)
- [Power BI Desktop](#power-bi-desktop)
- [VS Code](#vs-code)

## OneLake File Explorer:
This application automatically syncs all Microsoft OneLake items that you have access to in Windows File Explorer. To log in to Microsoft OneLake File Explorer and explore its features.

1. **Launch OneLake File Explorer:**
   - **OneLake File Explorer** may automatically launch and prompt for a Microsoft account login.
   - If not, manually start it by clicking the OneLake icon from the Taskbar or searching for **OneLake File Explorer** in the **Windows Start** menu.

2. **Log in to OneLake File Explorer:**
   - Upon launching, the login prompt will appear.
   
    <img src=images/at7pj0pl.jpg />

   - Enter the **username**: **Username** and click **Next**.
   - Enter the **password**: **Password** and click **Sign in**.
   - Select **No, Sign in to this app only** for quicker response (click Done if already signed in).

    <img src=images/7v161kzz.jpg />

3. **Explore OneLake File Explorer:**
   - OneLake data will be integrated into Windows File Explorer.
   - Navigate through folders and files as with any other directory.

    <img src=images/0z0usuc1.jpg />

   - To **download a file**, double-click on it.
   - To sync changes made outside of File Explorer,** right-click** on the item or subfolder and select **Sync from OneLake**.

    <img src=images/mdssvckf.jpg />

You have successfully logged in to **Microsoft OneLake File Explorer** and explored its features. You can now manage your OneLake data directly from Windows File Explorer.

- [ ]  Tick this box to indicate OneLake File Explorer setup is complete. 

## Power BI Desktop

1. **Launch Power BI Desktop:**
   - Click the **Power BI Desktop** icon from the Taskbar or search for **Power BI Desktop** in the Windows Start menu.
    <img src=images/twynbq5a.jpg />

2. **Signin to Power BI Desktop:**
   - Upon launching, click **Sign In**.
   
    <img src=images/gkappgt2.jpg />

   - Enter the **Email Address**: **Username** and click **Continue**.

    <img src=images/9zfamnhu.jpg />
   - Pick the account from the popup, enter the **password**: **Password** and click **Sign in**. 

   - (Only if it appears) Select **No, Sign in to this app only** for quicker response (click Done if already signed in).

    <img src=images/7v161kzz.jpg />

   - Click the **File** menu to create a **Blank report** if one is not automatically created for you.

    <img src=images/gorz8fm7.jpg />

3. **Explore Copilot with Power BI Desktop:**
   - Click on **Copilot** under the **Home menu** to select the **Workspace** and activate Copilot (**Note**: Copilot works only with **F64 and above SKUs** + Power BI license Pro/PPU/Trial for authoring).

    <img src=images/2f3yvrp8.jpg />

4. **Connect to the Semantic model:**

    - Navigate to the **Home menu**, click **Get Data**, and select **More...** to connect data from multiple sources.

    <img src=images/94mvesxr.jpg />

    - In the pop-up window, click **Microsoft Fabric** to choose the respective data stores. For this exercise, use **Power BI semantic models** and click **Connect**.

    <img src=images/16goudmr.jpg />

    - Select the **Power BI semantic model** you want to connect with from the list if you have more than one, create visuals, and then click **Connect** in the Semantic model selection window.

    <img src=images/njh6jcdb.jpg />

    - (Only if it appears) Choose **Keep setup** in the popup.

    <img src=images/o91v8edm.jpg />

5. **Build a report using Copilot:**

    - You can build the reports manually and use Copilot to suggest and create them based on your selection. Just follow along with Copilot's instructions and responses.

    <img src=images/m2lzaut4.jpg />

    - Ask Copilot to Suggest content for a new report page.
    
        ***##Copilot Prompt or Press Respective Suggestions:***

        ```
        Suggest content for a new report page
        ```

    - Ask Copilot to create a report from the selection with your enhanced prompt.

        ***##Copilot Prompt or Press Create from the suggestions:***

        ```
        Create a page to examine the age distribution of customers.
        ```

Now, you can explore **Copilot** with **Semantic Model** and Prompt in **Power BI Desktop**.

- [ ]  Tick this box to indicate Power BI Desktop setup is complete. 

## VS Code

1. **Configure VS Code for Data Science:**

    - Launch **Synapse** extension, Click **Select Workspace** and select **Set** in the popup window to set a local work folder. 

        <img src=images/fc6fuu3m.jpg />

    - Click **Sign in** from the popu up window.

        <img src=images/8d7q1um1.jpg />

        If prompted, the sign-in process will occur in the browser window in the background. Please follow the sign-in steps as needed.

        <img src=images/kh4eg113.jpg />
    
    - Create a **New folder** named *FabricVSC* in the file explorer popup, and then **Select As Local Work Folder**.

        <img src=98u3w7m1.jpg](images/98u3w7m1.jpg />
2. **Explore the Fabric Artifacts in the VS Code IDE:**

    - In VS Code, **Select Workspace** for the Synapse extension and choose the appropriate **ws-models2outcomes-##** from the dropdown menu.

        <img src=images/jxjlexby.jpg />
    
    - Explore the Fabric Artifacts in your local VS Code IDE by **expanding the artifact types**.

        <img src=images/4d3q8qk3.jpg />
    
    - Launch the notebooks by clicking **Download**, then select **Open notebook folder**.

        <img src=images/a0fzxyz2.jpg />

        *You can modify the code and push it back to the **Fabric workspace** for execution, and run it locally if you have **local Kernels** available in VS Code. If you would like to access the **Fabric Spark Kernel**, complete the installation of the **Synapse-Remote** extension as described below in "**4. (Optional) Install and Execute notebook with Fabric/Spark using Synapse-Remote extension**"*.

        <img src=images/xtp6qjgj.jpg />
<!--
    - Launch **Microsoft Fabric** extension, Click **Sign in to Fabric** and select **Allow** in the popup window. 
    <img src=8mlwwm34.jpg](images/8mlwwm34.jpg />

    - **Pick an account** from the redirected browser window. If you're not signed in, please follow the **sign-in** process.

        <img src=ictjkecf.jpg](images/ictjkecf.jpg />

    - Switch back to the **VS Code** and Click **Select Fabric workspace**.

        <img src=8j8u775f.jpg](images/8j8u775f.jpg />

-->

3. **Explore the Lakehouse data from VS Code:**

    - View the lakehouse table by clicking **Preview table**.

        <img src=images/svnm8fkc.jpg />
    
    - Download the file by clicking **Download** icon of respective file to local directory.

        <img src=images/zxht8rs7.jpg />

4. **(Optional) Install and Execute notebook with Fabric/Spark Spark using Synaspe-Remote extension:**

    - Navigate to the **Extensions** in VS Code and search for **Fabric Data Engineering - Remote** to install it. Then, follow the sign-in and workspace selection process as described in the previous one.

        <img src=images/47mlxre8.jpg />

    - Once connected to the workspace, you should be able to access the notebook. Click **Synapse Remote: Open Resource In Browser** and connect to Synapse Spark remote Kernels to execute the notebooks.

        <img src=images/e5kvqifc.jpg />

Exploring Fabric in the local VS Code IDE allows you to seamlessly manage and execute your data science projects with ease.

- [ ]  Tick this box to indicate VS Code setup is complete. 

===

# Exercise 3 - Copilot Data Science Experience (Enhanced Customer Churn sample with Copilot!)

This lab consists of **five different sections**/exercises and here is an overview. You are currently in **Exercise 3 - Copilot Data Science Experience (Optional)** exercise.

- **[Configure Workspace](#configure-workspace)**
- **[Exercise 1 - Data Science in Fabric](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part1 - Data Science in Fabric (Notebook Setup)](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part2 - Data Science in Fabric (Continued)](#exercise-1---data-science-in-fabric-continued)**
- **[Exercise 2 - Client Tools (Optional)](#exercise-2---client-tools)**
- **[Exercise 3 - Copilot Data Science Experience (Optional)](#exercise-3---copilot-data-science-experience-enhanced-customer-churn-sample-with-copilot)**
- **[Exercise 4 - Power BI Visualization + PBI Copilot (Optional)](#exercise-4---power-bi-visualization--pbi-copilot)**
- **[Thank You Note and Feedback Request!](#feedback-your-feedback-is-valuable)**

In this exercise, you'll leverage Copilot within a Fabric Notebook to enhance your data science workflow, focusing on a customer churn prediction model. 

**ATTENTION**: *AI-generated code may contain errors or unintended operations. Carefully review the code in this cell before running it.* Additionally, Copilot may not function properly if overloaded by multiple users simultaneously.

Let's create a **new notebook** from the **Lakehouse** Explorer and dive into the **Copilot **Data Science notebook experience.

<img src=g4yv2uw4.jpg](images/g4yv2uw4.jpg />

1. **Create a New Notebook**:

    Open the Lakehouse, click on **Open notebook**, and then select **New notebook**.

    <img src=images/rfg51nq2.jpg />

2. **Access Copilot**:
    
    After creating a new notebook, expand the Lakehouses to view tables and files, then click on **Copilot** in the notebook ribbon.

    <img src=images/oban564n.jpg />

    Click on **Get Started** in the Copilot chat window. You'll notice a new cell added at the top of the notebook, which is essential for activating Copilot in the notebook. Make sure to **execute this first cell to activate Copilot** and allow it to understand the notebook's context.

   <img src=images/6dyyepkq.jpg />

   **Note:** *Copilot must be enabled by the Administrator in the Fabric Admin Portal settings, and you need a minimum F64 SKU or above*.

3. **Load raw data into a DataFrame**:

    Expand your **Lakehouse** and the **Files** used in the previous excercise. After successfully executing the first cell, right click on **churn.csv** and select **Load data**, then **Spark**.

    <img src=images/5t4w8v5a.jpg />

    Load the **churn.csv** and displays the records.

    <img src=images/bpfaspuy.jpg />

    
4. **Get Copilot in Action**:

    Enter the following prompt in Copilot to analyze and get insights about the data.

    ***##Copilot Prompt:***

    `Analyze df and provide insights about the data`

    <img src=images/8civekw6.jpg />

5. **Generate Code with Natural Language Instructions**:

    Use the `%%code` magic command to write natural language instructions to generate code.

    ***##Copilot Prompt:***

    ```
    %%code
    Load churn.csv from the Files folder into a pandas dataframe and then print the first 5 records
    ```

    <img src=images/06oc341b.jpg />

6. **Use Copilot Chat Pane**:

    You can also ask the same instruction in the **Copilot Chat pane** on the right side. Once the code is generated, copy and add it to the notebook.

    ***##Copilot Prompt:***
    ```
    Generate a code for calculating the percentage of customers who have churned in the dataset and display the results.
    ```

    <img src=images/zzmut0d8.jpg />

    Alternatively, you can continue using the `%%code` magic **within the notebook cell**.

    ***##Copilot Prompt:***

    ```
    %%code
    Generate a code for calculating the percentage of customers who have churned in the dataset and display the results.
    ```
    
7. **Data Cleansing**:

    Perform Data Cleansing with the help of Copilot.

    ***##Copilot Prompt:***

    ```
    %%code
    delete null values and duplicate values from the df dataframe. Drop columns 'RowNumber', 'CustomerId', 'Surname' from the df dataframe
    ```

    <img src=images/0wuusz0i.jpg />

8. **Perform Exploratory Data Analysis**:

    Show data distributions using Copilot.
   
    ***##Copilot Prompt:***

    ```
    %%code
    show the data distribution from all features in the df dataframe as charts
    ```

    <img src=images/c2hkfvys.jpg />

    Create a correlation chart using Copilot. 
   
    ***##Copilot Prompt:***

    ```
    %%code
    create a correlation chart with 'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited' features from the df dataframe
    ```

    <img src=images/pbabu9x6.jpg />

9. **Feature Engineering**:

    ***##Copilot Prompt:***

    ```
    %%code
    one hot encode Geography and Gender features from df
    ```

    <img src=images/n9zftf02.jpg />

    If you encounter an error, you can **regenerate** the result by running the prompt again or copy and paste the **error message into the Copilot chat** window to resolve it.

    <img src=images/vpen1jbb.jpg />

10. **Model Training and Testing**:

    Train the random forest model using Copilot. 
   
    ***##Copilot Prompt:***

    ```
    %%code
    create a random forest classification model for customer churn using the 'Exited' feature for prediction
    ```

    Create a confusion matrix using Copilot.
   
    ***##Copilot Prompt:***

    ```
    %%code
    create a confusion matrix
    ```

This exercise demonstrates how Copilot can streamline your data science tasks, making it easier to build and refine models.

- [ ]  Tick this box to indicate Copilot Data Science Experience is complete. 

===

# Exercise 4 - Power BI visualization + PBI Copilot 

This lab consists of **five different sections**/exercises and here is an overview. You are currently in **Exercise 4 - Power BI visualization + PBI Copilot (Optional)** exercise.

- **[Configure Workspace](#configure-workspace)**
- **[Exercise 1 - Data Science in Fabric](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part1 - Data Science in Fabric (Notebook Setup)](#exercise-1---data-science-in-fabric-notebook-setup)**
    - **[Part2 - Data Science in Fabric (Continued)](#exercise-1---data-science-in-fabric-continued)**
- **[Exercise 2 - Client Tools (Optional)](#exercise-2---client-tools)**
- **[Exercise 3 - Copilot Data Science Experience (Optional)](#exercise-3---copilot-data-science-experience-enhanced-customer-churn-sample-with-copilot)**
- **[Exercise 4 - Power BI Visualization + PBI Copilot (Optional)](#exercise-4---power-bi-visualization--pbi-copilot)**
- **[Thank You Note and Feedback Request!](#feedback-your-feedback-is-valuable)**


## Power BI Service + Copilot

1. **Create a Semantic Model from Lakehouse:**

    Click on the Lakehouse table, then click **New Semantic Model** at the top. Select the table under the dbo schema and provide a name to **Confirm** as covered in the first exercise. 

    <img src=images/o0vnrltc.jpg />

2. **Create a Blank report:**

    Return to the **Workspace**, locate the newly created **semantic model**, click on the **more options (...)** menu, and select **Create report**.

    <img src=images/0dib32lx.jpg />

3. **Activate Copilot:**

    After creating the blank report on top of semantic model, click on **Copilot** in the top ribbon, then click **Get Started** in the Copilot chat window, similar to the Data Science experience. (**Note**: Copilot works only with **F64 and above SKUs** + Power BI license Pro/PPU/Trial for authoring).

    <img src=images/cne0srx3.jpg />

4. **Build a report using Copilot:**

    ***##Copilot Prompt or Press Respective Suggestions:***

    ```
    Suggest content for a new report page
    ```

    ***##Copilot Prompt or Press Create from the suggestions:***

    ```
    Create a page to examine the age distribution of customers.
    ```

    <img src=images/y253del9.jpg />

The Copilot experience in Power BI streamlines report creation, making data insights more accessible and actionable. Enjoy the enhanced productivity and ease of use that Copilot brings to your data analysis tasks!

- [ ]  Tick this box to indicate Power BI Service + Copilot is complete. 

## Power BI Desktop + Copilot 
(if you have not completed this in **Client tools** section.)

1. **Launch Power BI Desktop:**
   - Click the **Power BI Desktop** icon from the Taskbar or search for **Power BI Desktop** in the Windows Start menu.
        <img src=images/twynbq5a.jpg />

2. **Signin to Power BI Desktop:**
   - Upon launching, click **Sign In**.
   
       <img src=images/gkappgt2.jpg />

   - Enter the **Email Address**: **Username** and click **Continue**.

        <img src=images/9zfamnhu.jpg />
   - Pick the account from the popup, enter the **password**: **Password** and click **Sign in**. 

   - (Only if it appears) Select **No, Sign in to this app only** for quicker response (click Done if already signed in).

        <img src=images/7v161kzz.jpg />

   - Click the **File** menu to create a **Blank report** if one is not automatically created for you.

        <img src=images/gorz8fm7.jpg />

3. **Explore Copilot with Power BI Desktop:**
   - Click on **Copilot** under the **Home menu** to select the **Workspace** and activate Copilot (**Note**: Copilot works only with **F64 and above SKUs** + Power BI license Pro/PPU/Trial for authoring).

        <img src=images/2f3yvrp8.jpg />

4. **Connect to the Semantic model:**

    - Navigate to the **Home menu**, click **Get Data**, and select **More...** to connect data from multiple sources.

        <img src=images/94mvesxr.jpg />

    - In the pop-up window, click **Microsoft Fabric** to choose the respective data stores. For this exercise, use **Power BI semantic models** and click **Connect**.

        <img src=images/16goudmr.jpg />

    - Select the **Power BI semantic model** you want to connect with from the list if you have more than one, create visuals, and then click **Connect** in the Semantic model selection window.

        <img src=images/njh6jcdb.jpg />

    - (Only if it appears) Choose **Keep setup** in the popup.

        <img src=images/o91v8edm.jpg />

5. **Build a report using Copilot:**

    - You can build the reports manually and use Copilot to suggest and create them based on your selection. Just follow along with Copilot's instructions and responses.

        <img src=images/m2lzaut4.jpg />

    - Ask Copilot to Suggest content for a new report page.
    
        ***##Copilot Prompt or Press Respective Suggestions:***

        ```
        Suggest content for a new report page
        ```

    - Ask Copilot to create a report from the selection with your enhanced prompt.

        ***##Copilot Prompt or Press Create from the suggestions:***

        ```
        Create a page to examine the age distribution of customers.
        ```

Now, you can explore **Copilot** with **Semantic Model** and Prompt in **Power BI Desktop**.

- [ ]  Tick this box to indicate Power BI Desktop + Copilot is complete. 

**Congratulations** on successfully completing all the lab exercises! We hope you enjoyed the experience. Please feel free to share your feedback. We look forward to seeing you again soon.

===

# Feedback (Your feedback is valuable) 

###**Thank you for taking the time to complete this lab!!** 

<!-- ### End User Licensing Agreement

By using this lab solution, you agree to the terms outlined in the End User License Agreement (EULA), which you can review by following this ^[link][Reference Link].

> [Reference Link]:
> !instructions[](https://raw.githubusercontent.com/LODSContent/SkillableLabSolutions/main/EULA%20062424.md) -->

</br>

**[Back to Home](#configure-workspace)**
