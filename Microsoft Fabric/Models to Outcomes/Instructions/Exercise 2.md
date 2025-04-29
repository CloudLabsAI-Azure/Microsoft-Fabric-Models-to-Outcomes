# Advanced Analytics: From Models to Outcomes with Microsoft Fabric - Lab 2


## Contents
- Introduction
    - Task 1: Setting Up and Exploring OneLake File Explorer
    - Task 2: Creating Reports with Power BI Desktop and Copilot
    - Task 3: Configuring VS Code for Microsoft Fabric Development
- Summary

## Introduction:

In this lab, you will explore the core tools and capabilities of Microsoft Fabric, including OneLake File Explorer, Power BI Desktop, and Visual Studio Code (VS Code) with the Synapse extension. You will learn how to access and manage your OneLake data directly from Windows File Explorer, build and analyze reports using Copilot in Power BI, and configure VS Code to develop and interact with Fabric artifacts and Lakehouse data. This hands-on lab is designed to provide a foundational understanding of how these tools integrate to support end-to-end data science and analytics workflows within the Microsoft Fabric ecosystem.

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

    - Launch **Microsoft Fabric** extension, Click **Sign in to Fabric** and select **Allow** in the popup window. 
    <img src=8mlwwm34.jpg](images/8mlwwm34.jpg />

    - **Pick an account** from the redirected browser window. If you're not signed in, please follow the **sign-in** process.

        <img src=ictjkecf.jpg](images/ictjkecf.jpg />

    - Switch back to the **VS Code** and Click **Select Fabric workspace**.

        <img src=8j8u775f.jpg](images/8j8u775f.jpg />

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

    ```python
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

    ```python
    %%code
    show the data distribution from all features in the df dataframe as charts
    ```

    <img src=images/c2hkfvys.jpg />

    Create a correlation chart using Copilot. 
   
    ***##Copilot Prompt:***

    ```python
    %%code
    create a correlation chart with 'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited' features from the df dataframe
    ```

    <img src=images/pbabu9x6.jpg />

9. **Feature Engineering**:

    ***##Copilot Prompt:***

    ```python
    %%code
    one hot encode Geography and Gender features from df
    ```

    <img src=images/n9zftf02.jpg />

    If you encounter an error, you can **regenerate** the result by running the prompt again or copy and paste the **error message into the Copilot chat** window to resolve it.

    <img src=images/vpen1jbb.jpg />

10. **Model Training and Testing**:

    Train the random forest model using Copilot. 
   
    ***##Copilot Prompt:***

    ```python
    %%code
    create a random forest classification model for customer churn using the 'Exited' feature for prediction
    ```

    Create a confusion matrix using Copilot.
   
    ***##Copilot Prompt:***

    ```python
    %%code
    create a confusion matrix
    ```

This exercise demonstrates how Copilot can streamline your data science tasks, making it easier to build and refine models.

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

**Congratulations** on successfully completing all the lab exercises! We hope you enjoyed the experience. Please feel free to share your feedback. We look forward to seeing you again soon.


