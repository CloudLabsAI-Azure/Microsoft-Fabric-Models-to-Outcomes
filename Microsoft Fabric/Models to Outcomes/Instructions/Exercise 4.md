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
