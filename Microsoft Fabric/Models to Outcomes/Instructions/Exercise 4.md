# **Advanced Analytics: From Models to Outcomes with Microsoft Fabric - Lab 4**

![](./images/png4.png)

## Introduction
In this lab, you will explore how to use Copilot in Power BI Service and Power BI Desktop. You will create a semantic model from a Lakehouse table and use Copilot to generate reports using simple prompts.

1. On the Fabric/PowerBI portal, click on the **ws-<inject key="Deployment ID" enableCopy="false"/>** workspace and select the **CustomersLH** Lakehouse.

    ![](./images/29042025(66).png)

5. In the Home tab, click **New semantic model**.

   - Enter **ChurnResultDS** as the name for the semantic model. 

   - Under `dbo > Tables`, check the box for **`df_pred_results`**.   

   - Click **Confirm** to create the Power BI semantic model linked to the prediction results.

     ![](./images/29042025(81).png)

     ![](./images/29042025(82).png)

1. Return to the **Workspace** and locate the **Semantic model** you created.  

   - Click the **more options (...)** next to it, then select **Create report**.  

   - This action will open the Power BI report **authoring page**, where you can begin designing your report.

     ![](./images/29042025(83).png)

3. After creating the blank report on top of semantic model, click on **Copilot** in the top ribbon, then click **Get Started** in the Copilot chat window, similar to the Data Science experience.

    ![](./images/29042025(84).png)

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
    
        ***## Copilot Prompt or Press Respective Suggestions:***

        ```
        Suggest content for a new report page
        ```

    - Ask Copilot to create a report from the selection with your enhanced prompt.

        ***## Copilot Prompt or Press Create from the suggestions:***

        ```
        Create a page to examine the age distribution of customers.
        ```

## Summary
In this lab, you have explored how to connect to a semantic model and use Copilot to quickly build reports in Power BI. You’ve seen how AI can assist in creating report pages, making data analysis faster and easier.

### You have successfully completed the lab.