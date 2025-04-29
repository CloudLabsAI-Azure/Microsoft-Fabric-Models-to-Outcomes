# **Advanced Analytics: From Models to Outcomes with Microsoft Fabric - Lab 2**


## Contents
- Introduction
    - Task 1: Setting Up and Exploring OneLake File Explorer
    - Task 2: Creating Reports with Power BI Desktop and Copilot
    - Task 3: Configuring VS Code for Microsoft Fabric Development
- Summary

## Introduction:

In this lab, you will explore the core tools and capabilities of Microsoft Fabric, including OneLake File Explorer, Power BI Desktop, and Visual Studio Code (VS Code) with the Synapse extension. You will learn how to access and manage your OneLake data directly from Windows File Explorer, build and analyze reports using Copilot in Power BI, and configure VS Code to develop and interact with Fabric artifacts and Lakehouse data. This hands-on lab is designed to provide a foundational understanding of how these tools integrate to support end-to-end data science and analytics workflows within the Microsoft Fabric ecosystem.

## Task 1: Setting Up and Exploring OneLake File Explorer

This application automatically syncs all Microsoft OneLake items that you have access to in Windows File Explorer. To log in to Microsoft OneLake File Explorer and explore its features.

1. On the Lab-vm, open the **Search bar** menu, type **OneLake** in the search bar, and select it from the search results.

    ![](./images/29042025(46).png)

2. Upon launching, a login prompt will appear. Enter your email/username and click **Next**.

    - **Email/Username:** <inject key="AzureAdUserEmail"></inject>
   
      ![](./images/29042025(40).png)
   
1. Now enter the following password and click on **Sign in**.
 
   - **Password:** <inject key="AzureAdUserPassword"></inject>

        ![](./images/29042025(41).png)
   
1. When prompted with the **"Automatically sign in to all desktop apps and websites on this device?"** screen, click **Yes, all apps**.

    ![](./images/29042025(42).png)

1. Once your account has been added successfully, you will see the **"You're all set!"** confirmation screen. Click **Done** to complete the login process.

    ![](./images/29042025(43).png)

1. Navigate to the **File Explorer** located on the taskbar. On the left side of the File Explorer, click on **OneLake - Microsoft**, and then proceed to **ws-<inject key="Deployment ID" enableCopy="false"/>**. From there, go to **CustomersLH.Lakehouse** and select **Tables**. You should now be able to see the two tables folder.

    ![](./images/29042025(44).png)

1. Navigate to the **ws-<inject key="Deployment ID" enableCopy="false"/>** folder. To **download a file**, simply double-click on it. If you need to sync changes made outside of File Explorer, **right-click** on the item or subfolder and select **Sync from OneLake**.

    ![](./images/29042025(45).png)

You have successfully logged in to **Microsoft OneLake File Explorer** and explored its features. You can now manage your OneLake data directly from Windows File Explorer.

## Task 2: Creating Reports with Power BI Desktop and Copilot

1. Open the **Power BI Desktop** located on the desktop of your lab environment.

2. Once the "Enter your email address" dialog appears, copy the **Username** and paste it into the **Email** field of the dialog and select **Continue**.

   * Email/Username: <inject key="AzureAdUserEmail"></inject>

     ![](./images/29042025(47).png)
  
3. On the Sign into Microsoft Azure tab, you will see the login screen, enter the following Email/ Username and then click on **Next**.

   * Email/Username: <inject key="AzureAdUserEmail"></inject>

     ![](./images/29042025(40).png)

4. Now enter the following **Password** and click on **Sign in**.

    * Password: <inject key="AzureAdUserPassword"></inject>

      ![](./images/29042025(41).png)

1. When prompted with the **"Automatically sign in to all desktop apps and websites on this device?"** screen, click **Yes, all apps**.

    ![](./images/29042025(42).png)

1. Once your account has been added successfully, you will see the **"You're all set!"** confirmation screen. Click **Done** to complete the login process.

    ![](./images/29042025(43).png)

1. Click the **File** menu to create a **Blank report** if one is not automatically created for you.

    ![](./images/29042025(48).png)

1. Click on **Copilot** under the **Home** menu, then select the workspace **ws-<inject key="Deployment ID" enableCopy="false"/>** and click **OK**.

     ![](./images/29042025(50).png)

1. Navigate to the **Home menu**, click **Get data**, and select **More...** to connect data from multiple sources.

      ![](./images/29042025(51).png)

1. In the pop-up window, click **Microsoft Fabric** to choose the respective data stores. For this exercise, use **Power BI semantic models** and click **Connect**.

    ![](./images/29042025(52).png)

1. In the **OneLake catalog** window, locate and select **ChurnDS** from the list of available semantic models.  

1. After selecting **ChurnDS**, click the **Connect** button in the bottom-right corner of the window to proceed.

    ![](./images/29042025(53).png)

5. You can build the reports manually and use Copilot to suggest and create them based on your selection. Just follow along with Copilot's instructions and responses.

    ![](./images/29042025(54).png)

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

## Task 3: Configuring VS Code for Microsoft Fabric Development

1. Open the **Visual Studio Code** located on the desktop of your lab environment.

1. Navigate to the **Extensions** in VS Code and search for **Fabric Data Engineering VS Code** to install it.

    ![](./images/29042025(55).png)

1. Then, follow the sign-in Launch **Synapse** extension, Click **Select Workspace** and select **Set** in the popup window to set a local work folder. 

    ![](./images/29042025(57).png)

    >**Note:** Click **Sign in** from the popu up window.

    ![](./images/29042025(56).png)

1. If prompted, the sign-in process will occur in the browser window in the background. Please follow the sign-in steps as needed.

    ![](./images/29042025(58).png)
    
1. Create a **New folder** named **FabricVSC** in the file explorer popup, and then **Select As Local Work Folder**.

    ![](./images/29042025(59).png)

2. In VS Code, **Select** **Workspace** for the Synapse extension and choose the appropriate **ws-<inject key="Deployment ID" enableCopy="false"/>** from the dropdown menu.

    ![](./images/29042025(60).png)

1. When prompted with the **Do you trust the authors of the files in this folder?** message, check the box labeled **Trust the authors of all files in the parent folder**. Click the **Yes, I trust the authors** button to enable all features and exit restricted mode in Visual Studio Code.

    ![](./images/29042025(63).png)
   
1. Explore the Fabric Artifacts in your local VS Code IDE by **expanding the artifact types**.

    ![](./images/29042025(61).png)
    
1. Launch the notebooks by clicking **Download**, then select **Open notebook folder**.

    ![](./images/29042025(62).png)

3. To view the lakehouse table, navigate to **Lakehouse -> CustomersLH -> Tables -> df_pred_results**, then click on **Preview table**.

    ![](./images/29042025(65).png)
    
1. To download the churn file, navigate to **Files -> churn -> raw -> churn.csv**, then click the **Download** icon next to the file to save it to your local directory.

    ![](./images/29042025(64).png)

## Summary:

In this lab, you have explored the fundamental tools and features of Microsoft Fabric, including OneLake File Explorer, Power BI Desktop, and Visual Studio Code with the Synapse extension. You accessed and managed OneLake data through Windows File Explorer, built and analyzed reports using Copilot in Power BI, and configured VS Code to interact with Fabric artifacts and Lakehouse data. This lab provided a hands-on introduction to how these tools work together to enable end-to-end data science and analytics workflows within the Microsoft Fabric ecosystem.

Now, click on **Next** from the lower right corner to move on to the next page.

![alt text](image-7.png)

