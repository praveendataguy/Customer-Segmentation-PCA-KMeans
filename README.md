Customer Segmentation using K-Means & PCA

Description:
This project focuses on performing Customer Segmentation for a retail dataset (iFood) to identify distinct purchasing behaviors and demographic profiles. By understanding these segments, a business can better target its marketing efforts and optimize product offerings.

The analysis involves:

Data Cleaning: Removing outliers using the Interquartile Range (IQR) method and handling redundant features.

Exploratory Data Analysis (EDA): Visualizing distributions of income, age, and correlations between spending habits.

Feature Engineering: Creating new metrics like In_relationship to see how social status affects spending.

Machine Learning: Using K-Means Clustering optimized by the Elbow Method and Silhouette Score, with Principal Component Analysis (PCA) for dimensionality reduction and visualization.

Installation:
To run this project locally, ensure you have Python installed. You can install the required dependencies using the following steps:

Clone the repository:
git clone https://github.com/praveendataguy/Customer-Segmentation-PCA-KMeans.git

Navigate to the project directory:
cd Customer-Segmentation-PCA-KMeans

Install the required libraries:
pip install -r requirements.txt

Usage:
Ensure your dataset is located in the data/ folder and named ifood_df.csv.

Run the main analysis script:
python clustering_and_pca_model.py
The script will output statistical summaries to the console and generate several visualizations, including the final cluster scatter plot.

Results:
Through the analysis, we identified 4 distinct customer clusters based on Income, Total Spend (MntTotal), and Relationship Status:

Cluster Profiles: The segments range from high-income/high-spending "Premium" customers to budget-conscious "Value" shoppers.
PCA Visualization: By reducing features to two principal components, we can clearly see the separation between the identified groups.

Key Insight:
The Silhouette Score peaked at $K=4$, suggesting that the customer base is most naturally divided into four segments. High-spending clusters showed a strong correlation with higher income levels and specific product categories like Wines and Meat products.
