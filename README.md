# Customer Segmentation & Behavioral Analytics (iFood Dataset)

An unsupervised machine learning and advanced exploratory data analysis (EDA) project designed to perform data-driven market segmentation, discover hidden purchasing behaviors, and isolate distinct consumer archetypes using K-Means clustering and Principal Component Analysis (PCA).

---

## 📊 Business Objective
In modern e-commerce and retail marketing, a "one-size-fits-all" approach leads to high marketing spend and low conversion rates. This project engineers a behavioral segmentation framework to:
1.  **Perform Rigorous Data Auditing:** Detect variance constants and filter out extreme outlier spending anomalies.
2.  **Uncover Statistical Associations:** Run Point-Biserial correlation tests to measure how demographic categories impact customer lifetime value.
3.  **Optimize Cluster Separation:** Implement dual mathematical heuristics (Elbow Method and Silhouette Analysis) to systematically discover the optimal number of consumer segments.
4.  **Define Actionable Archetypes:** Map cluster attributes back to real-world behavioral traits to drive personalized CRM strategies.

---

## 🛠️ Tech Stack & Architecture
*   **Data Wrangling & Pipeline Execution:** `Pandas`, `NumPy`
*   **Statistical Significance Testing:** `SciPy (Stats)` (`pointbiserialr`)
*   **Unsupervised Machine Learning:** `Scikit-Learn` (`KMeans`, `StandardScaler`, `decomposition.PCA`, `silhouette_score`)
*   **Advanced Visualizations:** `Matplotlib`, `Seaborn`

## 📈 Methodology & Key Analytical Steps

### 1. Data Ingestion & Quality Controls:
*   **Feature Truncation:** Evaluated cardinality across the dataset and systematically dropped zero-variance constant features (`Z_CostContact`, `Z_Revenue`) to protect model processing efficacy.
*   **Outlier Management:** Utilized Interquartile Range (IQR) filtering on the total spend distribution (`MntTotal`) to eliminate highly skewed anomalies, ensuring a clean geometric space for K-Means distance calculations.

### 2. Statistical Insights & Feature Selection
Before modeling, the relationships between customer attributes and spending were validated using **Point-Biserial Correlation ($r_{pb}$)** to capture relationships between binary categorical markers and the continuous spending variable:
*   **Education Impact:** The analysis proved a statistically significant positive correlation between advanced degree holders (PhD/Master) and high-value spending platforms, while basic education profiles correlated negatively.
*   **Clustering Feature Formulation:** Based on distribution skews and correlations, three critical, orthogonal axes were isolated for behavioral modeling: `Income`, `MntTotal`, and an engineered binary indicator `In_relationship`.

### 3. Dimensionality Reduction & Cluster Optimization
*   **Feature Scaling:** Deployed `StandardScaler` to bring all features into the exact same scale, preventing high-magnitude income scales from overwhelming spatial distance metrics.
*   **Dimensionality Reduction:** Compressed the feature space into a 2D subspace using **Principal Component Analysis (PCA)** to visualize variance patterns effortlessly.
*   **Mathematical Hyperparameter Tuning:** Instead of guessing the cluster count, the optimization window ($K \in [2, 9]$) was run through both **Inertia (Elbow Method)** and **Silhouette Score Heuristics**. Both converged explicitly at $K=4$, pointing to the definitive optimal cluster topology.

---

## 👥 Engineered Consumer Archetypes

By grouping the final model labels with raw feature sets, four distinct consumer archetypes were isolated:

| Cluster | Household Dynamic | Income Profile | Spend Velocity | Primary Product Focus |
| :---: | :--- | :--- | :--- | :--- |
| **0** | Singles / Divorced | Moderate-Low | Low Spenders | Basic Staples & High Web Engagement |
| **1** | Couples / Families | High Income | Premium Spenders | Heavy Wine & Meat Consumption |
| **2** | Couples / Families | Moderate-Low | Value Browsers | High Deals & Bargain Purchases |
| **3** | Singles / Divorced | High Income | Affluent Elite | Premium Multi-Category / Catalog Buyers |

Results:

This section contains the results of the K-means clustering analysis, which aimed to identify distinct customer segments based on the total amount of purchases they made (MntTotal). The analysis utilised 'Income' and 'In_relationship' features.

Optimal number of clusters = 4

The Elbow Method and Silhouette Analysis suggested 4 clusters (k=4). The elbow method highlighted the number of 4 or 5 clusters as a reasonable number of clusters. The silhouette score analysis revealed a peak silhouette score for k=4.

Cluster Characteristics:

Cluster 0: High value customers in relationship (either married or together)

This cluster represents 26% of the customer base
These customers have high income and they are in a relationship

Cluster 1: Low value single customers

This cluster represents 21% of the customer base
These customers have low income and they are single
Cluster 2: High value single customers
This cluster represents 15% of the customer base
These customers have high income and they are single
Cluster 3: Low value customers in relationship
This cluster represents 39% of the customer base
These customers have low income and they are in a relationship

Recommendations:

Based on the clusters, tailored marketing strategies can be created. Customers from these segments will have different interests and product preferences.

Marketing Strategies for Each Cluster:

Cluster 0: High value customers in relationship (either married or together)

Preliminary analysis showed that high income customers buy more wines and fruits.
A tailored campaign to promote high quality wines may bring good results.
This cluster contains customers in relationship, family-oriented promo-images should be quite effective for this audience.

Cluster 1: Low value single customers

Promos with discounts and coupons may bring good results for this targeted group.
Loyalty program may stimulate these customers to purchase more often.

Cluster 2: High value single customers

Similar to the Cluster 0, these customers buy a lot of wines and fruits.
This cluster contains single customers. Promo images with friends, parties or single trips may be more efficient for single customers

Cluster 3: Low value customers in relationship

This cluster has the highest percentage of our customers (39%).
Family offers and discounts may influence these customers to make more purchases

Opportunities for the further analysis:

Further exploration on how children influence on the consumed products
Further analysis on the influence of education
analysis of frequent buyers
Analysis of sales channels, e.g. store, website, etc.
Analysis of the response to the marketing campaigns
It would be great to add gender data to the dataset
Test different clustering algorithms

Thank you!