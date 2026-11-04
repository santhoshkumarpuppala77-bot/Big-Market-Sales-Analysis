# 🛒 BigMart Sales Analysis & Prediction

## 📌 Project Overview

The BigMart Sales Prediction project aims to develop a Machine Learning regression model that predicts the sales of individual products across different BigMart outlets.

The dataset contains historical sales information for 1,559 products across 10 different stores. It includes product-related and outlet-related attributes that influence sales.

The primary objective is to identify important sales patterns and build a predictive model that helps businesses estimate product sales, improve inventory management, and make data-driven decisions.

## 🎯 Business Problem

Retail businesses face challenges in accurately estimating product sales across different outlets. Sales vary depending on product price, visibility, category, outlet size, location, and outlet type.

Inaccurate sales estimation can lead to:
- Overstocking and increased inventory costs.
- Understocking and lost sales opportunities.
- Inefficient inventory planning.
- Poor product placement and pricing decisions.

This project uses Machine Learning to estimate product sales and support better retail business decisions.

## 🎯 Project Objectives

- Analyze product-level and outlet-level sales patterns.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and inconsistent categorical data.
- Perform feature engineering and feature selection.
- Build and compare multiple regression algorithms.
- Optimize the best-performing model using Optuna.
- Deploy the final model using Streamlit.

## 📂 Dataset Information

| Attribute | Description |
|---|---|
| Dataset | BigMart Sales Prediction |
| Domain | Retail & E-commerce |
| Training Records | 8,523 |
| Products | 1,559 |
| Outlets | 10 |
| Input Features | 10 original features |
| Target Variable | Item_Outlet_Sales |
| Problem Type | Supervised Learning – Regression |

### Features Description

| Feature | Description |
|---|---|
| Item_Weight | Weight of the product |
| Item_Fat_Content | Fat content of the product |
| Item_Visibility | Percentage of display area allocated to the product |
| Item_Type | Category of the product |
| Item_MRP | Maximum Retail Price |
| Outlet_Identifier | Unique outlet identification |
| Outlet_Size | Size of the outlet |
| Outlet_Location_Type | Location category of the outlet |
| Outlet_Type | Type of retail outlet |
| Outlet_Age | Age of the outlet |

**Target Variable:** `Item_Outlet_Sales`

The target variable represents the sales of a particular product at a particular outlet.

## ⚙️ Data Preprocessing

The following preprocessing techniques were implemented:

### 1. Data Cleaning
- Identified and handled missing values.
- Standardized inconsistent categorical values.
- Removed unnecessary columns.

### 2. Missing Value Imputation
- Numerical features: Median imputation.
- Categorical features: Mode imputation.

### 3. Feature Engineering
Created additional features:
- `Item_Category_Code`
- `Outlet_Age`

### 4. Categorical Encoding

**One-Hot Encoding:**
- Item_Type
- Outlet_Type
- Outlet_Identifier
- Item_Category_Code

**Ordinal Encoding:**
- Item_Fat_Content
- Outlet_Size
- Outlet_Location_Type

### 5. Pipeline Implementation

Used Scikit-learn:
- SimpleImputer
- OneHotEncoder
- OrdinalEncoder
- ColumnTransformer
- Pipeline

### 6. Train-Test Split

- Training Data: 80%
- Testing Data: 20%

## 🤖 Model Building

Eight regression algorithms were implemented and evaluated:

1. Linear Regression
2. K-Nearest Neighbors (KNN) Regressor
3. Support Vector Regressor (SVR)
4. Decision Tree Regressor
5. Random Forest Regressor
6. AdaBoost Regressor
7. XGBoost Regressor
8. CatBoost Regressor

The models were compared using regression evaluation metrics to identify the best-performing algorithm.

## 📊 Model Evaluation

The following evaluation metrics were used:

- **R² Score:** Measures how much variation in sales is explained by the model.
- **RMSE:** Measures prediction error while penalizing larger errors.
- **MAE:** Measures the average absolute difference between actual and predicted sales.

### Best Model: CatBoost Regressor

CatBoost Regressor achieved the best overall performance among the evaluated models.

| Evaluation Metric | Score |
|---|---:|
| R² Score | 0.5744 |
| RMSE | 1,119.11 |
| MAE | 789.34 |

## 🔧 Cross-Validation & Hyperparameter Tuning

### Cross-Validation

**Method:** K-Fold Cross Validation

- Number of folds: 5
- Model: CatBoost Regressor
- Evaluation metric: RMSE

### Hyperparameter Optimization

**Optimization Tool:** Optuna

The following hyperparameters were explored:

| Hyperparameter | Search Range |
|---|---|
| Iterations | 300–1000 |
| L2 Leaf Regularization | 0.001–10 |
| Depth | 3, 5, 7 |
| Learning Rate | 0.01–0.1 |

Hyperparameter tuning helped improve model performance and address overfitting.

## 🚀 Model Deployment

The trained CatBoost regression model was integrated into a Streamlit application.

### Application Features
- User-friendly interface.
- Product and outlet feature inputs.
- Real-time sales prediction.
- Automated prediction using the trained model.

**Live Application:**  
https://big-market-sales-analysis.streamlit.app/

## 💡 Key Business Insights

1. Developed a Machine Learning regression system to predict product sales across different outlets.
2. Identified the importance of product and outlet characteristics in sales prediction.
3. Compared eight regression algorithms to identify the best-performing model.
4. CatBoost achieved an R² score of 0.5744 with an RMSE of 1,119.11.
5. Integrated the trained model into a Streamlit application.
6. The prediction system can support inventory planning and data-driven retail decisions.

## 📌 Business Recommendations

1. **Optimize Inventory Management:** Use predicted sales to maintain appropriate stock levels.
2. **Focus on High-Demand Products:** Ensure sufficient availability of products with higher predicted sales.
3. **Outlet-Specific Stock Planning:** Adjust inventory according to outlet size, location, and type.
4. **Improve Product Placement:** Strategically position products to improve visibility.
5. **Develop Pricing Strategies:** Analyze pricing and predicted sales to identify promotional opportunities.
6. **Monitor Sales Trends:** Compare actual sales with predicted sales to improve future planning.

## ⚠️ Challenges Faced

- Missing values in Item_Weight and Outlet_Size.
- Inconsistent categorical values in Item_Fat_Content.
- Outliers in numerical features.
- Handling multiple categorical features.
- Identifying meaningful relationships between product and outlet attributes.
- Preventing model overfitting.
- Improving prediction accuracy through feature engineering and hyperparameter tuning.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and ML |
| XGBoost | Gradient boosting |
| CatBoost | Regression modeling |
| Optuna | Hyperparameter optimization |
| Streamlit | Model deployment |
| Jupyter Notebook | Development environment |

## 📁 Project Structure

```text
BigMart-Sales-Analysis/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── BigMart_Sales_Analysis.ipynb
│
├── models/
│   └── catboost_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Improve prediction accuracy through advanced feature engineering.
- Experiment with additional ensemble techniques.
- Explore stacking and blending of regression models.
- Perform more extensive hyperparameter optimization.
- Incorporate additional sales and customer-related information.
- Monitor prediction performance using actual sales data.

## 👨‍💻 Author

**Santhosh Kumar**

- LinkedIn: https://www.linkedin.com/in/puppala-santhosh-kumar
- GitHub: https://github.com/santhoshkumarpuppala77-bot

## 🏁 Conclusion

The BigMart Sales Prediction project demonstrates how Machine Learning can be applied to retail sales forecasting. After evaluating eight regression algorithms, CatBoost Regressor was selected as the best-performing model, achieving an R² score of 0.5744 and an RMSE of 1,119.11. The model was deployed using Streamlit, providing an interactive application for predicting product sales. This solution can help businesses improve inventory planning, reduce potential stock-related issues, and make better data-driven decisions.
