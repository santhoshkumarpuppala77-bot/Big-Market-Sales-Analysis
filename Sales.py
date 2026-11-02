
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from catboost import CatBoostRegressor

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BigMart Sales Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. APPLICATION HEADER
# ============================================================

st.title("🛒 BigMart Sales Analysis & Prediction")

st.write(
    "Explore sales performance, item trends, outlet insights, "
    "and predict Item Outlet Sales."
)

st.divider()

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛒 BigMart")
st.sidebar.subheader("Sales Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sales Analysis",
        "Item Analysis",
        "Outlet Analysis",
        "Predict Sales"
    ]
)

st.sidebar.divider()

# ============================================================
# 4. DATASET UPLOAD
# ============================================================

st.sidebar.subheader("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload BigMart CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

elif os.path.exists("train_v9rqXOR.csv"):

    df = pd.read_csv("train_v9rqXOR.csv")

else:

    st.warning(
        "Please upload your BigMart training dataset."
    )

    st.stop()


# ============================================================
# 5. DATA VALIDATION
# ============================================================

required_columns = [
    "Item_Identifier",
    "Item_Outlet_Sales"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()


df = df.copy()

df["Item_Outlet_Sales"] = pd.to_numeric(
    df["Item_Outlet_Sales"],
    errors="coerce"
)

df = df.dropna(
    subset=["Item_Outlet_Sales"]
)

if df.empty:
    st.error("No valid sales records found.")
    st.stop()


# ============================================================
# 6. SIDEBAR FILTERS
# ============================================================

st.sidebar.divider()
st.sidebar.subheader("🔎 Sales Filters")

filtered_df = df.copy()

filter_columns = [
    "Outlet_Type",
    "Item_Type",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Item_Fat_Content"
]

for col in filter_columns:

    if col in df.columns:

        options = sorted(
            df[col].dropna().astype(str).unique().tolist()
        )

        selected = st.sidebar.multiselect(
            col.replace("_", " "),
            options=options,
            default=options,
            key=f"filter_{col}"
        )

        filtered_df = filtered_df[
            filtered_df[col].astype(str).isin(selected)
        ]


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def format_sales(value):

    if pd.isna(value):
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"₹ {value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"₹ {value / 1_000:.2f}K"

    return f"₹ {value:,.2f}"


def display_metrics(data):

    total_sales = data["Item_Outlet_Sales"].sum()

    total_items = data["Item_Identifier"].nunique()

    total_outlets = (
        data["Outlet_Identifier"].nunique()
        if "Outlet_Identifier" in data.columns
        else 0
    )

    average_sales = data["Item_Outlet_Sales"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Sales",
        format_sales(total_sales)
    )

    col2.metric(
        "📦 Total Items",
        f"{total_items:,}"
    )

    col3.metric(
        "🏬 Total Outlets",
        f"{total_outlets:,}"
    )

    col4.metric(
        "📈 Average Sales",
        format_sales(average_sales)
    )


def sales_chart(data, column):

    if column not in data.columns:
        st.info(f"{column} is not available.")
        return

    result = (
        data.groupby(column, dropna=False)
        ["Item_Outlet_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(result)


def sales_table(data):

    columns = [
        col for col in [
            "Item_Identifier",
            "Item_Type",
            "Outlet_Identifier",
            "Outlet_Type",
            "Item_MRP",
            "Item_Outlet_Sales"
        ]
        if col in data.columns
    ]

    st.dataframe(
        data[columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 8. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.header("📊 Sales Overview")

    display_metrics(filtered_df)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏬 Sales by Outlet Type")

        sales_chart(
            filtered_df,
            "Outlet_Type"
        )

    with col2:

        st.subheader("📦 Sales by Item Category")

        sales_chart(
            filtered_df,
            "Item_Type"
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("📍 Sales by Outlet Location")

        sales_chart(
            filtered_df,
            "Outlet_Location_Type"
        )

    with col4:

        st.subheader("🥛 Sales by Fat Content")

        sales_chart(
            filtered_df,
            "Item_Fat_Content"
        )

    st.divider()

    st.subheader("🏆 Top 5 Selling Item Types")

    if "Item_Type" in filtered_df.columns:

        top_items = (
            filtered_df.groupby("Item_Type")
            ["Item_Outlet_Sales"]
            .sum()
            .nlargest(5)
            .reset_index()
        )

        top_items.columns = [
            "Item Type",
            "Total Sales"
        ]

        st.dataframe(
            top_items,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Recent Dataset Records")

    sales_table(filtered_df.head(20))


# ============================================================
# 9. SALES ANALYSIS PAGE
# ============================================================

elif page == "Sales Analysis":

    st.header("📈 Sales Analysis")

    display_metrics(filtered_df)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sales by Outlet")

        sales_chart(
            filtered_df,
            "Outlet_Identifier"
        )

    with col2:

        st.subheader("Sales by Outlet Size")

        sales_chart(
            filtered_df,
            "Outlet_Size"
        )

    st.divider()

    st.subheader("Item MRP vs Sales")

    if "Item_MRP" in filtered_df.columns:

        mrp_sales = (
            filtered_df.groupby("Item_MRP")
            ["Item_Outlet_Sales"]
            .mean()
            .sort_index()
        )

        st.line_chart(mrp_sales)

    st.divider()

    st.subheader("Sales by Outlet Location")

    sales_chart(
        filtered_df,
        "Outlet_Location_Type"
    )

    st.divider()

    st.subheader("Sales Records")

    sales_table(filtered_df.head(100))


# ============================================================
# 10. ITEM ANALYSIS PAGE
# ============================================================

elif page == "Item Analysis":

    st.header("📦 Item Analysis")

    display_metrics(filtered_df)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sales by Item Type")

        sales_chart(
            filtered_df,
            "Item_Type"
        )

    with col2:

        st.subheader("Sales by Fat Content")

        sales_chart(
            filtered_df,
            "Item_Fat_Content"
        )

    st.divider()

    if "Item_MRP" in filtered_df.columns:

        st.subheader("Item MRP Distribution")

        mrp_distribution = (
            filtered_df["Item_MRP"]
            .dropna()
            .value_counts(bins=15)
            .sort_index()
        )

        st.bar_chart(mrp_distribution)

    if "Item_Visibility" in filtered_df.columns:

        st.subheader("Item Visibility Distribution")

        visibility_distribution = (
            filtered_df["Item_Visibility"]
            .dropna()
            .value_counts(bins=15)
            .sort_index()
        )

        st.bar_chart(visibility_distribution)

    st.divider()

    st.subheader("Top 10 Products by Sales")

    top_products = (
        filtered_df.groupby("Item_Identifier")
        ["Item_Outlet_Sales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 11. OUTLET ANALYSIS PAGE
# ============================================================

elif page == "Outlet Analysis":

    st.header("🏬 Outlet Analysis")

    display_metrics(filtered_df)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sales by Outlet Type")

        sales_chart(
            filtered_df,
            "Outlet_Type"
        )

    with col2:

        st.subheader("Sales by Outlet Location")

        sales_chart(
            filtered_df,
            "Outlet_Location_Type"
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Sales by Outlet Size")

        sales_chart(
            filtered_df,
            "Outlet_Size"
        )

    with col4:

        st.subheader("Sales by Outlet Age")

        if "Outlet_Age" in filtered_df.columns:

            age_sales = (
                filtered_df.groupby("Outlet_Age")
                ["Item_Outlet_Sales"]
                .sum()
                .sort_index()
            )

            st.line_chart(age_sales)

    st.divider()

    st.subheader("Outlet Performance Table")

    if "Outlet_Identifier" in filtered_df.columns:

        outlet_summary = (
            filtered_df.groupby("Outlet_Identifier")
            .agg(
                Total_Sales=(
                    "Item_Outlet_Sales", "sum"
                ),
                Average_Sales=(
                    "Item_Outlet_Sales", "mean"
                ),
                Total_Items=(
                    "Item_Identifier", "nunique"
                )
            )
            .reset_index()
            .sort_values(
                "Total_Sales",
                ascending=False
            )
        )

        st.dataframe(
            outlet_summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 12. INDIVIDUAL SALES PREDICTION
# ============================================================



elif page == "Predict Sales":

    st.header("🤖 BigMart Sales Prediction")

    st.write(
        "Enter the product and outlet details to predict "
        "the expected Item Outlet Sales."
    )

    # Load existing trained model
    with open("sales.pkl", "rb") as file:
        model = pickle.load(file)

    st.divider()

    st.subheader("📦 Product & Outlet Information")

    with st.form("bigmart_prediction_form"):

        # First row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            item_weight = st.number_input(
                "Item Weight",
                min_value=0.0,
                value=12.0,
                step=0.1
            )

        with col2:
            item_fat_content = st.selectbox(
                "Item Fat Content",
                sorted(df["Item_Fat_Content"].dropna().unique())
            )

        with col3:
            item_visibility = st.number_input(
                "Item Visibility",
                min_value=0.0,
                value=0.05,
                step=0.001,
                format="%.4f"
            )

        with col4:
            item_type = st.selectbox(
                "Item Type",
                sorted(df["Item_Type"].dropna().unique())
            )

        # Second row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            item_mrp = st.number_input(
                "Item MRP (₹)",
                min_value=0.0,
                value=150.0,
                step=1.0
            )

        with col2:
            outlet_identifier = st.selectbox(
                "Outlet Identifier",
                sorted(df["Outlet_Identifier"].dropna().unique())
            )

        with col3:
            outlet_size = st.selectbox(
                "Outlet Size",
                sorted(df["Outlet_Size"].dropna().unique())
            )

        with col4:
            outlet_location_type = st.selectbox(
                "Outlet Location Type",
                sorted(df["Outlet_Location_Type"].dropna().unique())
            )

        # Third row
        col1, col2 = st.columns(2)

        with col1:
            outlet_type = st.selectbox(
                "Outlet Type",
                sorted(df["Outlet_Type"].dropna().unique())
            )

        with col2:
            outlet_age = st.number_input(
                "Outlet Age (Years)",
                min_value=0,
                max_value=100,
                value=15
            )

        st.divider()

        predict_button = st.form_submit_button(
            "🚀 Predict Sales",
            type="primary",
            use_container_width=True
        )

    # Prediction
    if predict_button:

        input_data = pd.DataFrame({

            "Item_Weight": [item_weight],
            "Item_Fat_Content": [item_fat_content],
            "Item_Visibility": [item_visibility],
            "Item_Type": [item_type],
            "Item_MRP": [item_mrp],
            "Outlet_Identifier": [outlet_identifier],
            "Outlet_Size": [outlet_size],
            "Outlet_Location_Type": [outlet_location_type],
            "Outlet_Type": [outlet_type],
            "Outlet_Age": [outlet_age]

        })

        # Predict
        prediction = model.predict(input_data)

        predicted_sales = float(prediction[0])

        st.divider()

        st.success("Sales Prediction Completed!")

        st.subheader("📊 Predicted Item Outlet Sales")

        st.metric(
            label="Expected Sales",
            value=f"₹ {predicted_sales:,.2f}"
        )

        with st.expander("View Input Details"):

            st.dataframe(
                input_data,
                use_container_width=True
            )



# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "BigMart Sales Analysis & Prediction | "
    "Developed using Python, Pandas, Streamlit and CatBoost."
)
