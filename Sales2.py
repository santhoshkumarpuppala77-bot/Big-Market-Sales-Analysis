import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open('sales.pkl', 'rb'))
df = pd.read_csv('Train.csv')

st.title("Big Market Sales analysis")
st.image('image.png',width=700)
st.markdown("Predict the Sales of each Product.")


if "show_form" not in st.session_state:
    st.session_state.show_form = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None


if st.button('Click here to predict the Sales '):
    st.session_state.show_form = True


if st.session_state.show_form:
    st.header("📦 Product & Outlet Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        item_weight = st.number_input("Item Weight",min_value=0.0,value=12.0,step=0.1)

    with col2:
        item_fat_content = st.selectbox("Item Fat Content",df['Item_Fat_Content'].dropna().unique())

    with col3:
        item_visibility = st.number_input("Item Visibility",min_value=0.0,value=0.05,step=0.001,format="%.4f")

    with col4:
        item_type = st.selectbox("Item Type",df['Item_Type'].dropna().unique())


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        item_mrp = st.number_input("Item MRP",min_value=0.0,value=150.0)

    with col2:
        outlet_identifier = st.selectbox("Outlet Identifier",df['Outlet_Identifier'].dropna().unique())

    with col3:
        outlet_size = st.selectbox("Outlet Size",df['Outlet_Size'].dropna().unique())

    with col4:
        outlet_location_type = st.selectbox("Outlet Location Type",df['Outlet_Location_Type'].dropna().unique())


    col1, col2 = st.columns(2)

    with col1:
        outlet_type = st.selectbox("Outlet Type",df['Outlet_Type'].dropna().unique())

    with col2:
        outlet_age = st.number_input("Outlet Age",min_value=0,max_value=100,value=15)

    predict_button = st.button("Predict Item Outlet Sales", use_container_width=True)

    if predict_button:

        input_data = pd.DataFrame({
            'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content],
            'Item_Visibility': [item_visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Identifier': [outlet_identifier],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location_type],
            'Outlet_Type': [outlet_type],
            'Outlet_Age': [outlet_age]
        })

        prediction = model.predict(input_data)

        prediction_value = float(prediction[0])

        st.markdown("### Sales Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Sales",
                f"₹ {prediction_value:,.2f}"
            )

        with col2:
            st.metric(
                "Item MRP",
                f"₹ {float(item_mrp):,.2f}"
            )

        with col3:
            st.metric(
                "Item Visibility",
                f"{float(item_visibility):.4f}"
            )

        if prediction_value < 1000:
            category = "Low Sales"

        elif prediction_value < 3000:
            category = "Moderate Sales"

        elif prediction_value < 5000:
            category = "High Sales"

        else:
            category = "Very High Sales"

        st.success(category)

        st.markdown("### Predicted Sales Overview")

        progress = float(min(max(prediction_value / 10000, 0.0), 1.0))

        st.progress(progress)

        st.caption(
            f"Estimated Item Outlet Sales: ₹ {prediction_value:,.2f}"
        )