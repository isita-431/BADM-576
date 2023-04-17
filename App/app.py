import streamlit as st
import mlflow.sklearn
import pandas as pd

# Load the saved model
model_uri = "C:\\Users\\ashishk\\Box\\UIUC Teaching\\Spring_23\\BADM 576\\ML Ops\\mlruns\\0\\b441e19f8a6c480c899f4429dc6e63b5\\artifacts\\model"
model = mlflow.sklearn.load_model(model_uri)

# Define a function to make predictions
def predict(data):
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)[0]
    return prediction

# Define the Streamlit app
def app():
    st.set_page_config(page_title="Titanic Survival Prediction App")
    st.title("Titanic Survival Prediction App")
    st.write("Please enter the following information to get a prediction:")

    # Define the input form
    input_form = {
        "Pclass": st.selectbox("Passenger Class", [1, 2, 3]),
        "Name": st.text_input("Name"),
        "Sex": st.selectbox("Sex", ["male", "female"]),
        "Age": st.number_input("Age", min_value=0.0, max_value=100.0, step=0.1),
        "Siblings/Spouses Aboard": st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, step=1),
        "Parents/Children Aboard": st.number_input("Parents/Children Aboard", min_value=0, max_value=10, step=1),
        "Fare": st.number_input("Fare", min_value=0.0, max_value=1000.0, step=0.01)
    }

    # Make a prediction when the form is submitted
    if st.button("Submit"):
        prediction = predict(input_form)
        if prediction == 0:
            st.error("The passenger did not survive")
        else:
            st.success("The passenger survived")

# Run the Streamlit app
if __name__ == "__main__":
    app()
