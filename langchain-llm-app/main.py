import langchain_helper as lch
import streamlit as st

st.title="Pet generator"

animal_type=st.sidebar.selectbox("What is your pet?",("Dog","Cat","Cow","Hen"))
print(animal_type)

color=st.sidebar.text_area(label=f"what color is your {animal_type}?",
    max_chars=20)

if color:
    response=lch.generate_pet_name(animal_type,color)
    st.text(response['pet_name'])