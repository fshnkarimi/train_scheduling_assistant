import streamlit as st
import sys
import os


# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

# Import nlp_processing module and LLM loading function
from nlp.nlp_processing import extract_schedule_info
from llm.llm_model import load_model, generate_schedule

# Load the fine-tuned LLM
model = load_model('../models/llm/fine_tuned_llm_model.pth')

# Streamlit UI
st.title('Train Scheduling Assistant')

# Textbox for user to enter a scheduling scenario
user_input = st.text_area("Enter scheduling scenario:")

if st.button('Generate Schedule'):
    if user_input:
        # Extract scheduling information from the input
        schedule_info = extract_schedule_info(user_input)
        # print(schedule_info)
        # Create a string from the extracted information to feed into the LLM
        # This should be adjusted based on how your LLM expects input
        input_for_llm = f"Train {schedule_info['TRAIN_ID']} at {schedule_info['STATION']} with a delay of {schedule_info['DELAY']}."
        
        # Generate schedule using the fine-tuned LLM
        schedule_result = generate_schedule(input_for_llm, model)
        
        # Display extracted info and generated schedule
        st.subheader("Extracted Scheduling Information:")
        st.write(schedule_info)
        st.subheader("Generated Schedule:")
        st.write(schedule_result)
    else:
        st.warning("Please enter a scheduling scenario.")
