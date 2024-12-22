import os
import google.generativeai as genai
from PIL import Image
from gensim.models import KeyedVectors
import streamlit as st
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from essayGradeHelper import essay_to_wordlist,get_model,getAvgFeatureVecs
import json
import pandas as pd
import numpy as np


gemini_key = 'AIzaSyCGJA-usGDbFY0n_BvhF3pyGQeANSzleDY' 
genai.configure(api_key=gemini_key)
model_name = "gemini-1.5-flash"
ocr = genai.GenerativeModel(model_name)


# Check if the user has provided essay content
essay_set_data = {
    "essay_set": [1, 2, 3, 4, 5, 6, 7, 8],
    "type_of_essay": ["persuasive / narrative / expository", "persuasive / narrative / expository", "source dependent responses", "source dependent responses", "source dependent responses", "source dependent responses", "persuasive / narrative / expository", "persuasive / narrative / expository"],
    "grade_level": [8, 10, 10, 10, 8, 10, 7, 10],
    "min_domain1_score": [2, 1, 0, 0, 0, 0, 0, 0],
    "max_domain1_score": [12, 6, 3, 3, 4, 4, 30, 60]
}

def grade_essay(essay_content):
  if len(essay_content)>40:
     word2vec_path = 'word2vecmodel (1).bin'
     lstm_model_path = 'final_lstm (1).h5'
     num_features = 300
     word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
     clean_test_essays = [essay_to_wordlist(essay_content, remove_stopwords=True)]
     testDataVecs = getAvgFeatureVecs(clean_test_essays, word2vec_model, num_features)
     testDataVecs = np.array(testDataVecs).reshape((len(testDataVecs), 1, num_features))
     lstm_model = get_model()
     lstm_model.load_weights(lstm_model_path)
     preds = lstm_model.predict(testDataVecs)
     st.write("Predicted grade:", round(preds[0][0]))


st.title("Automated Grading Application")
st.write("Welcome to the Automated Grading Application. You can submit an essay or your exam answer either as text or as a handwritten image.")
image_path = "Grading-The-Good-the-Bad-and-the-Ugly.webp"  # Change to your image file path
image = Image.open(image_path)  # Open the image with PIL
st.image(image,use_column_width=True)  # Display the image with a caption


grading_option = st.selectbox("What would you like to do?", ["Grade an Essay"])

if grading_option == "Grade an Essay":
 essay_set_df = pd.DataFrame(essay_set_data)
 st.dataframe(essay_set_df)

 questions = ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5", "Question 6", "Question 7", "Question 8"]

 selected_question = st.selectbox("Select a question to answer", questions)

 question_number = int(selected_question.split(" ")[1])

 selected_essay_set = essay_set_df[essay_set_df["essay_set"] == question_number]
 min_score = selected_essay_set["min_domain1_score"].values[0]
 max_score = selected_essay_set["max_domain1_score"].values[0]

 st.write(f"The minimum score for {selected_question} is: {min_score}")
 st.write(f"The maximum score for {selected_question} is: {max_score}")

 input_type = st.radio("Choose your input method:", ("Text", "Handwritten Image"))
 if input_type == "Text":
   essay_content = st.text_area(f"Answer the following question: {selected_question}", height=150)
   grade_essay(essay_content)
 elif input_type == "Handwritten Image":
    uploaded_file = st.file_uploader("Upload a handwritten essay image:", type=["png", "jpg", "jpeg"])

    with st.spinner("Extracting text from the image..."):
     if uploaded_file is not None : 
        img = Image.open(uploaded_file)
        if ocr:
            response = ocr.generate_content(["Extract text", img])
            essay_content = response.text
            grade_essay(essay_content)
            st.markdown(f"**Extracted Text:**\n\n{essay_content}")
        else:
            st.error("No suitable model found for text extraction.")



elif grading_option == "Grade based on Synoptic":
    def load_question_synoptics():
      if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            return json.load(f)
      else:
        return {}

    json_file_path = "question_synoptics.json"
    question_synoptics = load_question_synoptics()
    def save_question_synoptics(data):
     with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    model = SentenceTransformer('bert-base-nli-mean-tokens')

  

    question_text = st.text_input("Enter the Question")
    synoptic_text = st.text_area("Enter the Synoptic for this Question")

    if st.button("Add Question"):
        if question_text and synoptic_text:
            question_synoptics[question_text] = synoptic_text
            save_question_synoptics(question_synoptics)
            st.success("Question added successfully!")
        else:
            st.warning("Please enter both a question and a synoptic.")

    if question_synoptics:
        selected_question = st.selectbox("Select a question", list(question_synoptics.keys()))
        selected_synoptic = question_synoptics[selected_question]

        # Enter the student's answer
        input_method = st.radio("Select the input method:", ("Text", "Handwritten Image"))
        if input_method == "Text":
            student_answer = st.text_area("Enter the student's answer:")
        elif input_method == "Handwritten Image":
            uploaded_file = st.file_uploader("Upload a handwritten essay image:", type=["png", "jpg", "jpeg"])

            with st.spinner("Extracting text from the image..."):
             if uploaded_file is not None : 
              img = Image.open(uploaded_file)
              if ocr:
                response = ocr.generate_content(["Extract text", img])
                student_answer = response.text
                st.markdown(f"**Extracted Text:**\n\n{student_answer}")
              else:
               st.error("No suitable model found for text extraction.")
               
        if st.button("Calculate Similarity"):
                synoptic_vec = model.encode([selected_synoptic])[0]
                student_vec = model.encode([student_answer])[0]
                print(synoptic_vec.shape)
                print(student_vec.shape)

                similarity_score = 1 - distance.cosine(synoptic_vec, student_vec)
                len_norm_factor = min(len(selected_synoptic), len(student_answer)) / max(len(selected_synoptic), len(student_answer))
                similarity_score *= len_norm_factor
                st.subheader("Similarity Score")
                st.write(f"The similarity score between the synoptic and the student's answer is: {similarity_score:.2f}")

                if similarity_score > 0.8:
                    st.success("High similarity. The answer closely matches the synoptic.")
                elif similarity_score > 0.5:
                    st.info("Moderate similarity. Some common points.")
                else:
                    st.warning("Low similarity. The answer is quite different from the synoptic.")




