import streamlit as st

#EDA Pkgs
import pandas as pd
import numpy as np 

import altair as alt
## Utils

import joblib


pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_2021.pkl","rb"))

def predict_emotion(doc):
	result = pipe_lr.predict([doc])
	return result[0]

def get_prediction_proba(doc):
	result = pipe_lr.predict_proba([doc])
	return result

## Emojis
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
	st.title("Emotion Detection App")

	menu = ['Home',"Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home-Emotion From Text")

		with st.form(key = "emotion_clf_form"):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label="Submit")
		if submit_text:
			col1,col2 = st.beta_columns(2)

			prediction = predict_emotion(raw_text)
			probability = get_prediction_proba(raw_text)
			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))
			with col2:
				st.success("Prediction Probability")
				#st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				#st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ['emotions',"probability"]
				

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emotions",y='probability',color='emotions')
				st.altair_chart(fig,use_container_width = True)
	elif choice == "Monitor":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")	



	else:
		st.subheader("About")
		add_page_visited_details("About",datetime.now())

if __name__ == "__main__":
	main()









