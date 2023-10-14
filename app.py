import pickle
import streamlit as st
import pyttsx3
import speech_recognition as sr

st.title(":rainbow[SPRINT] :fire:")
cv = pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.divider()

with st.container():
  st.title("Language Detector")

  input = st.text_input("Enter a text")

  if st.button('Detect'):
    text = cv.transform([input]).toarray()
    result = model.predict(text)[0]
    st.header(result)
st.divider()
# SPEECH TO TEXT CONVERTOR

# code needed
r = sr.Recognizer()
def talk(text):
  engine = pyttsx3.init()
  engine.say(text)
  engine.runAndWait()

# GUI portion

st.title("Speech - Text Convertor")

with st.container():
  if st.button("Speak"):
    with sr.Microphone() as source:
      # clear background noise
      r.adjust_for_ambient_noise(source, duration=0.3)

      st.subheader("<--- NOW SPEAK --->")
      talk("now speak")

      # capture the audio
      audio = r.listen(source)

      try:
         # converting the audio into text
         text = r.recognize_google(audio)
         st.subheader(text)
         talk(text)
      except sr.WaitTimeoutError:
         st.warning("Audio input timed out. Please speak again.")
         st.stop()
      except sr.RequestError as e:
         st.warning(f"Could not request results from Google's servers; {e}")
         st.stop()
      except sr.UnknownValueError:
         st.warning("Google Speech Recognition could not understand audio. Please speak again.")
         st.stop()
      except:
         st.toast("Please say again!!")
         st.stop()
st.divider()