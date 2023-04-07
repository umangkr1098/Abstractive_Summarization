import streamlit as st
import tensorflow
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('t5-small')
import numpy as np

@st.cache(allow_output_mutation=True)
#@st.cache_resource
def get_t5():
    model=TFT5ForConditionalGeneration.from_pretrained('t5-small')
    return model

def find_summary(context):
    m=get_t5()
    context="summarize: " + context

    input_ids = tokenizer.encode(context, return_tensors='tf')
    beam_output = m.generate(input_ids,max_length = 60,num_beams=5,temperature=0.6)
    output=tokenizer.decode(beam_output[0],skip_special_tokens=True)
    return output

st.title("Text Summarization")
st.subheader("Example Text:")
st.markdown("Kunal Shah's credit card bill payment platform, CRED, gave users a chance to win free food from Swiggy for one year. Pranav Kaushik, a Delhi techie, bagged this reward after spending 2000 CRED coins. Users get one CRED coin per rupee of bill paid, which can be used to avail rewards from brands like Ixigo, BookMyShow, UberEats, Cult.Fit and more.")
st.subheader("Summary:")
st.markdown(" Kunal Shah's credit card bill payment platform, CRED, gave users a chance to win free food from Swiggy for one year. users get one CRED coin per rupee of bill paid")

st.markdown("\n")
form = st.form(key="form")
context = form.text_area("Enter the text here")

predict_button = form.form_submit_button(label='Summarize')


if predict_button:
    with st.spinner('Summarizing Text'):
        answer = find_summary(context)
        st.write("Summary:",answer)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: black;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: grey;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/umang-kumar-b03372227" target="_blank">Umang Kumar</a></p>
<p>Email ID : <a style='display: block; text-align: center;' target="_blank">umangkumar1098@gmail.com</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
