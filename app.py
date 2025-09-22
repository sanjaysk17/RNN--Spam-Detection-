import gradio as gr
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
model=load_model("spam_rnn_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer=pickle.load(f)
with open("max_len.pkl", "rb") as f:
    MAX_LEN=pickle.load(f)
def predict_spam(text):
    seq=tokenizer.texts_to_sequences([text])
    seq=pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred=model.predict(seq)[0][0]
    label = "🚨 Spam" if pred > 0.5 else "✅ Ham"
    return f"{label} (probability: {pred:.2f})"
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=4, placeholder="Enter a message..."),
    outputs="text",
    title="📩 Spam vs Ham Detector (RNN)",
    description="Predicts Spam/Ham using trained RNN model + tokenizer",
    allow_flagging="never"
)
if __name__ == "__main__":
    demo.launch()
