import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import streamlit as st

# Sample dataset (You can replace this with a large dataset)
quote_data = {
    "happy": [
        "Happiness depends upon ourselves.",
        "The purpose of our lives is to be happy.",
        "Happiness is not something ready-made. It comes from your actions."
    ],
    "sad": [
        "Tears come from the heart and not from the brain.",
        "Sadness flies away on the wings of time.",
        "Every human walks around with a certain kind of sadness."
    ],
    "motivated": [
        "The only way to do great work is to love what you do.",
        "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "Do what you can, with what you have, where you are."
    ],
    "relaxed": [
        "Almost everything will work again if you unplug it for a few minutes.",
        "Take rest; a field that has rested gives a bountiful crop.",
        "Sometimes the most productive thing you can do is relax."
    ],
    "angry": [
        "For every minute you remain angry, you give up sixty seconds of peace of mind.",
        "Anger is one letter short of danger.",
        "Holding onto anger is like drinking poison and expecting the other person to die."
    ]
}

# Tokenization
all_quotes = [quote for quotes in quote_data.values() for quote in quotes]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_quotes)
total_words = len(tokenizer.word_index) + 1

# Prepare sequences
input_sequences = []
for mood, quotes in quote_data.items():
    for quote in quotes:
        token_list = tokenizer.texts_to_sequences([quote])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i+1])

# Pad sequences
max_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')
x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_length-1),
    tf.keras.layers.LSTM(150, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, verbose=1)

# Quote Generation Function
def generate_quote(mood):
    seed_text = random.choice(quote_data[mood])
    next_words = 10
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Streamlit App for UI
st.title("Mood-Based Quote Generator")
selected_mood = st.selectbox("Select a Mood", ["happy", "sad", "motivated", "relaxed", "angry"])
if st.button("Generate Quote"):
    quote = generate_quote(selected_mood)
    st.write(f"**Generated Quote:** {quote}")
