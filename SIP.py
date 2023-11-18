import streamlit as st
import sqlite3
import tensorflow as tf
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Connect to the MySQL database
# db_connection = sqlite3.connect(
#    'storage.db'
# )
# db_cursor = db_connection.cursor()

# # Create a table for storing user input and model predictions
# create_table_query = """
# CREATE TABLE IF NOT EXISTS user_input_predictions (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     user_input TEXT,
#     model_prediction FLOAT
# )
# """
# db_cursor.execute(create_table_query)

# Load your PyTorch model
def load_model():
    model_path ="C:\\Users\\dsuya\\Desktop\\ \\college\\linkedin_job_allocator_bart.pt"
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Adjust map_location based on your needs
    model.eval()
    return model

st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #008080;
            color: white;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            background-color: #f0f0f0;
            color: #333;
        }
        .stText {
            color: #333;
        }
        .stMarkdown {
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model1 = load_model()
# model1 ="C:\\Users\\dsuya\\Desktop\\ \\college\\linkedin_job_allocator_bart.pt"
torch.save(model1.state_dict(), "linkedin_job_allocator_bart.pth")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to("cpu")

model.load_state_dict(torch.load("linkedin_job_allocator_bart.pth", map_location=torch.device('cpu')))
st.title('Job Allocator')

# Input for user to enter text
user_input = st.text_area('Enter text for prediction:', '')
# Pre-Process:
# Load the tokenizer associated with your model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Tokenize the text
# tokens = tokenizer(user_input, return_tensors="pt")
# # st.write(tokens)

# # The 'input_ids' key contains the tokenized and encoded version of the text
# final_input = tokens["input_ids"]
tokens = tokenizer(user_input,
                       truncation=True,
                       max_length=256,
                       padding=True,
                       return_tensors="pt").to("cpu")

# st.write(final_input)
# seq2seq_pipeline = pipeline(task="text2text-generation", model="facebook/bart-large-cnn")
if st.button('Predict'):
    # st.write(final_input)
    if user_input:
        
        # Perform prediction with the model
        # with torch.no_grad():
#          output = model(final_input)
#         #  st.write(output)
#         #  generated_ids = output["sequences"].cpu().numpy().tolist()
#          logits=output.logits
#         #  st.write(logits[0].tolist())
#          probs = torch.nn.functional.softmax(logits, dim=-1)

# # Sample or decode the output sequence
# # For simplicity, we'll use greedy decoding here
#         decoded_output = torch.argmax(probs, dim=-1)

# # Convert the decoded sequence back into text using the tokenizer
#         final_output = tokenizer.decode(decoded_output[0], skip_special_tokens=True)
        # id=torch.argmax(logits, dim=-1)[0].tolist()
        # # flat_logits = logits[0].argmax(dim=-1).tolist()
        # #  st.write(flat_logits)
        # final_output=tokenizer.decode(output[0], skip_special_tokens=True)
        # #  st.write(logits[0].tolist())

        # Display the model's prediction
        output = model.generate(**tokens, max_length=20)
        final_output=tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        st.write('Model Prediction:', final_output)

#         # Save user input and model prediction to the database
#         insert_query = "INSERT INTO user_input_predictions (user_input, model_prediction) VALUES (%s, %s)"
#         values = (user_input, float(output))
#         db_cursor.execute(insert_query, values)
#         db_connection.commit()

# # Close the database connection when the app is done
# db_cursor.close()
# db_connection.close()
