
from streamlit_option_menu import option_menu
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Set the page configuration
st.set_page_config(page_title="SOCIAL MEDIA", page_icon="📊", layout="wide")

def calculate_sentiment(text):
    # Load the tokenizer, config, and model
    
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    config = AutoConfig.from_pretrained("best_model")
    model = AutoModelForSequenceClassification.from_pretrained("best_model")

    # Tokenize input text
    encoded_input = tokenizer(text, return_tensors='pt')

    # Get model output
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Find the highest scoring label
    max_score_idx = np.argmax(scores)
    label = config.id2label[max_score_idx]
    score = round(float(scores[max_score_idx]), 4)

    return label, score


#################################################################################################################
# Sidebar menu options
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Sentiment Analysis", "Recommendation System"],  # required
        icons=["emoji-smile", "list-task"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # default selected option
    )

# Sentiment Analysis Page
if selected == "Sentiment Analysis":
    st.title("Sentiment Analysis App")
    st.write("Enter a text below to analyze its sentiment:")

    # Text input from the user
    user_input = st.text_area("Your Text:", height=150)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    sentiment_label, sentiment_score = calculate_sentiment(user_input)

                    st.write("### Sentiment Analysis Result:")
                    st.write(f"**Sentiment:** {sentiment_label}")
                    st.write(f"**Confidence Score:** {sentiment_score}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to analyze.")
# File upload for batch processing
    # Sidebar for file upload
    st.sidebar.title("Batch Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'clean_text' column:", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)

            if 'clean_text' not in df.columns:
                st.error("The uploaded file must contain a 'clean_text' column.")
            else:
                with st.spinner("Analyzing sentiments for the dataset..."):
                    results = []
                    
                    # Convert 'clean_text' to string and handle NaN values
                    df['clean_text'] = df['clean_text'].fillna("").astype(str)

                    for text in df['clean_text']:
                        label, score = calculate_sentiment(text)
                        results.append({
                            "clean_text": text,
                            "predicted_sentiment": label,
                            "confidence_score": score
                        })

                    # Create a new DataFrame with results
                    result_df = pd.DataFrame(results)

                    st.write("### Sentiment Analysis Results for Uploaded Dataset:")
                    st.dataframe(result_df)

                    # Option to download the results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")



    


# Recommendation System Page
elif selected == "Recommendation System":
    


        # Recommendation System Function
        def recommend_related_posts(input_string, df, top_n=5, threshold=0.3):
            """
            Recommends related posts based on cosine similarity.
        
            Args:
                input_string: The input text provided by the user.
                df: DataFrame containing the 'clean_text' column.
                top_n: Number of top recommendations to return.
                threshold: Minimum similarity score to consider.
        
            Returns:
                DataFrame with the top recommended posts.
            """
            # Ensure the dataset has a 'clean_text' column
            if "clean_text" not in df.columns:
                st.error("The dataset must contain a 'clean_text' column.")
                return pd.DataFrame()
        
            # Create TF-IDF vectorizer and transform the text
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(df["clean_text"].astype(str))
        
            # Transform the input string into TF-IDF space
            input_vec = tfidf.transform([input_string])
        
            # Compute cosine similarity
            similarity_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
        
            # Apply threshold
            df["similarity"] = similarity_scores
            filtered_df = df[df["similarity"] >= threshold]
        
            # Get top N most similar posts
            top_recommendations = filtered_df.sort_values(by="similarity", ascending=False).head(top_n)
        
            return top_recommendations[["clean_text", "similarity"]]
        
        # Streamlit UI
        st.title("Recommendation System")
        st.subheader("Find related posts based on your input.")
        
        # Dataset Upload
        st.sidebar.header("Upload Dataset")
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_file:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
        
            # Display the dataset
            st.sidebar.write("### Dataset Preview")
            st.sidebar.dataframe(df.head())
        
            # Ensure the dataset has a 'clean_text' column
            if "clean_text" not in df.columns:
                st.error("The uploaded dataset must contain a 'clean_text' column.")
            else:
                # User input for recommendation
                user_input = st.text_input("Enter a sentence to find related posts:", "I love traveling and trying new foods.")
        
                # User-selected options
                top_n = st.slider("Number of recommendations to display:", min_value=1, max_value=10, value=5)
                threshold = st.slider("Similarity threshold:", min_value=0.0, max_value=1.0, value=0.3)
        
                # Generate recommendations
                if st.button("Get Recommendations"):
                    if user_input.strip():
                        recommendations = recommend_related_posts(user_input, df, top_n, threshold)
                        if not recommendations.empty:
                            st.write("### Top Recommendations:")
                            for idx, row in recommendations.iterrows():
                                st.write(f"- **{row['clean_text']}** (Similarity: {row['similarity']:.2f})")
                        else:
                            st.write("No recommendations found above the similarity threshold.")
                    else:
                        st.write("Please enter a valid input to get recommendations.")
        else:
            st.write("Please upload a dataset with a 'clean_text' column to get started.")


# Run this script with: streamlit run <script_name>.py
# Make sure to install the `streamlit-option-menu` package with `pip install streamlit-option-menu` before running the code.


