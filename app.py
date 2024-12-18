import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import datetime
import logging
import requests
from scholarly import scholarly
import plotly.express as px

# Add after imports
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configure Streamlit page
st.set_page_config(page_title="Research Publication Sentiment Analyzer", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .header-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .search-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .results-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def fetch_articles_from_crossref(query):
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": 20,  # Fetch more to account for filtering
        "filter": "type:journal-article,from-pub-date:2021-01-01,until-pub-date:2024-10-31",  # Only journal articles from 2021 to 2023
        "select": "title,abstract,published-print,author,type"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = []
            
            for item in data['message']['items']:
                # Check if it's a research/review article and has abstract
                if (item.get('type') == 'journal-article' and 
                    'abstract' in item and 
                    len(item.get('abstract', '')) > 100):  # Ensure meaningful abstract
                    
                    title = item.get('title', ['Untitled'])[0]
                    abstract = item.get('abstract')
                    year = item.get('published-print', {}).get('date-parts', [[datetime.datetime.now().year]])[0][0]
                    authors = ', '.join([
                        f"{author.get('given', '')} {author.get('family', '')}"
                        for author in item.get('author', [])
                    ])[:200]
                    
                    articles.append({
                        'title': title,
                        'abstract': abstract,
                        'year': str(year),
                        'authors': authors,
                        'type': 'Research Article'
                    })
            
            return articles[:10]  # Return top 10 filtered articles
        else:
            st.error("Error fetching articles from CrossRef")
            return []
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"CrossRef API error: {e}")
        return []

def fetch_articles_from_semantic_scholar(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {
        "x-api-key": "YOUR_API_KEY"  # Replace with your Semantic Scholar API key
    }
    params = {
        "query": query,
        "fields": "title,abstract,year,authors",
        "limit": 100  # Fetch up to 100 articles
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        articles = []
        
        for item in data['data']:
            # Check if it has an abstract
            if 'abstract' in item and len(item['abstract']) > 100:  # Ensure meaningful abstract
                title = item.get('title', 'Untitled')
                abstract = item.get('abstract')
                year = item.get('year', datetime.datetime.now().year)
                authors = ', '.join([author['name'] for author in item.get('authors', [])])
                
                articles.append({
                    'title': title,
                    'abstract': abstract,
                    'year': str(year),
                    'authors': authors,
                    'type': 'Research Article'
                })
        
        return articles
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching articles from Semantic Scholar: {e}")
        logging.error(f"Semantic Scholar API error: {e}")
        return []

def fetch_articles_from_google_scholar(query):
    try:
        search_query = scholarly.search_pubs(query)
        articles = []
        
        for i in range(100):  # Fetch up to 100 articles
            try:
                item = next(search_query)
                if 'abstract' in item['bib'] and len(item['bib']['abstract']) > 100:  # Ensure meaningful abstract
                    title = item['bib'].get('title', 'Untitled')
                    abstract = item['bib'].get('abstract')
                    year = item['bib'].get('pub_year', datetime.datetime.now().year)
                    authors = ', '.join(item['bib'].get('author', []))
                    
                    articles.append({
                        'title': title,
                        'abstract': abstract,
                        'year': str(year),
                        'authors': authors,
                        'type': 'Research Article'
                    })
            except StopIteration:
                break
        
        return articles
    except Exception as e:
        st.error(f"Error fetching articles from Google Scholar: {e}")
        logging.error(f"Google Scholar API error: {e}")
        return []

def analyze_sentiment(text):
    """Process text and return numeric sentiment score"""
    try:
        if not text or len(text) < 10:
            return {'label': 'NEUTRAL', 'score': 0.5}
            
        result = sentiment_classifier(text[:512])[0]
        
        # Convert sentiment to numeric
        sentiment_score = 1.0 if result['label'] == 'POSITIVE' else 0.0
        
        return {
            'label': result['label'],
            'score': sentiment_score,
            'confidence': float(result['score'])
        }
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.5}

def process_articles(articles):
    df = pd.DataFrame(articles)
    
    # Apply sentiment analysis
    sentiments = df['abstract'].apply(analyze_sentiment)
    df['sentiment_label'] = sentiments.apply(lambda x: x['label'])
    df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
    df['confidence'] = sentiments.apply(lambda x: x['confidence'])
    
    # Handle non-numeric year values
    current_year = datetime.datetime.now().year
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(current_year).astype(int)
    
    return df

def plot_sentiment_over_time(df):
    df['year'] = df['year'].astype(int)
    sentiment_over_time = df.groupby('year')['sentiment_score'].mean().reset_index()
    
    fig = px.line(sentiment_over_time, x='year', y='sentiment_score', title='Sentiment Analysis Over Time', markers=True)
    st.plotly_chart(fig)

def main():
    # Add sidebar
    add_sidebar()
    
    # Header
    with st.container():
        col1, col2 = st.columns([1, 5])
        #with col1:
            #st.image("https://img.icons8.com/color/96/000000/research.png", width=80)
        with col2:
            st.title("Research Publication Sentiment Analyzer")
            st.markdown("*Analyze the sentiment of research papers with AI*")
    
    # Search Section
    with st.container():
        st.markdown("### 🔍 Search Publications")
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter keywords, authors, or topics", placeholder="e.g., machine learning, climate change")
        with col2:
            st.write("")
            st.write("")
            search_button = st.button("Search", use_container_width=True)
    
    if query and search_button:
        with st.spinner('Fetching publications...'):
            articles = fetch_articles_from_google_scholar(query)
            
        if articles:
            # Process articles
            articles_data = process_articles(articles)
            
            # Results Section
            st.markdown("### 📊 Analysis Results")
            
            # Statistics Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Publications", len(articles_data))
            with col2:
                avg_sentiment = articles_data['sentiment_score'].mean()
                st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            with col3:
                recent_year = max(articles_data['year'].astype(int))
                st.metric("Most Recent", recent_year)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📑 Publications", "📈 Sentiment Analysis", "📅 Sentiment Over Time"])
            
            with tab1:
                st.dataframe(
                    articles_data[['title', 'authors', 'year', 'sentiment_label', 'confidence']],
                    use_container_width=True,
                    height=400
                )
            
            with tab2:
                sentiment_counts = articles_data['sentiment_label'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment_label', 'count']
                fig = px.bar(sentiment_counts, x='sentiment_label', y='count', title='Sentiment Distribution')
                st.plotly_chart(fig)
            
            with tab3:
                plot_sentiment_over_time(articles_data)
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Google Scholar and Hugging Face*")

def add_sidebar():
    with st.sidebar:
        st.header("📚 About the App")
        
        st.subheader("🎯 Purpose")
        st.write("Analyze sentiment patterns in research publications using AI.")
        
        st.subheader("✨ Features")
        st.markdown("""
        - Search academic publications
        - Sentiment analysis of abstracts
        - Visual analytics
        - Export results
        """)
        
        st.subheader("📝 Instructions")
        st.markdown("""
        1. Enter your search query
        2. Wait for results to load
        3. View sentiment analysis
        4. Export if needed
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer Info")
        st.markdown("Created by: Your Name")
        st.markdown("Version: 1.0.0")
        
        # GitHub link
        st.markdown("""
        <a href="https://github.com/juneedpk" target="_blank">
            <img src="https://img.shields.io/github/stars/juneedpk" alt="GitHub">
        </a>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
