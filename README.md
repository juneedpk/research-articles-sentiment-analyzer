# Research Publication Sentiment Analyzer

This Streamlit application analyzes the sentiment of research publications using AI. It fetches articles from Google Scholar, Crossref, and Semantic Scholar, performs sentiment analysis on the abstracts, and visualizes the results.

## Features

- Search academic publications using keywords, authors, or topics.
- Perform sentiment analysis on the abstracts of research papers.
- Visualize sentiment distribution and sentiment over time using interactive Plotly charts.
- Display publication details, including title, authors, year, sentiment label, and confidence score.

## Instructions

1.  Enter your search query in the text input field.
2.  Click the "Search" button to fetch publications.
3.  View the analysis results in the tabs:
    -   **Publications:** Displays a table of publication details.
    -   **Sentiment Analysis:** Shows the distribution of sentiment labels (positive, negative, neutral).
    -   **Sentiment Over Time:** Visualizes the average sentiment score over the years.

## Libraries Used

-   Streamlit: For creating the web application.
-   Pandas: For data manipulation and analysis.
-   Transformers: For sentiment analysis using the `distilbert-base-uncased-finetuned-sst-2-english` model.
-   Plotly: For interactive visualizations.
-   Scholarly: For fetching articles from Google Scholar.
-   Requests: For making HTTP requests to Crossref and Semantic Scholar APIs.

## Developer Info

-   Created by: Your Name
-   Version: 1.0.0

## GitHub

[Link to WEB APP](https://sentidoc.streamlit.app/)
