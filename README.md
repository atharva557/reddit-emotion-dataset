# 🧠 Reddit 28-Emotion Dataset: Deep Psychological NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Size](https://img.shields.io/badge/Sample_Size-500_Rows-blue)]()
[![Update Frequency](https://img.shields.io/badge/Status-Active_Scraping-success)]()

## 📌 Overview
Standard sentiment analysis is broken. Categorizing complex human text into generic "Positive, Negative, or Neutral" buckets strips away the actual psychological drivers of a community. 

This repository contains a sample dataset of Reddit posts and comments, processed through a custom multi-label NLP model trained to detect **28 distinct emotional states** (e.g., *optimism, annoyance, curiosity, grief, amusement, panic*). 

By analyzing the precise probabilities of these 28 emotions, we can map the actual psychological fingerprint of a subreddit, track event-driven emotional shifts, and predict community behavior much more accurately than legacy sentiment APIs.

## 📊 What's Inside?
This repo contains a 500-row sample (`wallstreetbets_500_emotion_sample.csv`) designed for data scientists, quantitative analysts, and NLP researchers.

**Key Features:**
* **Granularity:** 28 discrete emotion probability scores per post.
* **Macro & Micro:** Includes both top-level post titles and micro-level comment firehose data.
* **Chronological:** Clean UTC to datetime conversion for accurate time-series backtesting.
* **Deduplicated:** 100% unique text strings.

## 🗂 Data Dictionary
| Column Name | Description | Example |
| :--- | :--- | :--- |
| `post_id` | Unique Reddit identifier | `1rgbpm8` |
| `date` | Timestamp of creation (YYYY-MM-DD HH:MM:SS) | `2026-03-10 14:30:00` |
| `clean_text` | The scraped text (title + body) with newlines removed | *"Man, times really are tough out there."* |
| `dominant_sentiment`| Legacy categorization | `negative` |
| `dominant_emotion` | The highest-scoring distinct emotion | `sadness` |
| `model_confidence` | The probability score of the dominant emotion | `0.7931` |
| `prob_[emotion]` | 28 distinct columns for every emotion class | `prob_amusement: 0.012, prob_anger: 0.336` |

## 🚀 Quick Start (Pandas)
Want to see what drives this community? Here is how to quickly load the data and plot the top emotions.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('wallstreetbets_500_emotion_sample.csv')

# Show the top 5 driving emotions (excluding neutral)
top_emotions = df['dominant_emotion'].value_counts()
print(top_emotions[top_emotions.index != 'neutral'].head(5))

# Check the exact emotional breakdown of a specific post
print(df.iloc[0][['clean_text', 'dominant_emotion', 'prob_sadness', 'prob_optimism']])
