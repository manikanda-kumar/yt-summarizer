# YouTube Video Summarizer

A Streamlit-based web application that fetches a YouTube video's transcript and generates a detailed summary using OpenAI's GPT models.

## Overview

This application allows you to:
• Fetch video metadata and transcript (including automatic captions) using yt-dlp.
• Automatically parse chapters from the video description (if available) or split the transcript into time-based segments.
• Summarize each chapter or segment with OpenAI's GPT model (supports gpt-3.5-turbo, gpt-4, and gpt-4-turbo).
• Generate an overall combined summary along with detailed section summaries.
• View real-time progress updates while processing and download the final summary in markdown format.

## Features

- **Automatic Transcript Extraction:** Uses yt-dlp to download video information and English captions.
- **Chapter and Segment Processing:** Detects chapters from the video description using timestamps; falls back to time-based segmentation if chapters are unavailable.
- **Detailed Summarization:** Generates both section summaries and an overall summary with enhanced prompt instructions.
- **Exportable Markdown Document:** Saves the final summary as a markdown (.md) file and provides a download button in the interface.
- **Real-Time UI Updates:** Displays live progress and status messages during video processing.

## Requirements

The project depends on the following packages:
- openai==1.90.0
- streamlit==1.46.0
- tiktoken==0.9.0
- yt_dlp==2025.6.9

Install them via pip:

```bash
pip install -r requirements.txt


