#!/usr/bin/env python
"""
Streamlit YouTube Video Summarizer App
────
• Left panel: YouTube URL input and controls
• Right panel: Live markdown display
• Markdown files still saved to disk
• Real-time processing status updates

Dependencies
────
pip install streamlit openai tiktoken yt-dlp
"""
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import time

import streamlit as st
import yt_dlp
from openai import OpenAI
import tiktoken

# ────
# Configuration and Setup
# ────
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'summary_content' not in st.session_state:
    st.session_state.summary_content = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_url' not in st.session_state:
    st.session_state.last_url = ""

# ────
# Helper Functions
# ────
YOUTUBE_WATCH_RE = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

def fetch_info_and_transcript(url: str) -> Tuple[str, str, List[Dict]]:
    """
    Fetch video info and transcript using yt-dlp only.
    Returns (title, description, transcript_list)
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': False,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'subtitlesformat': 'json3',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Extract basic info
            title = info.get('title', f"YouTube Video {extract_video_id(url)}")
            description = info.get('description', '')
            
            # Extract transcript from automatic_captions or subtitles
            transcript = []
            
            # Try automatic captions first (usually more available)
            auto_captions = info.get('automatic_captions', {})
            subtitles = info.get('subtitles', {})
            
            # Look for English captions
            caption_data = None
            for lang in ['en', 'en-US', 'en-GB', 'en-us', 'en-gb']:
                if lang in auto_captions:
                    # Find json3 format
                    for fmt in auto_captions[lang]:
                        if fmt.get('ext') == 'json3':
                            caption_data = fmt
                            break
                    if caption_data:
                        break
                        
                if lang in subtitles:
                    for fmt in subtitles[lang]:
                        if fmt.get('ext') == 'json3':
                            caption_data = fmt
                            break
                    if caption_data:
                        break
            
            if caption_data and 'url' in caption_data:
                # Download and parse the caption data
                import urllib.request
                import json
                
                try:
                    with urllib.request.urlopen(caption_data['url']) as response:
                        caption_json = json.loads(response.read().decode('utf-8'))
                        
                    # Parse the JSON3 format
                    if 'events' in caption_json:
                        for event in caption_json['events']:
                            if 'segs' in event and event.get('tStartMs') is not None:
                                start_time = event['tStartMs'] / 1000.0  # Convert to seconds
                                duration = event.get('dDurationMs', 0) / 1000.0
                                
                                # Combine all segments in this event
                                text_parts = []
                                for seg in event['segs']:
                                    if 'utf8' in seg:
                                        text_parts.append(seg['utf8'])
                                
                                if text_parts:
                                    text = ''.join(text_parts).strip()
                                    if text and text != '\n':
                                        transcript.append({
                                            'start': start_time,
                                            'duration': duration,
                                            'text': text
                                        })
                except Exception as e:
                    st.warning(f"Could not parse transcript data: {e}")
            
            if not transcript:
                raise RuntimeError("No English transcript/captions available for this video")
                
            return title, description, transcript
            
    except Exception as exc:
        if "transcript" in str(exc).lower() or "caption" in str(exc).lower():
            raise RuntimeError(f"Transcript not available: {exc}")
        else:
            st.warning(f"Could not fetch video info - {exc}")
            raise exc

def extract_video_id(url: str) -> str:
    """Return the 11-character video id from any YouTube URL."""
    m = re.search(YOUTUBE_WATCH_RE, url)
    if not m:
        raise ValueError(f"Cannot parse video id from URL: {url}")
    return m.group(1)

def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are invalid in filenames"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    filename = re.sub(r'_+', '_', filename).strip('_')
    if len(filename) > 200:
        filename = filename[:200]
    return filename

TIMESTAMP_RE = re.compile(r"(\d{1,2}:)?\d{1,2}:\d{2}")

def hms_to_seconds(hms: str) -> int:
    parts = [int(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    else:  # mm:ss
        h, m, s = 0, *parts
    return h * 3600 + m * 60 + s

def parse_chapters(description: str) -> List[Tuple[str, int]]:
    """
    Return list of (title, start_seconds) from description lines that start with timestamps
    """
    chapters: List[Tuple[str, int]] = []
    for line in description.splitlines():
        if TIMESTAMP_RE.match(line.strip()):
            try:
                ts, title = line.split(maxsplit=1)
                chapters.append((title.strip(), hms_to_seconds(ts)))
            except ValueError:
                continue  # Skip malformed timestamp lines
    if chapters and chapters[0][1] != 0:
        chapters.insert(0, ("Intro", 0))
    return chapters

def split_transcript_by_time(transcript: List[Dict], duration_minutes: int = 10) -> Dict[str, str]:
    """Split transcript into time-based segments for videos without chapters."""
    if not transcript:
        return {"Full Video": ""}
    
    duration_seconds = duration_minutes * 60
    segments = {}
    current_segment = 1
    current_text = ""
    segment_start_time = 0
    
    for item in transcript:
        current_time = item["start"]
        text = item["text"].replace("\n", " ")
        
        if current_time - segment_start_time >= duration_seconds and current_text.strip():
            start_min = int(segment_start_time // 60)
            end_min = int(current_time // 60)
            segment_title = f"Segment {current_segment} ({start_min:02d}:{start_min%60:02d} - {end_min:02d}:{end_min%60:02d})"
            segments[segment_title] = current_text.strip()
            
            current_segment += 1
            segment_start_time = current_time
            current_text = text
        else:
            current_text += " " + text
    
    if current_text.strip():
        start_min = int(segment_start_time // 60)
        final_time = transcript[-1]["start"] + transcript[-1].get("duration", 0)
        end_min = int(final_time // 60)
        segment_title = f"Segment {current_segment} ({start_min:02d}:{start_min%60:02d} - {end_min:02d}:{end_min%60:02d})"
        segments[segment_title] = current_text.strip()
    
    return segments

def split_transcript(transcript: List[Dict], chapters: List[Tuple[str, int]]) -> Dict[str, str]:
    """Return {chapter_title: text}."""
    if not chapters:
        return split_transcript_by_time(transcript)

    ch_with_end = [
        (*chapters[i], chapters[i + 1][1]) if i + 1 < len(chapters) else (*chapters[i], math.inf)
        for i in range(len(chapters))
    ]
    buckets: Dict[str, str] = {title: "" for title, _, _ in ch_with_end}
    for item in transcript:
        t = item["start"]
        txt = item["text"].replace("\n", " ")
        for title, start, end in ch_with_end:
            if start <= t < end:
                buckets[title] += " " + txt
                break
    return buckets

# ────
# OpenAI Functions
# ────
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with caching"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()
_ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")

def _num_tokens(txt: str) -> int:
    return len(_ENCODER.encode(txt))

def _summarise_chunk(model: str, chunk: str, is_segment: bool = False) -> str:
    """Summarize a chunk of text with enhanced prompts for better quality"""
    if is_segment:
        prompt = (
            f"Provide a comprehensive and detailed summary of this video transcript segment. "
            f"Focus on:\n"
            f"• Main topics and key concepts discussed\n"
            f"• Important facts, data, or examples mentioned\n"
            f"• Key insights, conclusions, or takeaways\n"
            f"• Any actionable advice or recommendations\n"
            f"• Context and background information provided\n\n"
            f"Transcript segment:\n'''{chunk}'''"
        )
    else:
        prompt = (
            f"Provide a comprehensive and detailed summary of the following transcript section. "
            f"Include all key points, important details, main concepts, examples, and insights discussed:\n\n'''{chunk}'''"
        )
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def summarise_long_text(model: str, text: str, max_chunk_tokens: int = 3000, is_segment: bool = False) -> str:
    """Summarize long text without target length restrictions"""
    if _num_tokens(text) <= max_chunk_tokens:
        return _summarise_chunk(model, text, is_segment)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if _num_tokens(cur + " " + s) > max_chunk_tokens:
            chunks.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur:
        chunks.append(cur.strip())

    partials = [_summarise_chunk(model, c, is_segment) for c in chunks]
    section_summaries = '\n\n'.join([f'Part {i+1}: {summary}' for i, summary in enumerate(partials)])
    combined_prompt = (
        f"Create a comprehensive and detailed summary by combining and synthesizing the following "
        f"section summaries. Ensure all important information, key insights, examples, and details are preserved. "
        f"Organize the content logically and maintain the depth of information:\n\n"
        f"{section_summaries}"
    )
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def create_overall_summary(model: str, chapter_summaries: Dict[str, str], video_title: str) -> str:
    """Create a comprehensive overall summary from all chapter summaries"""
    all_summaries = '\n\n'.join([f'**{title}:**\n{summary}' for title, summary in chapter_summaries.items()])
    
    prompt = (
        f"Based on the following section summaries from the video '{video_title}', create a comprehensive "
        f"overall summary that:\n\n"
        f"• Captures the main theme and purpose of the entire video\n"
        f"• Highlights the most important key points and insights\n"
        f"• Identifies recurring themes or concepts\n"
        f"• Summarizes key takeaways and actionable advice\n"
        f"• Provides context about what viewers will learn\n"
        f"• Maintains sufficient detail to be valuable as a standalone summary\n\n"
        f"Section summaries:\n{all_summaries}"
    )
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ────
# Document Creation
# ────
DOC_TEMPLATE = """# {title}

## Overall Summary

{overall_summary}

## Detailed Section Summaries

{chapters}

"""

def word_count(txt: str) -> int:
    return len(txt.split())

def create_document(title: str, ch_summaries: Dict[str, str], overall_summary: str) -> str:
    ch_md = "\n\n".join(f"### {t}\n\n{s}" for t, s in ch_summaries.items())
    return DOC_TEMPLATE.format(title=title, overall_summary=overall_summary, chapters=ch_md)

# ────
# Main Processing Function
# ────
# ... (keep all the existing code the same until the process_youtube_video function)

def process_youtube_video(video_url: str, model: str, progress_placeholder, status_placeholder):
    """Process YouTube video and return markdown content"""
    try:
        # Fetch video info and transcript in one call
        status_placeholder.info("🔍 Fetching video information and transcript...")
        video_title, description, transcript = fetch_info_and_transcript(video_url)
        progress_placeholder.progress(0.30)  # Changed from 30 to 0.30
        
        # Parse chapters
        status_placeholder.info("📑 Parsing chapters...")
        chapters = parse_chapters(description)
        progress_placeholder.progress(0.40)  # Changed from 40 to 0.40
        
        # Split transcript
        status_placeholder.info("✂️ Processing transcript sections...")
        buckets = split_transcript(transcript, chapters)
        progress_placeholder.progress(0.50)  # Changed from 50 to 0.50
        
        # Initialize processing variables
        is_segment_based = len(chapters) == 0
        
        # Summarize chapters/segments
        chapter_summaries: Dict[str, str] = {}
        total_sections = len(buckets)
        
        for i, (title, text) in enumerate(buckets.items()):
            status_placeholder.info(f"🤖 Summarizing: {title}")
            chapter_summaries[title] = summarise_long_text(model, text, is_segment=is_segment_based)
            # Fixed progress calculation to stay within 0.0-1.0 range
            progress_placeholder.progress(0.50 + (0.30 * (i + 1) / total_sections))
        
        # Create overall summary
        status_placeholder.info("📋 Creating overall summary...")
        overall_summary = create_overall_summary(model, chapter_summaries, video_title)
        progress_placeholder.progress(0.90)  # Changed from 90 to 0.90
        
        # Generate document
        status_placeholder.info("📄 Generating final document...")
        doc = create_document(video_title, chapter_summaries, overall_summary)
        
        # Save to disk
        safe_title = sanitize_filename(video_title)
        output_file = f"{safe_title}_summary.md"
        output_path = Path(output_file)
        output_path.write_text(doc, encoding="utf-8")
        
        progress_placeholder.progress(1.0)  # Changed from 100 to 1.0
        status_placeholder.success(f"✅ Summary completed! Saved to: {output_file}")
        
        return doc, output_file, word_count(doc)
        
    except Exception as e:
        status_placeholder.error(f"❌ Error: {str(e)}")
        return None, None, 0
    
# ────
# Streamlit UI
# ────
def main():
    st.title("📺 YouTube Video Summarizer")
    st.markdown("---")
    
    # Create two columns
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.header("🎯 Input")
        
        # URL input
        video_url = st.text_input(
            "YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        # Model selection
        model_option = st.selectbox(
            "AI Model:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="Choose the OpenAI model for summarization"
        )
        
        # Processing button
        process_button = st.button(
            "🚀 Generate Summary",
            disabled=not video_url or st.session_state.processing,
            use_container_width=True
        )
        
        # Progress and status
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # File info
        if st.session_state.summary_content:
            st.markdown("---")
            st.subheader("📊 Summary Info")
            # This will be updated after processing
            
    with right_col:
        st.header("📖 Summary")
        
        # Markdown display area
        markdown_placeholder = st.empty()
        
        if not st.session_state.summary_content:
            markdown_placeholder.info("👈 Enter a YouTube URL and click 'Generate Summary' to see the results here!")
        else:
            markdown_placeholder.markdown(st.session_state.summary_content)
    
    # Process video when button is clicked
    if process_button and video_url:
        st.session_state.processing = True
        st.session_state.last_url = video_url
        
        # Clear previous content
        st.session_state.summary_content = ""
        markdown_placeholder.info("🔄 Processing video... Please wait...")
        
        # Process the video
        doc, output_file, word_count_result = process_youtube_video(
            video_url, model_option, progress_placeholder, status_placeholder
        )
        
        if doc:
            st.session_state.summary_content = doc
            markdown_placeholder.markdown(doc)
            
            # Update file info in left column
            with left_col:
                st.success(f"📁 Saved as: `{output_file}`")
                st.info(f"📊 Word count: {word_count_result:,}")
                
                # Download button
                st.download_button(
                    label="💾 Download Markdown",
                    data=doc,
                    file_name=output_file,
                    mime="text/markdown",
                    use_container_width=True
                )
        
        st.session_state.processing = False
        
        # Clear progress indicators
        progress_placeholder.empty()

if __name__ == "__main__":
    main()