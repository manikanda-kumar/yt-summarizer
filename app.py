#!/usr/bin/env python
#
# Run locally:  python app.py
# ----
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import yt_dlp    # youtube-dl replacement
from openai import OpenAI    # pip install openai
import tiktoken    # pip install tiktoken

# ----
# FastAPI bootstrap & helpers
# ----
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("yt_summarizer")

app = FastAPI(
    title="YouTube Video Summarizer",
    description="Turns any YouTube video into a detailed markdown summary.",
    version="1.0.0",
)

# CORS so the browser can call the API when the HTML is opened directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve everything inside "static/" under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the landing page
@app.get("/", include_in_schema=False)
def root_index():
    return FileResponse("static/index.html")


# ----
# OpenAI helper
# ----
def get_openai_client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key must be provided.")
    return OpenAI(api_key=api_key)

# ----
# Helper Functions
# ----
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
                logger.warning(f"Could not parse transcript data: {e}")

        if not transcript:
            raise RuntimeError("No English transcript/captions available for this video")

        return title, description, transcript

    except Exception as exc:
        if "transcript" in str(exc).lower() or "caption" in str(exc).lower():
            raise RuntimeError(f"Transcript not available: {exc}")
        else:
            logger.warning(f"Could not fetch video info - {exc}")
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

def _num_tokens(txt: str, encoder):
    return len(encoder.encode(txt))

def _summarise_chunk(client, model: str, chunk: str, is_segment: bool = False) -> str:
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

def summarise_long_text(client, encoder, model: str, text: str, max_chunk_tokens: int = 3000, is_segment: bool = False) -> str:
    """Summarize long text without target length restrictions"""
    if _num_tokens(text, encoder) <= max_chunk_tokens:
        return _summarise_chunk(client, model, text, is_segment)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if _num_tokens(cur + " " + s, encoder) > max_chunk_tokens:
            chunks.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur:
        chunks.append(cur.strip())

    partials = [_summarise_chunk(client, model, c, is_segment) for c in chunks]
    section_summaries = '\n\n'.join([f'Part {i+1}: {summary}' for i, summary in enumerate(partials)])
    combined_prompt = (
        f"Create a comprehensive and detailed summary by combining and synthesizing the following "
        f"section summaries. Ensure all important information, key insights, examples, and details are preserved. "
        f"Organize the content logically and maintain the depth of information:\n\n"
        f"{section_summaries}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def create_overall_summary(client, model: str, chapter_summaries: Dict[str, str], video_title: str) -> str:
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

# ----
# Document Creation
# ----
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

# ----
# Main Processing Function
# ----
def process_youtube_video(video_url: str, model: str, api_key: str):
    """Process YouTube video and return markdown content"""
    try:
        # Initialize OpenAI client and encoder
        client = get_openai_client(api_key)
        encoder = tiktoken.encoding_for_model(model if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"] else "gpt-3.5-turbo")

        # Fetch video info and transcript in one call
        logger.info("📥 Fetching video information and transcript...")
        video_title, description, transcript = fetch_info_and_transcript(video_url)

        # Parse chapters
        logger.info("📑 Parsing chapters...")
        chapters = parse_chapters(description)

        # Split transcript
        logger.info("📝 Processing transcript sections...")
        buckets = split_transcript(transcript, chapters)

        # Initialize processing variables
        is_segment_based = len(chapters) == 0

        # Summarize chapters/segments
        chapter_summaries: Dict[str, str] = {}
        total_sections = len(buckets)

        for i, (title, text) in enumerate(buckets.items()):
            logger.info(f"🔎 Summarizing: {title}")
            chapter_summaries[title] = summarise_long_text(client, encoder, model, text, is_segment=is_segment_based)

        # Create overall summary
        logger.info("🧠 Creating overall summary...")
        overall_summary = create_overall_summary(client, model, chapter_summaries, video_title)

        # Generate document
        logger.info("📄 Generating final document...")
        doc = create_document(video_title, chapter_summaries, overall_summary)

        logger.info("✅ Summary completed!")
        return doc, video_title, word_count(doc)

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise e

# ----
# API endpoints
# ----
@app.post("/summarize")
def summarize_endpoint(
    api_key: str = Form(...),
    youtube_url: str = Form(...),
    ai_model: str = Form(...)
):
    try:
        doc, video_title, wc = process_youtube_video(youtube_url, ai_model, api_key)
        
        return {
            "summary": doc,
            "title": video_title,
            "word_count": wc
        }
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=str(e)
        )

@app.post("/download")
def download_summary(
    summary: str = Form(...),
    title: str = Form(...)
):
    try:
        # Create safe filename
        safe_title = sanitize_filename(title)
        filename = f"{safe_title}_summary.md"
        
        # Return the markdown content as a file download
        return Response(
            content=summary,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error creating download: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# ----
# Dev entry-point
# ----
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )