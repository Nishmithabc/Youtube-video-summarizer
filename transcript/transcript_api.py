from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_video_id(youtube_url):
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/]+/.*|(?:v|e(?:mbed)?)|.*[?&]v=)|youtu\.be/)([^&]{11})'
    match = re.search(regex, youtube_url)
    return match.group(1) if match else None

def yt_transcript_api(youtube_url):
    video_id = get_video_id(youtube_url)

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    return " ".join(item.text for item in transcript)