from __future__ import annotations      #

import argparse
import os
import sys
from pathlib import Path        # Work with filepaths

from langchain_core.output_parsers import StrOutputParser       # input -> prompt formatted -> output
from langchain_core.prompts import PromptTemplate               # defines how prompts are structured
from langchain_google_genai import ChatGoogleGenerativeAI       # connect langchain to google models
from dotenv import load_dotenv

PLATFORMS = ("Meta", "Instagram", "LinkedIn", "Youtube")

# Load environment variables from .env file
load_dotenv()


def _platform_prompt(platform: str) -> PromptTemplate:
    platform = platform.strip()
    if platform not in PLATFORMS:
        raise ValueError(f"Unsupported platform {platform}. Choose on of: {", ".join(PLATFORMS)}")

    templates: dict[str, str] = {
        "Meta": (
            "you are a social media copywriter.\n"
            "Write a Meta (Facebook) post about: {topic}\n\n"
            "Tone: friendly, community-first, conversational.\n"
            "Format requirements:\n"
            "- 1 hook line\n"
            "- 2-3 short paragraphs (easy to read on mobile)\n"
            "- 1 clear call-toaction question at the end\n"
            "- 0-3 relevant emojies (not excessive)\n"
            "- No hashtags (or at most 1 subtle hashtage)\n"
        ),
        "Instagram": (
            "You are an Instagram content creator.\n"
            "Create an Instagram caption about: {topic}\n"
            "Tone: trendy, engaging, and visually descriptive.\n"
            "Format requirements:\n"
            "- 1-2 hook lines\n"
            "- 2-3 short paragraphs (easy to read on mobile)\n"
            "- 3-5 relevant emojies (to enhance visual appeal)\n"
            "- 5-10 relevant hashtags (to increase discoverability)\n"
        ),
        "LinkedIn": (
            "You are a LinkedIn content strategist.\n"
            "Write a LinkedIn post about: {topic}\n"
            "Tone: professional, insightful, and engaging.\n"
            "Format requirements:\n"
            "- 1 hook line\n"
            "- 2-3 informative paragraphs (providing value to professionals)\n"
            "- 1 clear call-to-action question at the end\n"
            "- 0-3 relevant emojies (to maintain professionalism)\n"
            "- 3-5 relevant hashtags (to increase visibility within professional circles)\n"
        ),
        "Youtube": (
            "You are a YouTube content creator.\n"
            "Write a YouTube video description about: {topic}\n"
            "Tone: engaging, informative, and optimized for search.\n"
            "Format requirements:\n"
            "- 1-2 hook lines\n"
            "- 2-3 informative paragraphs (providing value to viewers)\n"
            "- 3-5 relevant emojies (to enhance visual appeal)\n"
            "- 5-10 relevant hashtags (to increase discoverability)\n"
        ),
    }

    return PromptTemplate(input_variables=["topic"], template=templates[platform])


def generate_post(*, topic: str, platform: str, api_key: str) -> str:

    """
    Generate a social media post for a given topic and platform using Google Gemini.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                                 temperature=0.7,
                                 api_key=api_key)

    prompt = _platform_prompt(platform)
    chain = prompt | llm | StrOutputParser()

    return chain.run(topic=topic).strip()

# How human users will interact with the script from the command line. Example usage:
# python main.py --topic "The benefits of meditation" --platform "LinkedIn"
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate social media posts for different platforms using Google Gemini.")
    parser.add_argument("--topic", type=str, required=True, help="The topic to generate the post about.")
    parser.add_argument("--platform", type=str, required=True, choices=PLATFORMS, help="The social media platform to generate the post for.")
    parser.add_argument("--api-key", type=str, default=os.getenv("GOOGLE_API_KEY"), help="Google API key. Can also be set via the GOOGLE_API_KEY environment variable.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)

    api_key = (args.api_key or "").strip() or (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not args.api_key:
        raise ValueError("API key is required. Please provide it via the --api-key argument or the GOOGLE_API_KEY environment variable.")
    try:
        post = generate_post(topic=args.topic, platform=args.platform, api_key=api_key)
        print("\nGenerated Post:\n")
        print(post)
    except Exception as e:
        print(f"Error generating post: {e}", file=sys.stderr)
        sys.exit(1)
