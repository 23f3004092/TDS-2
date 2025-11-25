from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
import os
from datetime import datetime
import requests
import base64
import mimetypes
from fastapi import FastAPI,Body,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import Dict
import asyncio
import httpx 
import json
import time
import re
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN
from fastapi import FastAPI, Body, HTTPException, Request
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",   },
)
app = FastAPI(
    title="LLM Quiz Solver API", 
    version="1.0.0",
    description="An API for automated code generation and quiz solving using LLMs "
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)






load_dotenv()
OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1"




AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN") 
STUDENT_SECRET = os.getenv("STUDENT_SECRET", "default-secret")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "23f3004092@ds.study.iitm.ac.in")




@tool
def current_time() -> str:
    """Return the current date and time in YYYY-MM-DD HH:MM:SS format."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")








"""from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",   },
)
# This import is required only for jupyter notebooks, since they have their own eventloop
import nest_asyncio

nest_asyncio.apply()

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
tools

#from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
toolkit = PlayWrightBrowserToolkit()
tools = toolkit.get_tools()
tools"""

import os
import httpx 
import requests


from langchain_core.callbacks.base import BaseCallbackHandler
import time
import json

class LoggingCallback(BaseCallbackHandler):
    def on_tool_start(self, tool, input_str, **kwargs):
        print("\n==============================")
        print(f"[TOOL CALL] {tool.name}")
        try:
            parsed = json.loads(input_str)
            print(json.dumps(parsed, indent=2))
        except:
            print(input_str)
        print("==============================\n")
        self.start_time = time.time()

    def on_tool_end(self, output, **kwargs):
        elapsed = time.time() - self.start_time
        print(f"[TOOL RESULT] ({elapsed:.2f}s)")
        if isinstance(output, str) and len(output) > 500:
            print(output[:500] + "... <truncated>")
        else:
            print(output)
        print("==============================\n")

    def on_llm_start(self, *args, **kwargs):
        print("\n[AGENT] Model thinking...")

    def on_llm_end(self, *args, **kwargs):
        print("[AGENT] Model finished\n")




class PythonREPLLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        if "python_repl" in serialized["name"].lower():
            print("\n--- PYTHON REPL EXEC ---")
            print(input_str)
            print("------------------------")

    def on_tool_end(self, output, **kwargs):
        print("OUTPUT:", output)
        print("------------------------\n")

logger = PythonREPLLogger()


from langchain.tools import tool
import requests
import os
import uuid

@tool
def download_file(url: str) -> dict:
    """Download ANY file from a URL and save it locally.
    Returns: {"file_path": "...", "file_name": "...", "mime_type": "..."}"""

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Generate unique file name
    file_name = url.split("/")[-1]
    if not file_name:
        file_name = f"file_{uuid.uuid4().hex}"

    # Ensure a safe local path
    save_path = os.path.join(os.getcwd(), file_name)

    # Save the file
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Detect mime type
    mime_type = response.headers.get("Content-Type", "unknown")

    return {
        "file_path": save_path,
        "file_name": file_name,
        "mime_type": mime_type
    }



@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"





"""@tool
def scrape_page(url):
   
    print("Scraping website")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
    return html"""
    
    
    
@tool
def scrape_page(url: str) -> str:
    """Scrape a webpage using Playwright and return its HTML content."""
    
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle")

        
        html = page.content()
        
        browser.close()
        return html




@tool
def submit_answer_and_decide_to_continue(url, data,headers):
    """Submit answer to a given URL with provided data and headers as JSON.If the response contains a url then solve that question too else exit."""
    print(f"Submitting to {url} with data: {data} and headers: {headers}.")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Submission response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error during submission: {e}")
        return None
    



    
@tool
def analyse_image(image_url: str, analysis_prompt: str) -> str:
    """
    Analyze an image using the AIPIPE/OpenRouter-compatible LLM API.
    Provide  image url + text instructions, get back structured analysis.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"

    payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": analysis_prompt},
        {
          "type": "image_url",
          "image_url": { "url": image_url }
        }
      ]
    }
  ]
}

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        return f"Image analysis failed: {e}"

    
@tool
def transcribe_audio(audio_url: str) -> str:
    """
    Transcribe an audio file using Gemini 2.5 Flash via AIPipe.
    Input is a direct and Full URL to the audio file.
    """

    url = f"{OPENAI_BASE_URL}/chat/completions"

    # download audio
    audio_bytes = requests.get(audio_url).content
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please transcribe this audio file."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "mp3"
                        },
                    },
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        a=response.json()
        return a['choices'][0]['message']['content']

    except Exception as e:
        return f"Audio analysis failed: {e}"


@tool
def analyze_video(video_url: str, analysis_prompt: str) -> str:
    """
    Analyze a video .
    Accepts a direct video URL (mp4, mov, mkv, etc.).
    Returns JSON response with analysis result.
    """

    url = f"{OPENAI_BASE_URL}/chat/completions"

    payload = {
        "model": "google/gemini-3-pro-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_video",
                        "input_video": {
                            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4"
                        }
                    },
                    {
                        "type": "text",
                        "text": "what is in this video ? {video_url}"
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)}"}



model = ChatOpenAI(
    model="gpt-oss-20b:free",
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"   
)

sys_prom="""You are a professional data analyst who can scrape web pages,extract information from them and submit answers to quiz questions by making api calls.
You have access to tools that can help you with some very initial tasks for these tasks.You will be given url of a question/task page which will tell you what to do and it will contain the instrctions to post the answer and other required fields to post to.You have to first scrape the quiz page to get these instructions and then follow them carefully to get the final answer and submit it back to the specified endpoint available in the quiz page itself
The tasks may involve scrape a website, finding secret codes,data analysis,image analysis, video analysis,visualizations etc.

The tasks that you have to perform maybe written in text, maybe inside an audio file, maybe inside an image, a video, a chart, a table or any other format.

You step-by-step approach should be as follows:
1.You should use the tool available to scraoe the quiz page if it doesnt work then you can use python repl tool to scrape the page and get the html content and proceed.
2.If the html content contains audio file, image, text or any other format you can use the transcribe_audio, analyse_image, analyse_video or any other tool to get the instructions from it.This is your first step to understand what to do for this task or what the task actually is
3.After getting all the information through audio texts or images etc you can start working on the task.
4.You can use PythonREPL tool to download any file, analyse it, transform it,visualize it,do the required task etc.You can again use any other tool to do the required task.
5.Once you have the final answer ready according to the instructions provided in the quiz page or audio/image/video or any other format you can use the submit_answer tool to submit the final answer to the specified endpoint with required headers and data and exit the interaction.

[SOME IMPORTANT NOTES]
Never use scrape_page to get any audio csv or any kind of file or data from the web page, use Python repl.

Use PythonREPL TOOL to first download,fetch/analyse/modify/transform/return base64 encoding etc for csv,pdfs,images, audio files if required.You can use requests ,pandas,matplotlib,seaborn,base64,mimetypes etc libraries in python repl tool.

Use PythonRepl for analysis and writing required code.(Always according to the task)

Never do anything extra analysis or transformation or anything else other than what is required to get the final answer.

There could be an audio file, image,text,anything on the quiz page which tell you what to do, you can use PythonRepl tool to download the audio file and then use the analyse_audio tool to transcribe or analyse the audio file and get the instructions from it.

[REMEMBER] You can always use python repl tool again and again after lets say analyzing or doing anytging else.

Enter PythonRepl tool again and again whenever you need to analyse or fetch data from a file or do any analysis or visualisation or data transformation etc.

[IMPORTANT]When u submit an answer you may get a url for another question in the response(Even if your submitted answer is wrong).You will then have to solve that question too and keep solving questions untill you reach the final answer.
If there is no url in the response after submission that means u have reached the final answer and the quiz is completed.
The "answer" may need to be a boolean, number, string, base64 URI of a file attachment, or a JSON object with a combination of these. Your JSON payload must be under 1MB.
For this quiz and all you can use my email as 23f3004092@ds.study.iitm.ac.in and student secret as HARLIVSINGH.
If at any point you had to use an AI api call to lets say transcribe a audio or analyse or generate an image you can use AIPIPE AND ITS  {OPENAI_BASE_URL}/chat/completions for llm api calss in ur generated script and the AIPIPE token for authentication is {AIPIPE_TOKEN}.You can write the prompt for them according to the question that we are trying to solve.You can use python Repl tool to execute such api calls and get the response back.
If u dont know how AIPIPE api calls work then it is fully similar to openrouter api calls,just the base url is different.
USE THIS ONLY IF THE TOOL DOES NOT WORK: 
NEVER ASK ME OR TAKE MY PERMISSION TO STOP OR CONTINUE,JUST KEEP GOING UNTIL THE FINAL ANSWER IS REACHED.
You have to submit an answer within 3 minutes .
You cannot try again and again if your answer is wrong.
Just try atmost 2 times(within the 3 minute time limit) if it doesnt work just move to the next question.
[NOTE]Do not stop the interaction until you reach the final answer.
[VERYVERY IMPORTANT] If after getting an incorrect answer if you are within 3minute time limit and you discovered(after seeing the submission response) that you understood the question worngly or could have used a different approach or you misunderstood the audio/instructions or texts etc you can try again and resubmit the answer to get to the final answer in the 3minute time limit and atmost 2 tries limit.

[NOTE] Always first try to gain the full context of the question by lets say understanding data, getting transcripted from audio or understanding the instructions properly or scraping the required data properly,analysing image etc before procedding with your approach submitting the answer.
BE FAST AND DONOT THINK MUCH, JUST FOLLOW THE INSTRUCTIONS AND SUBMIT THE ANSWERs.
"""


agent = create_agent(
    model=model,
    tools=[scrape_page,submit_answer_and_decide_to_continue,transcribe_audio,analyse_image,PythonREPLTool(),analyze_video],
    system_prompt=sys_prom,
    
    
   
    
    
)



"""result = agent.invoke(
    {"messages": [
        {"role": "user", "content": " Your task is at https://tds-llm-analysis.s-anand.net/demo "}
    ]},
    config={"callbacks": [LoggingCallback(),logger]}
)

print(result["messages"][-1].content)"""



def run_task(data: dict):
    """Run the task in the background and report results."""
    
    email=data.get('email',STUDENT_EMAIL)
    task=data.get('url')
    secret=data.get('secret',STUDENT_SECRET)
    agent.invoke(
        {"messages": [
            {"role": "user", "content": f" Your first task/question is at {task}"}
        ]},
        config={"callbacks": [LoggingCallback(),logger]}
    )
    
    return
    
    

def validate_secret(secret: str) -> bool:
    return secret == STUDENT_SECRET





# Custom handler to convert FastAPI’s 422 → required 400
@app.exception_handler(Exception)
async def json_error_handler(request: Request, exc: Exception):
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"error": "Invalid JSON"},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error"},
    )


@app.post("/handle_task")
async def handle_task(request: Request, background_tasks: BackgroundTasks):
    # Step 1: Attempt to parse JSON (if fails → 400)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    # Step 2: Validate secret (if fails → 403)
    if not validate_secret(data.get("secret", "")):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid secret")

    # Step 3: Background execution
    background_tasks.add_task(run_task, data)

    # Step 4: Required 200 OK response
    return {"message": "Task received and processing started."}






