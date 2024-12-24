import tempfile
import whisper
import openai
import torch
import os
import uuid
import subprocess
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime,timedelta
from bson import ObjectId
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import asyncio


# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("turbo", device=device)

# MongoDB setup (replace with your Azure MongoDB connection string)
WATCH_FOLDER = os.getenv("WATCH_FOLDER", "./default_folder")
MONGO_URI = os.getenv("MONGO_URI", "default_mongo_uri")
ATS_CUSTOMER_KEY = os.getenv("ATS_CUSTOMER_KEY","default_ats_key")


#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

client = AsyncIOMotorClient(MONGO_URI)
db = client["WhisperTranscription"]
collection = db["Transcriptions"]



async def process_audio_file(file_path: str, unique_id: str):
    """Process the audio file, transcribe it, and save the result to MongoDB."""
    try:
        # Load the audio file
        audio = whisper.load_audio(file_path)

        # Transcribe the audio
        result = model.transcribe(audio, language="hy", condition_on_previous_text=True, verbose=False)

        # Generate formatted output
        formatted_output = ""
        total_words = 0
        for index, segment in enumerate(result["segments"]):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            formatted_output += f"{index+1}:  {text}\n"
            #f"{start_time:.2f} --> {end_time:.2f}  {text}\n"
            total_words += len(text.split())

        # Make GPT-4o correction request
        # prompt_gpt = f"Correct the following Armenian text:\n\n{formatted_output}"
        # response = openai.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": "You are an Armenian text editor."},
        #         {"role": "user", "content": prompt_gpt}
        #     ]
        # )

        # corrected_text = response.choices[0].message.content

        

        # Save the result to MongoDB
        await collection.insert_one({
            "_id": ObjectId(),
            "uniqueID": unique_id,
            "filepath": file_path,
            "model1transcription": formatted_output,
            "model2transcription": "",
            "total_words": total_words,
            "processed_at": datetime.utcnow()
        })

        print(f"Processed and saved transcription for: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

@app.get("/transcription/{unique_id}")
async def get_transcription(unique_id: str):
    """Endpoint to retrieve the transcription by unique ID."""
    result = await collection.find_one({"uniqueID": unique_id})
    if result:
        return {
            "uniqueID": result["uniqueID"],
            "transcription": result["model1transcription"],
            "total_words": result["total_words"],
            "processed_at": result["processed_at"]
        }
    else:
        raise HTTPException(status_code=404, detail="Transcription not found")
    


def make_get_request(url, params=None, headers=None):
    
    try:
        with  requests.get(url, headers=headers, params=params, timeout=10) as api_response:
            api_response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            result = api_response.json()
            # print(result)
            return result
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
def download_audio_call(url, params=None, headers=None):
    
    try:
        with  requests.get(url, headers=headers, params=params, timeout=10) as api_response:
            api_response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            return api_response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    

# @app.get("/transcriptionhistory}")
async def fetch_history_and_call_records():
    """Fetches history data and makes call-record requests for each uniqueid."""
    
    # Endpoint URLs
    history_url = "https://account.ats.am/docs/api/v1/history"
    call_record_url = "https://account.ats.am/docs/api/v1/call-record"
    
    # Common headers
    headers = {
        "accept": "application/json"
    }

    date_end = datetime.now()
    date_start = date_end - timedelta(days=1)
    
    # History endpoint parameters
    history_params = {
        "key": ATS_CUSTOMER_KEY,
        "dateStart": date_start.strftime("%Y-%m-%d"),
        "dateEnd": date_end.strftime("%Y-%m-%d"),
        "start": 0,
        "rows": 10
    }

    # Fetch history data
    history_response =  make_get_request(history_url, params=history_params, headers=headers)
    
    if history_response:
        print("History Response Received")
        call_records = history_response.get("docs", [])
    if not call_records:
        print("No call records found.")
        return
    try:
        # Iterate over each call record and get the uniqueid
        for record in call_records:
            # Extract the uniqueid for each record
            uniqueid = record['uniqueid']
            print(f"Unique ID: {uniqueid}")
            if uniqueid:
                result =  await collection.find_one({"uniqueID": uniqueid})
                if result:
                    continue
                call_record_params = {
                    "uid": uniqueid,
                    "key": ATS_CUSTOMER_KEY
                }
                
                # Fetch call record data
                call_record_response = download_audio_call(call_record_url, params=call_record_params, headers=headers)
                
                if call_record_response:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
                         temp_audio_file.write(call_record_response.content)
                         temp_audio_path = temp_audio_file.name

                    await process_audio_file(temp_audio_path,uniqueid)
                    try:
                        os.remove(temp_audio_path)
                        print(f"Temporary file deleted: {temp_audio_path}")
                    except FileNotFoundError:
                        print(f"File not found: {temp_audio_path}")
                    except Exception as e:
                        print(f"Error deleting file: {e}")
                else:
                    print(f"Failed to fetch call record for UID {uniqueid}")
            else:
                print("No uniqueid found in the record.")
    except Exception as e:
        print(f"Error in loop: {e}")
    return "Request processed"


# Start the background scheduler
# scheduler = AsyncIOScheduler()
# scheduler.add_job(fetch_history_and_call_records, "interval", seconds=3600)  # Check every 1 hour
# scheduler.start()

# Keep the async loop running

    
# @app.post("/transcribe")
# async def transcribe_audio(path:str):
#         diarization = pipeline(path)
#         temp_dir = "C:/Users/hovse/Downloads/Temp"
#         combined_result = []

#         # Loop through diarized segments
#         for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
#             start, end = turn.start, turn.end
#             segment_file = os.path.join(temp_dir, f"segment_{i}.wav")

#             # Split audio using ffmpeg
#             subprocess.run([
#                 "ffmpeg", "-i", path,
#                 "-ss", str(start), "-to", str(end),
#                 "-c", "copy", segment_file,
#                 "-loglevel", "error"  # Suppress ffmpeg output
#             ])

#             # Transcribe with Whisper
#             result = model.transcribe(segment_file,language="hy",condition_on_previous_text=False,verbose=False,word_timestamps=True)
#             combined_result.append(f"{speaker} [{start:.1f} - {end:.1f}]: {result['text']}")

#         # Return the combined result as a single string
#         return "\n".join(combined_result)
    
    # Transcribe the audio
    # result = model.transcribe(audio,language="hy",condition_on_previous_text=False,verbose=False,word_timestamps=True,temperature=1)

    # # Generate formatted output
    # formatted_output = ""
    # total_words = 0
    # for segment in result["segments"]:
    #     start_time = segment["start"]
    #     end_time = segment["end"]
    #     text = segment["text"].strip()
    #     formatted_output += f"{start_time} --> {end_time}  {text}\r\n"
    #     total_words += len(text.split())

    
    # return formatted_output

    
