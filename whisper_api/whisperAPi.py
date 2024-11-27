# from fastapi import FastAPI, File, UploadFile, HTTPException
# from openai import OpenAI
# client = OpenAI()
# import io

# # Initialize the FastAPI app
# app = FastAPI()

# # Set your OpenAI API key here

# @app.post("/transcribe")
# async def transcribe_audio(audioFile: UploadFile = File(...)):
#     # Check if the uploaded file is an audio file
#     if not audioFile.content_type.startswith("audio"):
#         raise HTTPException(status_code=400, detail="File is not an audio file.")

#     # Read the audio file
#     audio_bytes = await audioFile.read()
#     audio_file= open("C:/Users/hovse/Downloads/test2.mp3", "rb")
    
#     try:
#         # Call the Whisper API
#         response = client.audio.transcriptions.create(model="whisper-1", 
#                                                       file=audio_file,
#                                                       response_format="verbose_json",
#                                                       timestamp_granularities=["word"],
#                                                       language='hy')

#         return {"transcription": response['text']}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Error processing audio file.")

