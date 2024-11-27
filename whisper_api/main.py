import whisper
import openai
import torch
from fastapi import FastAPI, File, UploadFile
from typing import List
from openai import OpenAI
client = OpenAI()

app = FastAPI()
        
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("turbo", device=device)

prompt = "Բարև ձեզ, Մերսոֆտ, խնդիր, ծրագիր, հաջողություն, զանգահարել, պատուհան, քանակ, գին, ՀԴՄ, սարք, սեղմել"

@app.post("/transcribe")
async def transcribe_audio(path:str):

    audio = whisper.load_audio(path)
    audio.itemsize
    
    # Transcribe the audio
    result = model.transcribe(audio,language="hy",condition_on_previous_text=True,verbose=False)

    # # Generate formatted output
    formatted_output = ""
    total_words = 0
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        formatted_output += f"{start_time} --> {end_time}  {text}\r\n"
        total_words += len(text.split())

    # Calculate the duration in minutes
    duration_minutes = (result["segments"][-1]["end"] / 60) if result["segments"] else 0

    # Return the response as JSON
    return {
        "minutes": round(duration_minutes, 2),
        "words_count": total_words,
        "text": formatted_output.strip()  # Remove trailing \r\n
    }

# @app.post("/transcribeParallel")
# async def transcribe_audio_paralel(path:str):
#     # Save the uploaded file temporarily
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = whisper.load_model("turbo", device=device)
#     audio = whisper.load_audio(path)
#     # audio = whisper.pad_or_trim(audio)
    
#     # Transcribe the audio
#     result = model.transcribe(audio,language="hy",condition_on_previous_text=True,verbose=False)

#     # # Generate formatted output
#     formatted_output = ""
#     for segment in result["segments"]:
#         start_time = segment["start"]
#         end_time = segment["end"]
#         text = segment["text"].strip()
#         formatted_output += f"{start_time} --> {end_time}  {text}"

    
#     return formatted_output

# @app.post("/transcribe2")
# async def transcribe_audio2(path:str):
#     audio_file= open(path,"rb")
#     transcription = client.audio.transcriptions.create(
#         model="whisper-1", 
#         language="hy",
#         file=audio_file,
#         response_format="srt",
#         prompt=prompt,
#         temperature=0.5
#     )

#     transc = "'1\n00:00:00,000 --> 00:00:06,040\nԲարև ձեզ, խնդրում ենք մնալգծում, առաջին իսկ ազադ աշխատակից ու կպատասխանի ձեզ։\n\n2\n00:00:06,040 --> 00:00:13,040\nԲարև ձեզ, հնդրում ենք մնալգծում, առաջին իսկ ազադ աշխատակից ու կպատասխանի ձեզ։\n\n3\n00:00:13,040 --> 00:00:32,119\nԵս երեկ զանգել էի, մեր մոտ խնդիր կար որ այդ կանակն ու գինա չէր պոխում, ինքը պետք ամ մտնենք տողի մեջ որ նոր պոխ էր, դորհիս մատիտով չէր պոխում։\n\n4\n00:00:32,119 --> 00:00:36,680\nԻնձ ասեց զինքը դժենք, դեսք զանգենք, բայց որ դժանգ է հավ, որ չէ դժվել է։\n\n5\n00:00:36,680 --> 00:00:40,360\n— Որ տեղից եք ուզում փոխել մատիտով? Ինչ գործար արգիշ էք անել։\n\n6\n00:00:40,619 --> 00:00:44,160\n— Ակացաս տեղ ասենք, ապրանք ենք առնոմ։\n\n7\n00:00:44,160 --> 00:00:47,560\n— Ապրանքի ձարբերման մեջ, որով սմատիտով պոխում եք, չի ստացվող։\n\n8\n00:00:47,560 --> 00:00:52,820\n— Բոչ։ Բետք մտնել տողը խմբագրես, որ ինգ է հասկանա։\n\n9\n00:00:52,959 --> 00:00:55,980\n— Պոխում ես պոխոմը, բայց թետ որ դարմածում ես տալել։\n\n10\n00:00:55,980 --> 00:00:59,340\nԻնք նորից հետաքծում մեր կանակը, ոչ։\n\n11\n00:00:59,340 --> 00:01:01,279\n— Կորխանչ եմ կացեք դրարունը։\n\n12\n00:01:01,279 --> 00:01:02,959\n— Արմեն։ — Արմեն։\n\n13\n00:01:02,959 --> 00:01:06,300\nԵս մի հատ ճշտ եմ ինչպոլում է, խնդիրը ուզան գաղացքա։\n\n14\n00:01:06,300 --> 00:01:09,199\n— Այս մի հատ ճշտ եմ, որդեղ լիրեք վանից ասիլում ես եմ կափությում ես երբ։\n\n15\n00:01:09,199 --> 00:01:12,580\n— Շարդահի չշտ եմ կափությում ես երբ։ — Բարի։\n\n\n'"
#         # Make chatGPT request
#     promptGpt = f"Make corrections in the following Armenian text which is generated by whisper model from audio.{transcription}"

#     response = openai.chat.completions.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are an assistant proficient in Armenian language and grammar."},
#         {"role": "user", "content": f"Please correct this text:\n\n{transc}"}
#     ],
#       # Adjust based on expected output length
#     )

#     # Extract and print the response
#     completion = response.choices[0].message.content
#     return completion