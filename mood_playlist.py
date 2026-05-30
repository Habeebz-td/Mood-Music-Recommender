import random
import webbrowser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Playlist dictionary
playlist = {

    "happy":[("Happy - Pharrell Williams 🎶", "https://music.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Can't Stop the Feeling - Justin Timberlake 😄", "https://music.youtube.com/watch?v=ru0K8uYEZWw"),
        ("Uptown Funk - Bruno Mars 🎤", "https://music.youtube.com/watch?v=OPf0YbXqDm0")],

    "romantic":[("Someone Like You - Adele 😢", "https://music.youtube.com/watch?v=hLQl3WQQoQ0"),
                ("Fix You - Coldplay 💔", "https://music.youtube.com/watch?v=k4V3Mo61fJM"),
                ("The Night We Met - Lord Huron 🌙", "https://music.youtube.com/watch?v=KtlgYxa6BMU")],

    "nostalgic":[("Perfect - Ed Sheeran ❤️", "https://music.youtube.com/watch?v=2Vv-BfVoq4g"),
                 ("All of Me - John Legend 💖", "https://music.youtube.com/watch?v=450p7goxZqg"),
                 ("Thinking Out Loud - Ed Sheeran 💞", "https://music.youtube.com/watch?v=lp-EO5I60KA")],

    "energetic": [("Eye of the Tiger - Survivor 🐯", "https://music.youtube.com/watch?v=btPJPFnesV4"),
                  ("Stronger - Kanye West 💪", "https://music.youtube.com/watch?v=AO1iITF09OE"),
                  ("Don't Stop Me Now - Queen ⚡", "https://music.youtube.com/watch?v=HgzGwKwLmgM")],

    "sad": [("Let Her Go - Passenger 😔", "https://music.youtube.com/watch?v=RBumgq5yVrA"),
            ("When I Was Your Man - Bruno Mars 💔", "https://music.youtube.com/watch?v=ekzHIouo8Q4"),
            ("Jealous - Labrinth 😢", "https://music.youtube.com/watch?v=50VWOBi0VFs")]
}

# # Keywords to detect mood
# mood_keywords = {
#     "happy": ["happy", "excited", "joy", "awesome", "great", "fun"],
#     "romantic": ["love", "romantic", "crush", "date", "heart"],
#     "nostalgic": ["memories", "past", "old times", "school", "college"],
#     "energetic": ["workout", "energy", "hyped", "running", "party"],
#     "sad": ["sad", "lonely", "cry", "upset", "depressed"]
# }

# Load Hugging Face sentiment model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Get user's input
feeling = input("How are you feeling today? ")

# Detect mood using Transformers
inputs = tokenizer(feeling, return_tensors="pt")
outputs = model(**inputs)
scores = outputs.logits
pred = torch.argmax(scores) + 1

# Map stars to moods
if pred <= 2:
    detected_mood = "sad"
elif pred == 3:
    detected_mood = "nostalgic"
elif pred == 4:
    detected_mood = "happy"
else:
    detected_mood = "energetic"


feeling = input("How are you feeling today? ").lower()

if detected_mood:
    songs = playlist[detected_mood][:]
    random.shuffle(songs)
    print(f"\n🎧 Looks like you're feeling {detected_mood}! Here's your playlist:")
    for i, (song_name, _) in enumerate(songs, start=1):
        print(f"{i}. {song_name}")

    choice = int(input("\nEnter the number of the song to play: "))
    if 1 <= choice <= len(songs):
        webbrowser.open(songs[choice - 1][1])
    else:
        print("❌ Invalid choice.")
else:
    print("\n😕 I couldn't figure out your mood. Please try using mood words like happy, sad, romantic, etc.")