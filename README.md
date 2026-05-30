# Mood-Based Music Recommender 🎧

A Python project that reads how you're feeling, figures out your mood using an AI model, and recommends songs — then opens the one you pick directly in YouTube Music.

This started as a simple idea but ended up being my first proper project using a real machine learning model. I wanted it to actually *understand* what you type, not just look for keywords.

---

## What it does

- Asks you how you're feeling in plain English
- Runs your input through a **BERT-based sentiment model** (from Hugging Face) to detect your mood
- Maps the result to one of five moods: `happy`, `sad`, `romantic`, `nostalgic`, or `energetic`
- Shows you a shuffled playlist for that mood
- Opens your chosen song directly in YouTube Music via your browser

---

## How to run it

**1. Install the dependencies**

```bash
pip install transformers torch
```

**2. Run the script**

```bash
python mood_music.py
```

The first time you run it, it'll download the BERT model automatically (about 600MB). After that it loads from cache.

---

## Example

```
How are you feeling today? I'm exhausted but kind of hyped for tonight

🎧 Looks like you're feeling energetic! Here's your playlist:
1. Don't Stop Me Now - Queen ⚡
2. Stronger - Kanye West 💪
3. Eye of the Tiger - Survivor 🐯

Enter the number of the song to play: 1
```

Your browser opens the song. Done.

---

## How the mood detection works

I used the `nlptown/bert-base-multilingual-uncased-sentiment` model from Hugging Face. It's a BERT model trained to give a star rating (1–5) to text — like a review score.

I mapped those ratings to moods like this:

| Stars | Mood |
|---|---|
| 1–2 | Sad |
| 3 | Nostalgic |
| 4 | Happy |
| 5 | Energetic |

Romantic mood can be added as a manual override or keyword layer — it's in the playlist, just needs better mapping.

---

## Things I learnt building this

- How to use Hugging Face Transformers in a real project
- What tokenizers do and why they're needed before passing text to a model
- How `torch.argmax` picks the most likely prediction
- How `webbrowser.open()` works for launching URLs from Python

---

## Possible improvements

- Better mood mapping (especially for romantic)
- Add more songs to each playlist
- Build a proper GUI with Tkinter or a web interface
- Let users add their own songs

---

## Built with

- Language: Python 3
- Libraries: `transformers`, `torch`, `tkinter`, `webbrowser`, `random`
- Model: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
