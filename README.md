# Grammar Scoring from Spoken Audio

The goal here is simple to say and a bit tricky to do: listen to a 45–60 second spoken answer and predict a grammar score between 0 and 5.

The catch is the data. I only have **769 labelled clips**. That's nowhere near enough to fine-tune a big speech or language model without it memorising the training set. So I took a different route.

## The idea

Instead of training a neural network, I use pretrained ones as **frozen feature extractors**. They turn each clip into numbers that describe how it sounds and what was said. Then I let small, cheap models (Ridge regression and LightGBM) learn the actual mapping from those numbers to the grammar score. Cheap models are much harder to overfit on a few hundred samples, which is the whole point.

## What goes into each clip

I wanted the model to see the clip from a few different angles, because grammar shows up in more than one place.

**1. Clean-up first.** Everything is resampled to 16 kHz mono, and silence at the start and end is trimmed off.

**2. How the person speaks (about 54 features, librosa).** Things like:
- how much of the clip is speech vs. pauses (pauses of 250 ms or longer, how many, how long, the longest one)
- speaking rate and articulation rate, estimated from onsets
- rhythm, i.e. how evenly spaced the speech bursts are
- general sound stats: energy, spectral shape, MFCCs, and pitch (via YIN)

Hesitant, choppy speech tends to look very different from smooth speech, and that often goes hand in hand with weaker grammar.

**3. How it sounds, according to a speech model.** I run `wav2vec2-base-960h` over the audio in 20-second windows and average the frame embeddings. I take two layers, a middle one (layer 8) and the last one, so I get both lower-level acoustic information and higher-level information.

**4. What was said.** Whisper transcribes every clip (the `small` model on GPU, `base` on CPU).

**5. How the text reads, according to a language model.** The transcript goes through `deberta-v3-base`, and I mean-pool the token states (ignoring padding) into one vector per clip.

**6. Simple text stats.** Word count, vocabulary variety (type-token ratio), average sentence length, average word length, share of long words, filler words ("um", "uh"…), immediate word repeats, and words per second.

All of this ends up as one feature matrix of about **2,369 columns** per clip: 65 hand-crafted, 1,536 from wav2vec2, and 768 from DeBERTa.

### A thing worth knowing about Whisper

Whisper is good at being *helpful*, which is a problem here. It tends to quietly fix stumbles, repeated words and small grammar slips when it transcribes. So the text side can look cleaner than what the person really said. That's the reason I didn't rely on the transcript alone. The raw-audio embedding and the prosody features are there to pick up what the transcript smooths over.

## The models

I use two very different models and average them.

- **Ridge regression** on the full standardised feature matrix. The regularisation strength was picked by sweeping a handful of values and checking cross-validated error; `alpha = 1000` won.
- **LightGBM** on the hand-crafted features plus a **PCA-reduced** version of each embedding (48 components each). Trees don't love 768-dimensional dense vectors, so squashing them down helps. The PCA is fit only on the training part of each fold, so nothing leaks from the validation data.

The final prediction is a plain 50/50 average of the two, clipped to the valid range of 0–5. Ridge is good at using the big embeddings; LightGBM is good at the messy, non-linear hand-made features. Their mistakes don't overlap perfectly, which is why blending helps.

## How I checked it

5-fold cross-validation with shuffling and a fixed seed, using the **same folds for every model** so the comparison is fair. The scores that matter are the out-of-fold ones (predictions on clips the model never trained on):

- Out-of-fold RMSE for the blend lands around **0.51–0.58** across folds.
- For context, just predicting the average score every time gives an error equal to the label spread, about **1.24**.
- Blending beat or matched the better single model in every fold.

I also report in-sample training error and Pearson correlation, but I treat those as a sanity check only. They're always flattering.

For the test set, predictions come from all 5 fold models (for both model types) averaged together, so every prediction is the result of 10 models voting.

## Running it

Everything expensive (acoustic features, wav2vec2 embeddings, Whisper transcripts, DeBERTa embeddings, and the final feature matrices) is **cached to disk**. The first run is slow because of the neural nets; after that, re-running or tweaking the models takes seconds.

Rough order of the notebook:

1. Setup and config
2. Load data
3. Feature code (prosody, wav2vec2, Whisper, DeBERTa, lexical)
4. Build and cache the feature matrices
5. Cross-validated Ridge + LightGBM
6. Metrics, plots (predicted vs. true, top LightGBM feature importances), and the final predictions file

A GPU makes the embedding and transcription steps much faster, but it also works on CPU with the smaller Whisper model.
