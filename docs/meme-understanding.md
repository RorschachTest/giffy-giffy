# Meme understanding: background and what to build on

The engine in `app/humor.py`, `app/intent.py` and `app/understand.py` answers two
questions without an LLM:

1. **What does this meme say when someone sends it in a chat?** (intent, who it
   is aimed at, how sure we are, and the evidence)
2. **What makes it funny?** (which humor mechanisms fire, and a funniness estimate)

This note lists the open projects and datasets the design draws on, what each
one is good for here, and why none of them is used as-is.

## Projects and datasets on GitHub (and nearby)

| Project | What it is | What we took | Why not use it as-is |
|---|---|---|---|
| [MUStARD](https://github.com/soujanyaporia/MUStARD) (Castro et al., ACL 2019) | Sarcasm in TV-show clips (audio + video + text), each with the dialogue before it | The core idea: a clip's meaning depends on the **preceding utterances**; context + multimodal beats any single signal | English sitcoms, sarcasm only (one intent of our 16), trained deep models |
| [UR-FUNNY](https://github.com/ROC-HCI/UR-FUNNY) (Hasan et al., EMNLP 2019) | Punchline detection in TED talks with the sentences that build up to it | **Setup -> punchline** framing: in a chat, the message is the setup and the meme is the punchline | Talks, not memes; PyTorch models trained on TED features |
| [MemeCap](https://github.com/eujhwang/meme-cap) (Hwang & Shwartz, EMNLP 2023) | 6.3K memes with post title, literal image caption, **meme meaning** and visual metaphors | The split between what the picture literally shows and what the meme means; the meme expert in `intent.py` reads the clip's moods/topics as the "literal" layer and maps them to meanings | Static images; their meaning task is generative (VLMs), which is what we avoid |
| [Memotion / SemEval-2020 Task 8](https://github.com/cozek/memotion2020-code) | ~7K memes labelled humorous / sarcastic / offensive / motivational with intensity | Label set for funniness and sarcasm; good **calibration data** for `fit_funny` | Images with OCR text; best systems reached only ~0.51 macro-F1, so labels are noisy |
| [SICKNet](https://github.com/xing-wei-zeng/SICKNet) | Humor detection from the semantic gap between setup and punchline plus commonsense | Incongruity = distance between setup and punchline **plus** a knowledge bridge that resolves it (our `BRIDGES`) | Trained network on text jokes |
| [ColBERT humor](https://arxiv.org/abs/2004.12765) (Annamoradnejad & Zoghi) | Sentence embeddings per sentence, parallel layers judge their congruity; 200K short texts | Sentence-level embeddings compared with each other, rather than one embedding of the whole text | Dataset is behind IEEE DataPort; formal one-liners, not chat |
| [LOLgarithm](https://github.com/tanisha1112/LOLgarithm) | Humor classifier on incongruity, ambiguity and phonetic features | Hand-built incongruity features alongside embeddings | Text jokes only |
| Tanaka et al., [Learning to Evaluate Humor in Memes Based on the Incongruity Theory](https://aclanthology.org/2022.cai-1.9) | 7.5K memes with crowd humor scores; image-caption incongruity module | Picture-vs-words incongruity (our CLIP `visual_gap`); dropping items whose rating is pure personal taste | No code link found |
| Bates et al., *A Template Is All You Meme* (NAACL 2025) | Knowledge base of 5.2K meme templates and 54K instances, distance-based template lookup | Treating the **template** as the unit of meaning (our per-clip Dirichlet over intents, `folk_names`) | Static image templates; no code link found |
| SemEval-2021 Task 7 [HaHackathon](https://aclanthology.org/2021.semeval-1.34) | 10K texts rated for humor and offense by 20 annotators each | Humor as a **rating** with disagreement, not a yes/no | Text only; transformer fine-tuning |

Also relevant as theory: benign violation (McGraw & Warren 2010), Raskin's
Semantic Script Theory of Humor (script opposition), and Riloff et al. 2013
(sarcasm as positive sentiment about a negative situation).

## How the engine works

```
            clip facts (title, caption, dialogue, moods, topics, comments, CLIP gap)
                │                                  chat context + sender caption
                ▼                                                │
   humor.py  setup/punchline vectors ──► 12 mechanisms (0..1) ◄──┘
                │                              │
                │                              ▼
                │                 logistic funniness (prior weights, MAP refit)
                ▼
   intent.py meme expert   context expert        valence expert     caption expert
             moods,topics,  dialogue acts ×       sentiment contrast  pronouns, "me rn",
             text, ironic   transition matrix     message vs meme     sarcasm marks
             use, Dirichlet (adjacency pairs,
             over feedback   Dirichlet-updated)
                └──────────── product of experts (geometric pooling) ──────────┘
                                         ▼
                      intent, aimed at, confidence, evidence
```

* All text similarities are **z-scores** against a fixed set of everyday chat
  sentences (`semspace.BACKGROUND`), so thresholds do not depend on the
  embedding model's raw cosine range.
* Incongruity-resolution checks only the setup's and the punchline's own top
  three "bridge" situations: taking the best of all ~50 would resolve any
  random pair by chance.
* Product of experts, not averaging: the winning reading must be plausible to
  every expert that has an opinion (a shocked clip after good news reads as
  disbelief, not as "shocked" or "celebrate").
* Memes are polysemous: a quarter of the clip's own reading goes to its ironic
  uses (a happy clip is also the classic sarcasm/mockery reply).

## Measuring and improving it

1. Collect feedback through `POST /understand/feedback`; `GET /understand/stats`
   reports agreement between the engine and people.
2. For an offline benchmark, the closest labelled data is MUStARD (sarcasm with
   context: does the engine pick `sarcasm` where they say sarcastic?) and
   Memotion (humorous / sarcastic labels for `funny`). Both need their text and
   moods mapped onto clip facts first.
3. The cheapest improvements are in the tables: prototype sentences
   (`INTENTS`, `CONTEXT_ACTS`, `BRIDGES`) in the languages your chats use, and
   the `TRANSITIONS` weights.

Not built: OCR of on-screen text (a lot of meme meaning lives there), audio
cues (laugh tracks, music), and an offline benchmark script.
