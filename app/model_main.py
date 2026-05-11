# inference_model.py
import random
import torch
import torch.nn as nn
import math
import re
from pythainlp.tokenize import word_tokenize

# =========================
# CONFIG
# =========================
PAD, SOS, EOS, UNK = 0, 1, 2, 3
EMB_SIZE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# VOCAB
# =========================
class Vocab:
    def __init__(self):
        self.w2i = {}
        self.i2w = {}

    def load(self, vocab_dict):
        self.w2i = vocab_dict
        self.i2w = {v: k for k, v in vocab_dict.items()}

    def tokenize(self, text):

        text = re.sub(r'(\d)', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)

        return word_tokenize(
            text,
            engine="newmm",
            keep_whitespace=False
        )

    def encode(self, text):
        return [
            self.w2i.get(w, UNK)
            for w in self.tokenize(text)
        ]

    def decode(self, ids):

        tokens = [
            self.i2w.get(i, "")
            for i in ids
            if i not in [PAD, SOS, EOS]
        ]

        text = " ".join(tokens)

        # merge numbers
        text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)

        # decimal
        text = re.sub(r'\s*\.\s*', '.', text)

        # slash
        text = re.sub(r'\s*/\s*', ' / ', text)

        # %
        text = re.sub(r'\s*%\s*', '%', text)

        # °C
        text = re.sub(r'\s*°\s*C', '°C', text)

        # normalize
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        pe = torch.zeros(5000, d_model)

        pos = torch.arange(0, 5000).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# =========================
# MODEL
# =========================
class Model(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, EMB_SIZE)

        self.pos = PositionalEncoding(EMB_SIZE)

        self.tr = nn.Transformer(
            d_model=EMB_SIZE,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(EMB_SIZE, vocab_size)

    def mask(self, n):
        return torch.triu(
            torch.ones(n, n),
            diagonal=1
        ).bool()

    def forward(self, src, tgt):

        # -------------------------
        # padding mask BEFORE emb
        # -------------------------
        src_pad_mask = (src == PAD)
        tgt_pad_mask = (tgt == PAD)

        # -------------------------
        # embedding
        # -------------------------
        src = self.pos(self.emb(src))
        tgt = self.pos(self.emb(tgt))

        # -------------------------
        # transformer
        # -------------------------
        out = self.tr(
            src,
            tgt,
            tgt_mask=self.mask(tgt.size(1)).to(src.device),

            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc(out)


# =========================
# INFERENCE CLASS
# =========================
class RainCaptionInference:

    def __init__(self, checkpoint_path):

        print("Loading checkpoint...")

        ckpt = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        # -------------------------
        # VOCAB
        # -------------------------
        self.vocab = Vocab()
        self.vocab.load(ckpt["vocab"])

        # -------------------------
        # MODEL
        # -------------------------
        self.model = Model(
            len(self.vocab.w2i)
        ).to(DEVICE)

        self.model.load_state_dict(
            ckpt["model"]
        )

        self.model.eval()

        print("Loaded model successfully")


    # =========================
    # GENERATE
    # =========================
    def generate(self, text, max_len=256,temperature=1.0,top_k=0):

        x = torch.tensor([
            [SOS]
            + self.vocab.encode(text)
            + [EOS]
        ]).to(DEVICE)

        y = torch.tensor([[SOS]]).to(DEVICE)

        self.model.eval()

        with torch.no_grad():

            for _ in range(max_len):

                out = self.model(x, y)

                # logits ของ token ล่าสุด
                logits = out[:, -1, :]

                # =========================
                # TEMPERATURE
                # =========================
                logits = logits / temperature

                # =========================
                # TOP-K SAMPLING
                # =========================
                if top_k > 0:

                    values, indices = torch.topk(
                        logits,
                        k=top_k,
                        dim=-1
                    )

                    probs = torch.softmax(values, dim=-1)

                    sampled_idx = torch.multinomial(
                        probs,
                        num_samples=1
                    )

                    next_id = indices[0, sampled_idx.item()].item()

                else:
                    # greedy
                    next_id = torch.argmax(
                        logits,
                        dim=-1
                    ).item()

                # append token
                y = torch.cat([
                    y,
                    torch.tensor([[next_id]]).to(DEVICE)
                ], dim=1)

                if next_id == EOS:
                    break

        return self.vocab.decode(
            y[0].tolist()
        )
def generate_caption(data,temperature=1.0,top_k=0):
    model = RainCaptionInference(
        "app/checkpoint-100.pth"
    )

    no_rain_max_index = 23
    coverate_max_index = 3
    rain_mid_max_index = 15
    rain_light_max_index = 15
    data = f"""
<xml>
no_rain:({random.randint(0, no_rain_max_index - 1)})
coverage:{data['coverage']}({random.randint(0, coverate_max_index - 1)})
large:{'|'.join(data['large_rain_district_name'])}
mid({random.randint(0, rain_mid_max_index - 1)}):{'|'.join(data['mid_rain_district_name'])}
light({random.randint(0, rain_light_max_index - 1)}):{'|'.join(data['light_rain_district_name'])}
temp:{data['tempurature']}
humid:{data['rh']}
day:{data['day']}
month:{data['month']}
year:{data['year']}
hour:{data['hour']}
minute:{data['minute']}
max_rain_name:{data['max_rain_name']}
max_rain_level_value:{data['max_rain_level_value']}
</xml>
"""
    print(data)
    return model.generate(data, 256 ,temperature, top_k)

# =========================
# TEST
# =========================
if __name__ == "__main__":

    model = RainCaptionInference(
        "app/checkpoint-100.pth"
    )

    test_input = """
<xml>
no_rain:(1)
coverage:75(0)
large:บางกะปิ|ลาดพร้าว
mid(1):ดินแดง
light(2):วัฒนา
temp:30
humid:80
day:12
month:5
year:2569
hour:14
minute:36
max_rain_name:บางกะปิ
max_rain_level_value:25.5
</xml>
"""

    result = model.generate(test_input)

    print("\n========== RESULT ==========")
    print(result)