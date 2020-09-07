import torch
import torch.nn as nn

from openvqa.models.vqabert.adapter import Adapter
from openvqa.models.vqabert.vqa_bert import VQA_BERT, BERTPooler
from openvqa.ops.layer_norm import LayerNorm
from openvqa.utils.make_mask import make_mask


# -------------------------
# ---- Main VQA BERT Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, token_to_ix):
        super(Net, self).__init__()
        self.__C = __C
        self.token_to_ix = token_to_ix
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_proj = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE)
        self.token_proj = nn.Linear(__C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE)

        self.adapter = Adapter(__C)
        self.backbone = VQA_BERT(__C)

        self.text_pooler = BERTPooler(__C)
        self.img_pooler = BERTPooler(__C)

        # Classification layers
        self.dense = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.activation = nn.Tanh()
        self.layer_norm = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.cls = nn.Linear(__C.HIDDEN_SIZE, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        batch_size = ques_ix.shape[0]
        device = ques_ix.device

        # Pre-process Language Feature
        text_feat_mask = make_mask(ques_ix.unsqueeze(2))
        text_feat = self.embedding(ques_ix)
        text_feat, _ = self.lstm(text_feat)
        text_feat = self.lstm_proj(text_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        cls_token = torch.tensor(self.token_to_ix['CLS'], device=device).repeat(batch_size, 1)
        cls_token = self.embedding(cls_token)
        cls_token = self.token_proj(cls_token)

        img_token = torch.tensor(self.token_to_ix['IMG'], device=device).repeat(batch_size, 1)
        img_token = self.embedding(img_token)
        img_token = self.token_proj(img_token)

        text_feat = torch.cat([cls_token, text_feat], dim=1)
        img_feat = torch.cat([img_token, img_feat], dim=1)

        img_mask = make_mask(img_token)
        cls_mask = make_mask(cls_token)
        text_feat_mask = torch.cat([cls_mask, text_feat_mask], dim=-1)
        img_feat_mask = torch.cat([img_mask, img_feat_mask], dim=-1)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            text_feat,
            img_feat,
            text_feat_mask,
            img_feat_mask
        )

        text_pool = self.text_pooler(lang_feat)
        img_pool = self.img_pooler(img_feat)

        # Classification layers
        pooled_output = self.dropout(text_pool * img_pool)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        output = self.cls(pooled_output)

        return output
