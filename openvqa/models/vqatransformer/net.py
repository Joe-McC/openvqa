import torch
import torch.nn as nn
from transformers import BertModel

from openvqa.models.vqatransformer.adapter import Adapter
from openvqa.models.vqatransformer.transformer import Transformer, TransformerPooler
from openvqa.ops.layer_norm import LayerNorm
from openvqa.utils.make_mask import make_mask


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, token_to_ix):
        super(Net, self).__init__()
        self.__C = __C

        self.token_to_ix = token_to_ix

        self.word_embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        # self.text_position_embeddings = nn.Embedding(43, __C.HIDDEN_SIZE)

        # Loading the GloVe embedding weights
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # Segment embedding
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.segment_embedding = bert.embeddings.token_type_embeddings
        self.segment_proj = nn.Linear(768, __C.HIDDEN_SIZE)
        # self.segment_embedding = nn.Embedding(2, __C.HIDDEN_SIZE)

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_proj = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE)
        self.cls_project = nn.Linear(__C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE)

        self.img_encoder = Adapter(__C)
        self.img_pos_emb = nn.Linear(2, __C.HIDDEN_SIZE)

        self.transformer = Transformer(__C)

        self.layer_norm = LayerNorm(__C.HIDDEN_SIZE)
        self.embbeding_dropout = nn.Dropout(__C.DROPOUT_R)

        # Classification layers
        self.pooler = TransformerPooler(__C)
        self.cls_dropout = nn.Dropout(__C.DROPOUT_R)
        self.classifier = nn.Linear(__C.HIDDEN_SIZE, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        batch_size = ques_ix.shape[0]
        device = ques_ix.device

        # create text feature
        text_feat_mask = make_mask(ques_ix.unsqueeze(2))
        text_feat = self.word_embedding(ques_ix)
        text_feat, _ = self.lstm(text_feat)
        text_feat = self.lstm_proj(text_feat)

        # seq_length = text_feat.size()[1]
        # text_position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # text_position_ids = text_position_ids.unsqueeze(0).expand(ques_ix.size())
        # text_position_embeddings = self.text_position_embeddings(text_position_ids)

        # create text segment embedding
        text_seg_ids = torch.zeros(text_feat.size()[:-1], dtype=torch.long, device=device)
        text_seg_embedding = self.segment_embedding(text_seg_ids)
        text_seg_embedding = self.segment_proj(text_seg_embedding)

        # text embedding
        text_feat = text_feat + text_seg_embedding #+ text_position_embeddings

        # image features and mask
        img_feat, img_feat_mask = self.img_encoder(frcn_feat, grid_feat, bbox_feat)

        # create image segment embedding
        img_seg_ids = torch.ones(img_feat.size()[:-1], dtype=torch.long, device=device)
        img_seg_embedding = self.segment_embedding(img_seg_ids)
        img_seg_embedding = self.segment_proj(img_seg_embedding)

        # image position embeddign
        width = 14
        height = 14
        img_pos = torch.meshgrid([torch.arange(width, dtype=torch.float, device=device),
                                  torch.arange(height, dtype=torch.float, device=device)])
        img_pos = torch.stack([img_pos[1], img_pos[0]], dim=-1).view(width * height, 2).unsqueeze(0).expand(batch_size,
                                                                                                            width * height,
                                                                                                            2)
        img_pos_emb = self.img_pos_emb(img_pos)

        # image embedding
        img_feat = img_feat + img_seg_embedding + img_pos_emb

        # CLS embedding
        cls_token = torch.tensor(self.token_to_ix['CLS'], device=device).repeat(batch_size, 1)
        cls_token = self.word_embedding(cls_token)
        cls_token = self.cls_project(cls_token)

        # prepare input embedding for transformer
        embeddings = torch.cat([cls_token, img_feat, text_feat], dim=1)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.embbeding_dropout(embeddings)

        # prepare mask for self attention
        cls_mask = make_mask(cls_token)
        attention_mask = torch.cat([cls_mask, img_feat_mask, text_feat_mask], dim=-1)

        # Backbone Framework
        feat = self.transformer(embeddings, attention_mask)

        # Classification layers
        cls_output = self.pooler(feat)
        cls_output = self.cls_dropout(cls_output)
        output = self.classifier(cls_output)

        return output
