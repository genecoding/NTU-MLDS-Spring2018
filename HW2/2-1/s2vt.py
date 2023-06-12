import torch
import random
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, *hidden):
        output, (hidden, cell) = self.lstm(self.dropout(input), *hidden)
        output = (output[:, :, :self.hid_dim] + output[:, :, self.hid_dim:]) / 2
        return output, hidden, cell
        
        
class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, *hidden):
        output, (hidden, cell) = self.lstm(self.dropout(input), *hidden)
        return output, hidden, cell


class Attention(nn.Module):
    """
    Additive attention
        
    Inputs:
        hidden (1, batch_size, de_hid_dim): hidden state of decoder
        en_output (batch_size, src_len, en_hid_dim): outputs of encoder
        
    Outputs:
        weighted (batch_size, 1, en_hid_dim): weighted outputs of encoder
        attention (batch_size, src_len)
    """
    def __init__(self, en_hid_dim, de_hid_dim):
        super().__init__()

        self.w_en = nn.Linear(en_hid_dim, de_hid_dim)
        self.w_de = nn.Linear(de_hid_dim, de_hid_dim)
        self.v = nn.Linear(de_hid_dim, 1, bias=False)
        
    def forward(self, hidden, en_output):
        src_len = en_output.shape[1]
        
        # repeat decoder hidden state src_len times
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        # hidden: (batch_size, src_len, de_hid_dim)

        energy = torch.tanh(self.w_en(en_output) + self.w_de(hidden)) 
        # energy: (batch_size, src_len, de_hid_dim)

        attention = F.softmax(self.v(energy).squeeze(2), dim=1)

        weighted = torch.bmm(attention.unsqueeze(1), en_output)

        return weighted, attention


class S2VT(nn.Module):
    """
    Implementation of Sequence to Sequence - Video to Text (S2VT)
    
    Args:
        device
        output_dim: size of the vocabulary
        bos_id: bos token id
        pad_id: pad token id
        max_length: max length when in inference mode
        use_attention
        
    Inputs:
        video_feats (batch_size, 80, 4096): features of videos
        captions (batch_size, cap_len): tokenized captions of videos
        teacher_forcing_ratio: the probability that teacher forcing will be used.
        
    Outputs:
        s2vt_output (batch_size, max_len, output_dim): caption predictions
        
    """
    def __init__(self, device, output_dim, bos_id, pad_id, max_length=40, use_attention=False):
        super().__init__()

        VIDEO_FEAT_DIM = 4096
        INPUT_DIM = 200 #500
        EN_HID_DIM = 1000
        DE_HID_DIM = 1000
        EMB_DIM = 200 #500
        DROPOUT = 0.5
        
        self.device = device 
        self.output_dim = output_dim
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.max_length = max_length
        self.use_attention = use_attention

        self.fc_proj = nn.Linear(VIDEO_FEAT_DIM, INPUT_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.embedding = nn.Embedding(output_dim, EMB_DIM, padding_idx=pad_id)
        self.encoder = Encoder(INPUT_DIM, EN_HID_DIM, DROPOUT)
        self.decoder = Decoder(EN_HID_DIM+EMB_DIM, DE_HID_DIM, DROPOUT)
        self.fc_out = nn.Linear(DE_HID_DIM, output_dim)

        if use_attention:
            self.attn = Attention(EN_HID_DIM, DE_HID_DIM)
            self.fc_out = nn.Linear(EN_HID_DIM+DE_HID_DIM, output_dim)

    def forward(self, video_feats, captions, teacher_forcing_ratio=0.5):
        batch_size, num_frames, _ = video_feats.shape
        max_len = captions.shape[1] if captions is not None else self.max_length
        s2vt_output = torch.zeros(batch_size, max_len, self.output_dim).to(self.device)
        
        # ================
        #  encoding stage 
        # ================
        # when in the encoding stage, feed all inputs to encoder and decoder at once.
        en_input = self.fc_proj(self.dropout(video_feats))
        en_output, en_hidden, en_cell = self.encoder(en_input)
        enstage_en_output = en_output  # encoder outputs in the encoding stage
        # en_input: (batch_size, num_frames(=80), input_dim)
        # en_output: (batch_size, num_frames, en_hid_dim)
        # en_hidden: (n_layers(1)*n_directions(2), batch_size, en_hid_dim)
        # en_cell: (n_layers(1)*n_directions(2), batch_size, en_hid_dim)

        padding = torch.tensor([[self.pad_id] * num_frames] * batch_size).to(self.device)
        # concatenate en_output with <pad> as the input to decoder
        de_input = torch.cat((en_output, self.embedding(padding)), dim=2)
        de_output, de_hidden, de_cell = self.decoder(de_input)
        # padding: (batch_size, num_frames)
        # de_input: (batch_size, num_frames, en_hid_dim+emb_dim)
        # de_output: (batch_size, num_frames, de_hid_dim*n_directions(1))
        # de_hidden: (n_layers(1)*n_directions(1), batch_size, de_hid_dim)
        # de_cell: (n_layers(1)*n_directions(1), batch_size, de_hid_dim)

        # ================
        #  decoding stage
        # ================
        # when in the decoding stage, 
        # (1) feed one input to encoder and decoder step by step,
        # (2) pass hidden and cell from the encoding stage to encoder and decoder.

        padding = torch.tensor([[self.pad_id]] * batch_size).to(self.device)  # (batch_size, 1)
        prev_word = torch.tensor([self.bos_id] * batch_size).to(self.device)  # <bos> token id, (batch_size)
        
        for t in range(max_len):
            en_input = self.embedding(padding)
            en_output, en_hidden, en_cell = self.encoder(en_input, (en_hidden, en_cell))
            # en_input: (batch_size, 1, emb_dim)

            # concatenate en_output with previous word id as the input to decoder
            de_input = torch.cat((en_output, self.embedding(prev_word.unsqueeze(1))), dim=2)
            de_output, de_hidden, de_cell = self.decoder(de_input, (de_hidden, de_cell))
            # de_output: (batch_size, 1, de_hid_dim*n_directions(1))

            if self.use_attention:
                weighted, _ = self.attn(de_hidden, enstage_en_output)
                de_output = torch.cat((de_output, weighted), dim=2)
                # de_output: (batch_size, 1, en_hid_dim+de_hid_dim)

            prediction = self.fc_out(de_output.squeeze(1))
            # prediction: (batch_size, output_dim)

            s2vt_output[:, t] = prediction

            # pick the word with the highest probability from the predictions
            top_pred = prediction.argmax(1)

            # if teacher forcing, use actual next token as next input,
            # else use predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            prev_word = captions[:, t] if teacher_force else top_pred

        return s2vt_output
    
    def inference(self, video_feats):
        """
        inference mode: captions is None, teacher_forcing_ratio=0
        """
        return self.forward(video_feats, None, teacher_forcing_ratio=0)
