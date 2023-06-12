import torch
import random
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding, emb_dim, en_hid_dim, dropout):
        super().__init__()
        
        self.embedding = embedding
        self.gru = nn.GRU(emb_dim, en_hid_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.gru(embedded)
        return output, hidden
        

class Decoder(nn.Module):
    def __init__(self, embedding, attention, output_dim, emb_dim, en_hid_dim, de_hid_dim, dropout):
        super().__init__()

        self.embedding = embedding
        self.attn = attention
        self.output_dim = output_dim
        
        self.gru = nn.GRU(en_hid_dim+emb_dim, de_hid_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(en_hid_dim+de_hid_dim+emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, en_output):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # input: (batch_size, 1)
        # embedded: (batch_size, 1, emb_dim)
        
        # use hidden state of the last layer
        weighted, _ = self.attn(hidden[-1].unsqueeze(0), en_output)
        # weighted: (batch_size, 1, en_hid_dim)
        
        gru_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        # output: (batch_size, 1, de_hid_dim*n_directions(1))
        # hidden: (n_layers(2)*n_directions(1), batch_size, de_hid_dim)

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction: (batch_size, output_dim)

        return prediction, hidden
        
        
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
        
        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, bos_id, max_length=30):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.bos_id = bos_id
        self.max_length = max_length
        self.output_dim = self.decoder.output_dim
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1] if trg is not None else self.max_length
        s2s_output = torch.zeros(batch_size, max_len, self.output_dim).to(self.device)
        
        en_output, hidden = self.encoder(src)
                
        # the first input to the decoder is the <bos> tokens,
        # for compatible with inference mode, create the first input by <bos> directly
        de_input = torch.tensor([self.bos_id] * batch_size).to(self.device)  # <bos> token id, (batch_size)
        
        for t in range(1, max_len):
            prediction, hidden = self.decoder(de_input, hidden, en_output)
            
            s2s_output[:, t] = prediction

            # pick the word with the highest probability from the predictions
            top_pred = prediction.argmax(1)
            
            # if teacher forcing, use actual next token as next input,
            # else use predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            de_input = trg[:, t] if teacher_force else top_pred

        return s2s_output

    def inference(self, src):
        """
        inference mode: trg is None, teacher_forcing_ratio=0
        """
        return self.forward(src, None, teacher_forcing_ratio=0)