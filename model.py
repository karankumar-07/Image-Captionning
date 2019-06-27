
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        #lstm layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        #linear layer
        self.vocab = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions[:,:-1]) 
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        lstm_out, _ = self.lstm(inputs)   
        outputs = self.vocab(lstm_out)     
        return outputs  
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
    
        c = []
        # for h0 - hidden state, c0- cell state
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))

        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.vocab(lstm_out)        
            outputs = outputs.squeeze(1)                 
            w_id  = outputs.argmax(dim=1)              
            c.append(w_id.item())
            #embedding
            inputs = self.word_embeddings(w_id.unsqueeze(0))  
          
        return c