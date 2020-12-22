import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertForClozeBaseline(BertPreTrainedModel):

    def __init__(self, conf, idiom_num, pretrained_model_path=None):
        super(BertForClozeBaseline, self).__init__(conf)
        # 768 is the dimensionality of bert-base-uncased's hidden representations
        # Load the pretrained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_path, config=conf)
        self.idiom_embedding = nn.Embedding(idiom_num, conf.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conf.hidden_size, 1),
            # nn.Sigmoid()
        )
        

        torch.nn.init.normal_(self.classifier[1].weight, std=0.05)

    def forward(self, input_ids, attention_mask, token_type_ids, idiom_ids, positions):
        # input_ids [batch, max_seq_length]  encoded_layer [batch, max_seq_length, hidden_state]
        sequence_outputs, pooled_outputs, encoder_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        blank_states = sequence_outputs[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        encoded_idiom = self.idiom_embedding(idiom_ids)  # [batch, 10， hidden_state]

        multiply_result = torch.einsum('abc,ac->abc', encoded_idiom, blank_states)  # [batch, 10， hidden_state]

        logits = self.classifier(multiply_result)  # [batch, 10, 1]
        logits = logits.view(-1, idiom_ids.shape[-1])  # [batch, 10]

        return logits