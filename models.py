from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, AlbertConfig
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_albert import AlbertPreTrainedModel, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import RobertaLMHead
from transformers import GPT2DoubleHeadsModel
from transformers.modeling_albert import AlbertEmbeddings, AlbertTransformer
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from loss import FocalLoss
from functions import gelu


# %%
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # fix 住 Linear 层以外的
        # for p in self.parameters():
        #     p.requires_grad = False

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
                              2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# %%
class BertForMultipleChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # fix 住 Linear 层以外的
        # for p in self.parameters():
        #     p.requires_grad = False

        # self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.view(
                -1,
                attention_mask.size(-1)) if attention_mask is not None else None
        else:
            attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1,
            token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(
            -1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        # print(outputs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        # pre_logits = self.pre_classifier(pooled_output)
        # logits = self.classifier(F.relu(pre_logits))

        logits = self.classifier(pooled_output)
        # print(logits)
        reshaped_logits = logits.view(-1, num_choices)
        # print(reshaped_logits.shape)

        outputs = (reshaped_logits,) + outputs[
                                       2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class RobertaForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    }
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dropout = nn.Dropout(0.4)
        # self.loss_fct = FocalLoss(3)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits, pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                                                2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            # loss = self.loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class RobertaForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size),
                                      masked_lm_labels.reshape(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


# %%
class RobertaForMultipleChoiceWithLM2(nn.Module):
    def __init__(self, tokenizer):
        super(RobertaForMultipleChoiceWithLM2, self).__init__()
        self.roberta_lm = RobertaForMaskedLM.from_pretrained(
            'pre_weights/roberta-large_model.bin', config=RobertaConfig.from_pretrained('roberta-large'))
        self.roberta = RobertaForMultipleChoice.from_pretrained(
            'pre_weights/roberta-large_model.bin', config=RobertaConfig.from_pretrained('roberta-large'))
        self.tokenizer = tokenizer
        self.lamda = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        output1 = self.roberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               labels=labels, position_ids=position_ids, head_mask=head_mask)
        input_ids_tmp = attention_mask_tmp = token_type_ids_tmp = position_ids_tmp = head_mask_tmp = None
        if input_ids is not None:
            input_ids_tmp = input_ids.reshape(-1, input_ids.shape[-1])
        if attention_mask is not None:
            attention_mask_tmp = attention_mask.reshape(-1, attention_mask.shape[-1])
            # for i in range(attention_mask_tmp.shape[0]):
            #     for j in range(attention_mask_tmp.shape[1]):
            #         if input_ids_tmp[i][j] != self.tokenizer.sep_token_id:
            #             attention_mask_tmp[i][j] = 0
            #         else:
            #             attention_mask_tmp[i][j] = 0
            #             break
        if token_type_ids is not None:
            token_type_ids_tmp = token_type_ids.reshape(-1, token_type_ids.shape[-1])
        if position_ids is not None:
            position_ids_tmp = position_ids.reshape(-1, position_ids.shape[-1])
        if head_mask is not None:
            head_mask_tmp = head_mask.reshape(-1, head_mask.shape[-1])

        output2 = self.roberta_lm(input_ids=input_ids_tmp, attention_mask=attention_mask_tmp,
                                  token_type_ids=token_type_ids_tmp,
                                  position_ids=position_ids_tmp, head_mask=head_mask_tmp,
                                  masked_lm_labels=input_ids_tmp)
        output2 = output2[0].reshape(-1, input_ids.shape[-1]).mean(dim=1).reshape(-1, input_ids.shape[-2])
        if labels is not None:
            loss2 = CrossEntropyLoss()(-output2, labels)
            output1 = (output1[0] + self.lamda * self.lamda * loss2,) + output1[1:]
        return output1


# %%
class RobertaForMultipleChoiceWithLM(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    }
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoiceWithLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.lamda 控制语言模型辅助程度
        self.lamda1 = nn.Parameter(torch.rand(1) * 2 + 1)
        self.lamda2 = nn.Parameter(torch.rand(1) * 2 + 1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        if True:
            '''语言模型 loss'''
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            '''
            masked_lm_loss 是一个长度为 (batch_size * num_choices * max_seq_length,) 的 Tensor
            需要将其转换为 (batch_size, num_choices, max_seq_length)，再对每一个 (max_seq_length) 求平均
            即每个问题三个选项分别计算出 loss，lm_loss.shape = (batch_size, num_choices)
            '''
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size), input_ids.reshape(-1))
            lm_loss = masked_lm_loss.view_as(input_ids).mean(dim=2)

            '''
            在 lm_loss 的基础上做一个分类问题，lm_loss 较低的认为更正确，因此取 -lm_loss
            '''
            loss_fct.reduction = 'mean'
            lm_loss_for_classification = loss_fct(-lm_loss, labels)

        outputs = (reshaped_logits,
                   pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                               2:]  # add hidden states and attention if they are here
        # outputs = (reshaped_logits / (2.0 * self.lamda1 * self.lamda1) + \
        #            -lm_loss / (2.0 * self.lamda2 * self.lamda2),
        #            pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
        #                                                                        2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            # 这里平衡一下两个 loss
            loss = (1.0 / (2.0 * self.lamda1 * self.lamda1) * loss) + (
                    1.0 / (2.0 * self.lamda2 * self.lamda2) * lm_loss_for_classification) + torch.log(
                self.lamda1 * self.lamda2)

            # low = max(loss, lm_loss_for_classification) + 1e-7
            # loss = loss * loss / low + \
            #        lm_loss_for_classification * lm_loss_for_classification / low

            # loss = loss + self.lamda1 * lm_loss_for_classification
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class AlbertModel(AlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    # load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
            while [2,3] correspond to the two inner groups of the second hidden layer.

            Any layer with in index other than [0,1,2,3] will result in an error.
            See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 这里是自己修改的，为了支持 attention 矩阵
        extended_attention_mask = attention_mask.unsqueeze(1)
        if attention_mask.dim() != 3:
            extended_attention_mask = extended_attention_mask.unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
                                                     1:]  # add hidden_states and attentions if they are here
        return outputs


# %%
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "albert"

    def __init__(self, config):
        super(AlbertForMultipleChoice, self).__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None

        outputs = self.albert(input_ids=flat_input_ids,
                              position_ids=flat_position_ids,
                              token_type_ids=flat_token_type_ids,
                              attention_mask=flat_attention_mask,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits, pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                                                2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class GPT2ForMultipleChoice(nn.Module):
    def __init__(self, pretrained_model_name_or_path, config):
        super(GPT2ForMultipleChoice, self).__init__()
        self.gpt2 = GPT2DoubleHeadsModel.from_pretrained(pretrained_model_name_or_path, config=config)

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        outputs = self.gpt2(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            mc_labels=labels)
        return outputs[0], outputs[2]  # mc loss, mc logits


# %%
class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        from torch_geometric.nn import GCNConv, GATConv, GINConv
        # nn1 = nn.Sequential(
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # nn2 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.conv1 = GINConv(nn1)
        # self.conv2 = GINConv(nn2)
        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT
        x = self.conv1(x, edge_index, edge_weight)
        x = gelu(x)
        x = F.dropout(x, training=self.training)
        logits = self.conv2(x, edge_index, edge_weight)
        logits = torch.stack(
            [logits[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)

        x = gelu(logits)
        x = self.fc1(x)

        outputs = (x.reshape(-1, data.num_graphs), logits,)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], data.y.reshape(-1, data.num_graphs).argmax(dim=1))
            outputs = (loss,) + outputs

        return outputs


# %%
class SOTA_goal_model(nn.Module):
    def __init__(self, args):
        super(SOTA_goal_model, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2

        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)

        from utils.attentionUtils import SelfAttention
        self.gcn = GCNNet()
        self.merge_fc1 = nn.Linear(roberta_config.hidden_size + 128, 512)
        self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        self.fc3 = nn.Linear(512 + roberta_config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])
        input_ids = torch.stack([j[1] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][1].shape).to(
            self.args['device'])
        attention_mask = torch.stack([j[2] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][2].shape).to(
            self.args['device'])
        token_type_ids = torch.stack([j[3] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][3].shape).to(self.args['device'])
        position_ids = torch.stack([j[4] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][4].shape).to(
            self.args['device'])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcn(i) for i in graph_features]

        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

        graph_features = [i[1].to('cpu') for i in x]

        loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        roberta_logits = roberta_outputs[2]

        loss = loss + torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])  # [4, 3, 64]
        del gcn_tmp_features, roberta_outputs  # 清理显存

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        merge_features = self.merge_fc1(
            torch.cat((roberta_logits, gcn_features), dim=2))
        merge_features = self.attn(merge_features)[0]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(torch.cat((roberta_logits, merge_features), dim=2)).view(-1, num_choices)
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)

        outputs = merge_features,

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss + loss_fct(outputs[0], labels)  # merge loss

            outputs = (loss,) + outputs
        return outputs


# %%
if __name__ == '__main__':
    from transformers import *

    # net = RobertaForMultipleChoiceWithLM2()

    import numpy as np
    from bidict import bidict
    from collections import defaultdict
    from utils.GraphUtils import GraphUtils
    from utils.getGraphUtils import get_datas, get_data_from_task_2, load_graph_pickle, merge_graph_by_downgrade, \
        encode_index

    data = np.array(get_datas(
        get_data_from_task_2(
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv'),
        get_data_from_task_2(
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv')
    ))
    graph = GraphUtils()
    graph.init()
    graph.merge_graph_by_downgrade()

    words_to_id = bidict()  # 将一个词映射为 id
    words_encode_idx = 0  # 实现上述两种操作的 idx
    conceptnet_numberbatch_en = dict()
    mp = defaultdict(set)

    mp_all, node_id_to_label_all, _, _ = load_graph_pickle('pre_weights/res.pickle')
    mp_all, node_id_to_label_all = merge_graph_by_downgrade(mp_all, node_id_to_label_all)
    # x, edge_index, edge_weight = encode_index(mp)
