import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
import json
import re

sys.path.append('../')

import transformers
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup # WarmupLinearSchedule
from tqdm import tqdm
from utils import load_pkl_data, save_pkl_data, EarlyStopping
from models import BertForClozeBaseline
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold

workdir = os.getcwd()
project_dir = os.path.split(workdir)[0]
data_dir = os.path.join(project_dir, "data")
midata_dir = os.path.join(data_dir, "midata")

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

idiom_vocab = eval(open(data_dir + '/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

# Set device as `cuda` (GPU)
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()

class config:
    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 40
    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    EPOCHS = 5
    SEED = 42
    lr = 5e-5

    DO_TRAIN = True
    DO_TEST = True
    BERT_PATH = project_dir + "/pretrained_models/chinese_wwm_pytorch"
    # BERT_PATH = project_dir + "/pretrained_models/ernie_based_pretrained"

    TOKENIZER = BertTokenizer.from_pretrained(
        f"{BERT_PATH}/vocab.txt",
        lowercase=True
    )

    # train_features = os.path.join(midata_dir, "train_features.pkl")
    # valid_features = os.path.join(midata_dir, "valid_features.pkl")
    MODEL_SAVE_PATH = project_dir + f"/output/trained_{BERT_PATH.split('/')[-1]}_baseline"
    PREDICT_FILE_SAVE_PATH = project_dir + f"/output"


def check_dir():
    if not os.path.exists(midata_dir):
        os.makedirs(midata_dir)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    if not os.path.exists(config.PREDICT_FILE_SAVE_PATH):
        os.makedirs(config.PREDICT_FILE_SAVE_PATH)


class ClozeDataset:
    """
    Dataset which stores the tweets and returns them as processed features
    """

    def __init__(self, tokenizer, data_id, tag, text, candidate, groundTruth=None, max_len=128):
        self.data_id = data_id
        self.tag = tag
        self.text = text
        self.candidate = candidate
        self.groundTruth = groundTruth

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        feature_id = int(self.tag[item][6: -1])
        left_part, right_part = re.split(self.tag[item], self.text[item])
        left_ids = self.tokenizer.encode(left_part, add_special_tokens=False)
        right_ids = self.tokenizer.encode(right_part, add_special_tokens=False)

        half_length = int(self.max_len / 2)
        if len(left_ids) < half_length:  # cut at tail
            st = 0
            ed = min(len(left_ids) + 1 + len(right_ids), self.max_len - 2)
        elif len(right_ids) < half_length:  # cut at head
            ed = len(left_ids) + 1 + len(right_ids)
            st = max(0, ed - (self.max_len - 2))
        else:  # cut at both sides
            st = len(left_ids) + 3 - half_length
            ed = len(left_ids) + 1 + half_length

        text_ids = left_ids + [self.tokenizer.mask_token_id] + right_ids
        input_ids = [self.tokenizer.cls_token_id] + text_ids[st:ed] + [self.tokenizer.sep_token_id]

        position = input_ids.index(self.tokenizer.mask_token_id)

        token_type_ids = [0] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_masks = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [0] * (self.max_len - len(input_ids))

        label = self.candidate[item].index(self.groundTruth[item])
        idiom_ids = [idiom_vocab[each] for each in self.candidate[item]]

        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len
        assert len(token_type_ids) == self.max_len

        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            'data_id': torch.tensor(self.data_id[item], dtype=torch.long),
            'feature_id': torch.tensor(feature_id, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_masks': torch.tensor(input_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'idiom_ids': torch.tensor(idiom_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'position': torch.tensor(position, dtype=torch.long)
        }


def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as fin:
        data_id = 100000000
        for line in fin.readlines():
            cur_data = json.loads(line)
            groundTruth = cur_data["groundTruth"]
            candidates = cur_data["candidates"]
            content = cur_data["content"]
            realCount = cur_data["realCount"]
            for i in range(realCount):
                content = content.replace("#idiom#", f"#idiom{i+1}#", 1)
            tags = re.findall("#idiom\d+#", content)
            for tag in tags:
                tmp_context = content
                for other_tag in tags:
                    if other_tag != tag:
                        tmp_context = tmp_context.replace(other_tag, config.TOKENIZER.unk_token)
                data.append({
                    "data_id": data_id,
                    "tag": tag,
                    "text": tmp_context,
                    "candidate": candidates[i],
                    "groundTruth": groundTruth[i]
                })
            data_id += 1
    df_data = pd.DataFrame(data)
    return df_data


def convert_to_features(df_data, save_path, is_train=False):

    if is_train:
        if os.path.exists(save_path):
            dataset = []
            for root, dirs, files in os.walk(save_path):
                for file in files:
                    dataset.extend(load_pkl_data(os.path.join(root, file)))

        else:
            os.makedirs(save_path)
            dataset = ClozeDataset(
                tokenizer=config.TOKENIZER,
                data_id=df_data.data_id.values,
                tag=df_data.tag.values,
                text=df_data.text.values,
                candidate=df_data.candidate.values,
                groundTruth=df_data.groundTruth.values,
                max_len=config.MAX_LEN
            )
            datas = []
            data = []
            batch_id = 1
            tk = tqdm(dataset, total=len(dataset))
            for bi, item in enumerate(tk):
                data.append(item)
                if len(data) == 50000 or bi == len(dataset) - 1:
                    path = save_path + f"/train_features_{batch_id}.pkl"
                    save_pkl_data(data, path)
                    batch_id += 1
                    datas.extend(data)
                    data = []
            dataset = datas
    else:
        if os.path.exists(save_path):
            dataset = load_pkl_data(save_path)
        else:

            dataset = ClozeDataset(
                tokenizer=config.TOKENIZER,
                data_id=df_data.data_id.values,
                tag=df_data.tag.values,
                text=df_data.text.values,
                candidate=df_data.candidate.values,
                groundTruth=df_data.groundTruth.values,
                max_len=config.MAX_LEN
            )
            tk = tqdm(dataset, total=len(dataset))
            dataset = [item for item in tk]
            save_pkl_data(dataset, save_path)
    return dataset


def run_one_step(batch, model, device):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    input_mask = batch["input_masks"].to(device)
    position = batch["position"].to(device)
    idiom_ids = batch["idiom_ids"].to(device)

    # Use ids, masks, and token types as input to the model
    # Predict logits for each of the input tokens for each batch
    logits = model(
        input_ids,
        input_mask,
        token_type_ids=token_type_ids,
        idiom_ids=idiom_ids,
        positions=position,
    )  #

    return logits


def train_fn(data_loader, model, optimizer, device, epoch, scheduler=None):
    """
    Trains the bert model on the twitter data
    """
    # Set model to training mode (dropout + sampled batch norm is activated)
    model.train()
    # Set tqdm to add loading screen and set the length
    tk0 = tqdm(data_loader, total=len(data_loader))
    # Train the model on each batch
    # Reset gradients

    for bi, batch in enumerate(tk0):
        model.zero_grad()
        logits = run_one_step(batch, model, device)
        label = batch["label"].to(device)
        # # Calculate batch loss based on CrossEntropy
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, label)
        # Calculate gradients based on loss

        loss.backward()
        optimizer.step()   #更新模型参数
        optimizer.zero_grad()
        # Update scheduler
        #   # 更新learning rate
        # print(f"lr={scheduler.get_lr()}")
        scheduler.step()

        # Calculate the acc score based on the predictions for this batch
        outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        pred_label = np.argmax(outputs, axis=1)
        acc = accuracy_score(label.cpu().numpy(), pred_label)
        # Print the average loss and jaccard score at the end of each batch
        tk0.set_postfix(epoch=epoch, acc=acc, loss=loss.item())



def eval_fn(valid_data_loader, model, device):
    """
    Evaluation function to predict on the test set
    """
    # Set model to evaluation mode
    model.eval()
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))
        # Make predictions and calculate loss / acc, f1 score for each batch
        for bi, batch in enumerate(tk0):

            # Use ids, masks, and token types as input to the model
            # Predict logits for each of the input tokens for each batch
            logits = run_one_step(batch, model, device)
            label = batch["label"].to(device)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, label)

            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=1)
            acc = accuracy_score(label.cpu().numpy(), pred_label)
            pred_labels.extend(pred_label.tolist())
            true_labels.extend(label.cpu().numpy().tolist())
            # Print the running average loss and acc
            tk0.set_postfix(loss=loss.item(), acc=acc)

    total_acc = accuracy_score(true_labels, pred_labels)
    return total_acc


def train():
    """
    Train model for a speciied fold
    """
    df_train = read_data(os.path.join(data_dir, "train_data.txt"))
    train_dataset = convert_to_features(df_train, midata_dir + f"/train_features_{config.MAX_LEN}", is_train=True)
    # Instantiate DataLoader with `train_dataset`
    # This is a generator that yields the dataset in batches
    logger.info(f"the number of train example:{len(train_dataset)}")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=0
    )
    df_dev = read_data(os.path.join(data_dir, "dev_data.txt"))
    valid_dataset = convert_to_features(df_dev, midata_dir + f"/dev_features_{config.MAX_LEN}.pkl")
    logger.info(f"the number of dev example:{len(valid_dataset)}")
    # Instantiate DataLoader with `valid_dataset`
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0
    )

    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
    model_config.output_hidden_states = True
    # Instantiate our model with `model_config`
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.BERT_PATH,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    # Move the model to the GPU
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Calculate the number of training steps
    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    # Get the list of named parameters
    param_optimizer = list(model.named_parameters())
    # Specify parameters where weight decay shouldn't be applied
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # Instantiate AdamW optimizer with our two sets of parameters, and a learning rate of 3e-5
    optimizer = AdamW(optimizer_parameters, lr=config.lr)
    # Create a scheduler to set the learning rate at each training step
    # "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
    # Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_train_steps,

    )
    # Apply early stopping with patience of 2
    # This means to stop training new epochs when 2 rounds have passed without any improvement
    es = EarlyStopping(patience=2, mode="max", delta=0.00001)
    # I'm training only for 3 epochs even though I specified 5!!!
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, epoch + 1, scheduler)
        eval_acc = eval_fn(valid_data_loader, model, device)
        logger.info(f"epoch: {epoch + 1}, acc = {eval_acc}")
        es(epoch, eval_acc, model, model_path=config.MODEL_SAVE_PATH)
        if es.early_stop:
            print("********** Early stopping ********")
            break


def predict():
    df_test = read_data(os.path.join(data_dir, "test_data.txt"))
    test_dataset = convert_to_features(df_test, midata_dir + f"/test_features.pkl")
    # Instantiate DataLoader with `test_dataset`
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.TEST_BATCH_SIZE
    )
    # Load pretrained BERT (bert-base-uncased)
    model_path = config.MODEL_SAVE_PATH
    model_config = transformers.BertConfig.from_pretrained(model_path)
    model_path = config.MODEL_SAVE_PATH
    # Instantiate our model with `model_config`
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config, idiom_num=len(idiom_vocab))
    # # Load each of the five trained models and move to GPU
    trained_model_path = os.path.join(model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    pred_labels = []
    true_labels = []
    # Turn of gradient calculations
    with torch.no_grad():
        tk0 = tqdm(test_dataloader, total=len(test_dataloader))
        # Predict the span containing the sentiment for each batch
        for bi, batch in enumerate(tk0):
            # Predict logits
            logits = run_one_step(batch, model, device)

            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=1).tolist()
            # pred_logits = np.append(pred_logits, logits.view(-1).cpu().detach().numpy())
            pred_labels.extend(pred_label)
            true_labels.extend(batch["label"].numpy())
    # return pred_logits
    test_acc = accuracy_score(true_labels, pred_labels)
    logger.info(f"test acc: {test_acc}")
    return pred_labels


def seed_set(seed):
    '''
    set random seed of cpu and gpu
    :param seed:
    :param n_gpu:
    :return:
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run():
    check_dir()
    seed_set(config.SEED)
    if config.DO_TRAIN:
        train()
    if config.DO_TEST:
        predict()


if __name__ == '__main__':
    run()

