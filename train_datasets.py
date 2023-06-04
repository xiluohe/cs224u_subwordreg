import torch
import transformers
import datasets

class BPEDropoutTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, tokenizer_path, dataset_language=None, bpe_dropout_p=0.0, cache_dir=None, train=True):

        if dataset_language is not None:
            self.dset = datasets.load_dataset(dataset_path, dataset_language, cache_dir=cache_dir) 
        else: 
            self.dset = datasets.load_dataset(dataset_path, cache_dir=cache_dir) 
        
        self.train = train
        if self.train:
            bpe_dropout = bpe_dropout_p > 0.0
            alpha = 1.0 - bpe_dropout_p
            self.dataset = self.dset['train']
        else:
            bpe_dropout = False
            alpha = 1.0
            self.dataset = self.dset['test']
            
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, sp_model_kwargs={'enable_sampling': bpe_dropout, 'alpha': alpha}, cache_dir=cache_dir)

    def __getitem__(self, idx):

        tokenized_inputs = self.tokenizer(self.dataset["tokens"][idx], truncation=True, max_length=512, is_split_into_words=True)
        label_old = self.dataset["ner_tags"][idx]

        word_idx = 0
        label_new = []

        for id in tokenized_inputs['input_ids']:
            token = self.tokenizer.convert_ids_to_tokens(id)
            if token == "<s>":
                label_new.append(-100) #assign <s> to dummy token
            elif chr(9601) in token:
                label_new.append(label_old[word_idx]) #only label first token of a word
                word_idx += 1
            else:
                label_new.append(-100) #assign non-first token of word to dummy token
    
        data = tokenized_inputs['input_ids']
        label = label_new #+ [-100] * (512 - len(label_new))
        
        return {'input_ids': data, 'labels': label}

    def __len__(self):
        return len(self.dataset)