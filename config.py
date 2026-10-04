from dataclasses import dataclass
import torch

@dataclass                #kind of works in place of constructor
class Config:
    num_layers:int=6
    d_model:int=256
    num_heads:int=4         #multihead attention
    feed_forward_size:int=1024
    dropout:float=0.1        #turn off some random numbers to prevent from overfitting

    dataset_name:str="CohleM/english-to-nepali"
    num_pairs:int=100000
    max_length:int=60          #maximum no of token in a sentence
    train_ratio:float=0.90
    validation_ratio:float=0.05

    vocab_size:int=8000
    pad_id:int=0          #padding value is 0
    unk_id:int=1          #unknown symbol
    bos_id:int=2          #beginning of sentence
    eos_id:int=3          #end of sentence

    batch_size:int=32
    num_epochs:int=20
    warmup_steps:int=1000        #Warmup means the learning rate starts near zero and grows slowly for the first 1000 steps, then slowly decreases afterwards
    label_smoothing:float=0.1    
    seed:int=42

    data_dir:str="data"
    checkpoint_dir:str="checkpoints"
    tokenizer_prefix:str="data/tokenizer"

    def __post_init__(self):     #@dataclass creates the __init__ for you, so you can't put your own checks inside it. __post_init__ is a special function that Python runs automatically right after __init__ finishes. It's the place for checks.
        assert self.d_model%self.num_heads==0,"d_model must be divisible by num_heads"


    @property                    #A property is a value that looks like a variable but is calculated when you read it
    def head_size(self):
        return self.d_model//self.num_heads  #// is floor division

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    config=Config()
    print(config)
    print("device:",config.device)
    print("head_size:",config.head_size)

