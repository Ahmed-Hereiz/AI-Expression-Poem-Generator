import torch
import torch.nn as nn
from torchvision import models
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    pipeline,
    TextGenerationPipeline,
    GPT2LMHeadModel,
    AutoTokenizer
)

def load_models():
    # Load Models we trained from it's directories :
    
    poem_generation_model_path = "Poem_Generation/Tuned_model"
    poem_generation_model = GPT2LMHeadModel.from_pretrained(poem_generation_model_path)
    poem_generation_tokenizer = AutoTokenizer.from_pretrained(poem_generation_model_path)
    loaded_poem_generator = TextGenerationPipeline(model=poem_generation_model, tokenizer=poem_generation_tokenizer)
    
    person_detect_path = 'Person_Detect_Model/detect-person.pt'
    person_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=person_detect_path)
    person_detect_model.to("cpu")
    class_names = ['person', 'person']
    
    emotion_classify_model_path = "Face_Emotion_Classify/custom_model.pth"
    emotion_classify_model = torch.load(emotion_classify_model_path)
    
    
    return loaded_poem_generator, person_detect_model, emotion_classify_model, class_names
















