import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    pipeline,
    Trainer,
    TrainingArguments
)

model_ckpt = "papluca/xlm-roberta-base-language-detection"
device = "cuda"

class PreprocessFiles:
    """
    A class to preprocess data and perform various operations.

    Attributes:
        data (pandas.DataFrame): The loaded data.
        pipe: Text classification pipeline for language detection.

    Methods:
        load_json(file_path):
            Load JSON data from the specified file path.
        
        create_text_classification_pipeline(model_ckpt):
            Create a text classification pipeline for language detection.

        detect_language():
            Detect the language of poems and remove non-English ones.
        
        preprocess_poems(sup_token, data):
            Preprocess poems by adding special tokens and processing text.
        
        choose_cols(data, col):
            Select a specific column from the data.
        
        merge_dataframes(dataframes_list):
            Merge a list of dataframes and shuffle the rows.
        
        split_data(data):
            Split data into training and testing sets.
        
        save_to_txt(data, output_file):
            Save processed data to a text file.
    """

    def __init__(self):
        self.data = None
        self.pipe = None

    def load_json(self, file_path):
        """
        Load JSON data from the specified file path.

        Args:
            file_path (str): Path to the JSON file.
        """
        with open(file_path, 'r') as f:
            self.data = pd.read_json(f)

    def create_text_classification_pipeline(self, model_ckpt):
        """
        Create a text classification pipeline for language detection.

        Args:
            model_ckpt (str): Path or identifier of the pre-trained model checkpoint.
        """
        self.pipe = pipeline("text-classification", model=model_ckpt, device=device)

    def detect_language(self, sup_token):
        """
        Detect the language of poems and remove non-English ones.
        """
        print(f"Processing {sup_token} poems...")
        idx = 0
        en_poems = []
        text_column = self.data["poem"]
        for text in text_column:
            text = text[:512]
            result = self.pipe(text)
            detected_language = result[0]['label']
            if detected_language == 'en':
                en_poems.append('en')
            else:
                en_poems.append('no')

            idx += 1
            if idx % 1000 == 0:
                print(f"Processed {idx} poem...")

        print("Detection of all dataframes is done.")
        print(f"Processed all {sup_token} poems...")
    
        self.data["en_poems"] = en_poems
        self.data = self.data[self.data["en_poems"] == 'en']
        self.data.drop(columns=["en_poems"], inplace=True)
        return self.data

    def preprocess_poems(self, sup_token, data):
        """
        Preprocess poems by adding special tokens and processing text.

        Args:
            sup_token (str): Special token to be added.
            data (pandas.DataFrame): Dataframe containing poems.
        
        Returns:
            pandas.DataFrame: Dataframe with processed poems.
        """
        data['processed_poem'] = data['poem'].apply(lambda poem: "<BOS> " + str(sup_token) + " " + " ".join(poem.split()) + " <EOS>")
        return data
    
    def choose_cols(self, data, col):
        """
        Select a specific column from the data.

        Args:
            data (pandas.DataFrame): Dataframe containing the desired column.
            col (str): Column name to select.
        
        Returns:
            pandas.DataFrame: Dataframe with only the selected column.
        """
        data = data[col]
        return pd.DataFrame(data)

    def merge_dataframes(self, dataframes_list):
        """
        Merge a list of dataframes and shuffle the rows.

        Args:
            dataframes_list (list): List of dataframes to merge.
        
        Returns:
            pandas.DataFrame: Merged and shuffled dataframe.
        """
        merged_data = pd.concat(dataframes_list, ignore_index=True)
        shuffled_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)
        return shuffled_data
    
    def split_data(self, data):
        """
        Split data into training and testing sets.

        Args:
            data (pandas.DataFrame): Dataframe to split.
        
        Returns:
            pandas.DataFrame: Training and testing dataframes.
        """
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        return train_data, test_data

    def save_to_txt(self, data, output_file):
        """
        Save processed data to a text file.

        Args:
            data (pandas.DataFrame): Dataframe containing processed data.
            output_file (str): Path to the output text file.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for processed_poem in data['processed_poem']:
                f.write(processed_poem + '\n')
                
                
                
def main():
    preprocessor = PreprocessFiles()

    json_files_dir = ["/content/AutoCompose/data/anger.json",
                      "/content/AutoCompose/data/disgust.json",
                      "/content/AutoCompose/data/fear.json",
                      "/content/AutoCompose/data/joy.json",
                      "/content/AutoCompose/data/sadness.json",
                      "/content/AutoCompose/data/surprise.json",
                      "/content/AutoCompose/data/trust.json",
                      "/content/AutoCompose/data/anticipation.json"
                     ]

    sup_tokens = ["<anger>","<disgust>","<fear>","<happy>","<sad>","<surprise>","<neutral>","<neutral>"]

    preprocessor.create_text_classification_pipeline(model_ckpt)
    dataframes = {}

    for json_file, sup_token in zip(json_files_dir, sup_tokens):
        preprocessor.load_json(json_file)
        data = preprocessor.detect_language(sup_token)
        data = preprocessor.preprocess_poems(sup_token, data)
        data = preprocessor.choose_cols(data=data, col="processed_poem")
        train_data, test_data = preprocessor.split_data(data=data)
        dataframes[sup_token] = {"train": train_data, "test": test_data}

    train_dataframes = [dfs["train"] for dfs in dataframes.values()]
    test_dataframes = [dfs["test"] for dfs in dataframes.values()]

    full_train_data = preprocessor.merge_dataframes(train_dataframes)
    full_test_data = preprocessor.merge_dataframes(test_dataframes)

    preprocessor.save_to_txt(data=full_train_data, output_file="train_data.txt")
    preprocessor.save_to_txt(data=full_test_data, output_file="test_data.txt")

    
if __name__ == "__main__":
    main()