import os
import pickle
from sentence_transformers import SentenceTransformer
from trafilatura import  extract
from googletrans import Translator
import glob
import random
import argparse


def html_to_text(html_content):
    # Use Trafilatura to extract main body text from HTML

    text_content = extract(html_content)
    print(text_content)
    return text_content

def translate_to_english(text, translator):
    # Use Google Translate to translate non-English content to English
    print("asdf")
    translated_text = translator.translate(text, dest='en').text
    print(translated_text)
    return translated_text

def generate_embedding(text_content, model_name):
    # Use Sentence Transformer to generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode([text_content])

    return embeddings

def sample_data(path):
    file_path = path + '/*.txt'

    all_files = glob.glob(file_path)

    num_files_to_select = 1000

    random_files = random.sample(all_files, num_files_to_select)

    return random_files
def save_to_pickle(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def prepare_embedding(path, transformer_model):
    # Extract content from HTML using Trafilatura
    data = []

    # Generate embedding using Sentence Transformer
    if transformer_model == "sbert":
        model_name = "sentence-transformers/bert-base-nli-mean-tokens"
    elif transformer_model == "xlm-roberta":
        model_name = "aditeyabaral/sentencetransformer-xlm-roberta-base"
    elif transformer_model == "electra":
        model_name = "sentence-transformers/electra-base-nli-mean-tokens"
    else:
        raise ValueError("Invalid transformer model. Available options: 'sbert', 'xlm-roberta', 'electra'")
    html_folder = sample_data(path)
    for fn in html_folder:

        try:
            with open(fn, "r", encoding="utf-8") as file:
                html_content = file.read()
                file.close()
        except Exception as error:
            continue

        text_content = html_to_text(html_content)

        if (text_content != None):

            try:
                if transformer_model != "xlm-roberta":
                    translator = Translator()
                    text_content = translate_to_english(text_content, translator)
                embeddings = generate_embedding(text_content, model_name)
                label = 1 if path == "Legitimate" else 0

                data.append((embeddings, label))
            except Exception as error:
                continue

    return data

def main():
    parser = argparse.ArgumentParser(description="Prepare embeddings from HTML content.")
    parser.add_argument("-transformer", choices=["sbert", "xlm-roberta", "electra"], required=True,
                        help="Select Sentence Transformer model")
    args = parser.parse_args()
    # Specify the path to the folder containing HTML files

    all_data = []
    legitimate_folder = "Legitimate"
    phishing_folder = "Phishing"
    embedding = prepare_embedding(legitimate_folder, args.transformer)
    all_data.append(embedding)

    embedding = prepare_embedding(phishing_folder, args.transformer)
    all_data.append(embedding)
    # Specify the path to save the embeddings
    output_folder = "embeddings"
    os.makedirs(output_folder, exist_ok=True)

    if args.transformer == "sbert":
        out_name = "embeddings-xlm-sbert.pkl"
    elif args.transformer == "xlm-roberta":
        out_name = "embeddings-xlm-roberta.pkl"
    elif args.transformer == "electra":
        out_name = "embeddings-xlm-electra.pkl"
    output_path = os.path.join(output_folder, out_name)
    save_to_pickle(all_data, output_path)


if __name__ == "__main__":
    main()
