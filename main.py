import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from typing import List, Tuple
import os

def load_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        return model, tokenizer
    except Exception as e:
        raise Exception(f"Error loading model and tokenizer: {str(e)}")

def quantize_model(model: AutoModelForSequenceClassification) -> AutoModelForSequenceClassification:
    """Quantize the model to int8."""
    try:
        # Quantize the model to int8
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        raise Exception(f"Error quantizing model: {str(e)}")

def evaluate_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: List[str], batch_size: int = 32) -> List[dict]:
    """Evaluate the model on a list of texts."""
    try:
        model.eval()
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_classes = torch.argmax(logits, dim=1).tolist()
            for cls in predicted_classes:
                results.append({"label": model.config.id2label[cls]})
        return results
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def save_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, path: str):
    """Save the quantized model and tokenizer."""
    try:
        os.makedirs(path, exist_ok=True)

        # Save the tokenizer and config using save_pretrained
        tokenizer.save_pretrained(path)
        model.config.save_pretrained(path)

        # Save the quantized model using torch.save
        torch.save(model, os.path.join(path, "quantized_model.pth"))

        print(f"Quantized model and tokenizer saved to {path}")
    except Exception as e:
        raise Exception(f"Error saving model and tokenizer: {str(e)}")

def load_quantized_model(path: str, device: torch.device) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the quantized model and tokenizer."""
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)

        # Load the quantized model using torch.load
        quantized_model = torch.load(os.path.join(path, "quantized_model.pth"), map_location=device)
        quantized_model.to(device)

        return quantized_model, tokenizer
    except Exception as e:
        raise Exception(f"Error loading quantized model and tokenizer: {str(e)}")

def test_quantized_model(original_model: AutoModelForSequenceClassification,
                         quantized_model: AutoModelForSequenceClassification,
                         tokenizer: AutoTokenizer, device: torch.device):
    """Test the quantized model against the original model."""
    try:
        # Load a small dataset for testing
        dataset = load_dataset("imdb", split="test[:100]")
        texts = dataset["text"]

        print("Evaluating original model...")
        original_results = evaluate_model(original_model, tokenizer, texts)

        print("Evaluating quantized model...")
        quantized_results = evaluate_model(quantized_model, tokenizer, texts)

        # Calculate accuracy
        correct = sum(1 for o, q in zip(original_results, quantized_results) if o['label'] == q['label'])
        accuracy = correct / len(texts)
        print(f"Accuracy of quantized model compared to original: {accuracy:.2%}")

        # Compare model sizes
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size / original_size):.2%}")
    except Exception as e:
        raise Exception(f"Error testing quantized model: {str(e)}")

def main():
    # Define configuration parameters directly
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = "./quantized_bert_model"
    batch_size = 32  # You can adjust this as needed

    device = torch.device("cpu")  # Dynamic quantization is typically for CPU

    try:
        # Load model and tokenizer
        print("Loading the original model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_name, device)

        # Quantize the model
        print("Quantizing the model...")
        quantized_model = quantize_model(model)

        # Test the quantized model
        print("Testing the quantized model...")
        test_quantized_model(model, quantized_model, tokenizer, device)

        # Save the quantized model
        print("Saving the quantized model...")
        save_model(quantized_model, tokenizer, output_dir)

        # Load the saved quantized model
        print("Loading the saved quantized model...")
        loaded_model, loaded_tokenizer = load_quantized_model(output_dir, device)

        # Test the loaded quantized model
        print("Testing the loaded quantized model...")
        test_quantized_model(model, loaded_model, loaded_tokenizer, device)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Execute the main function
main()
