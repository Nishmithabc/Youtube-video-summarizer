from transformers import PegasusTokenizer, PegasusForConditionalGeneration

MODEL_NAME = "google/pegasus-xsum"

tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)

# Save locally
tokenizer.save_pretrained("./models/pegasus-xsum")
model.save_pretrained("./models/pegasus-xsum")

print("Model saved successfully!")