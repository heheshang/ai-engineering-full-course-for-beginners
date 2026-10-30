# pip install transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

tokenizer.src_lang = "eng_Latn"
inputs = tokenizer(
    "My puppy is adorable, Your kitten is cute. Her panda is friendly. His llama is thoughtful. We all have nice pets!",
    return_tensors="pt",
)
translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("spa_Latn"),
    max_length=512,
)
result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
print(result)
