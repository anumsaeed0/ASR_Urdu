import torch, os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# path/to/LoRA_finetuned_weights
lora_path = r"D:\Anum\ASR_Urdu\outputs\openai_large_v3_30Downgrade_LR"

# Model Ids
base_model_id = "openai/whisper-large-v3"
llm_model_id = "large-traversaal/Alif-1.0-8B-Instruct"

# Load base model
print("Loading OpenAI Base Whisper Model...")
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter weights
print("Load LoRA adapter weights...")
asr_model = PeftModel.from_pretrained(base_model, lora_path)

processor = AutoProcessor.from_pretrained(lora_path)

pipe_lora = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
)

# LLM Loader
print("Loading LLM tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

print("Loading LLM model...")
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

chatbot = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer
)

# Correction Function
def correct_urdu_text(text):
    prompt = f"""
    آپ ایک ماہر اردو پروف ریڈر ہیں۔ آپ کا کام صرف ہجے کی غلطیاں اور علامات درست کرنا ہے۔

    ###ہدایات###
    1. صرف ہجے کی غلطیاں درست کریں۔
    2. صرف علامات، وقفے اور نقطہ ویرگول درست کریں۔
    3. تمام الفاظ، جملے، اور افعال بالکل ویسے کے ویسے رہیں۔  
    - جیسے "کر دیا" یا "بتائیں گے" کو کبھی نہ بدلیں۔
    4. صرف درست شدہ اردو متن واپس کریں۔  
    - کوئی وضاحت، تبصرہ یا اضافی متن شامل نہ کریں۔
    5. جہاں ممکن ہو، اصل فارمیٹنگ اور لائن بریکس برقرار رکھیں۔
    6. ایسے بے معنی یا ٹوٹے ہوئے الفاظ کو سیاق و سباق کے مطابق درست اور بامعنی لفظ میں تبدیل کرنا  

    آپ کو متن کی اصل ساخت، اسلوب اور معنی کو برقرار رکھتے ہوئے صرف ضروری درستی کرنی ہے۔

    ###تفصیلی سوچ (Chain of Thoughts)###
    درج ذیل سخت عمل کریں:

    1. متن کو لفظ بہ لفظ اسکین کریں۔
    2. ہر لفظ کو الگ الگ دیکھیں:
    - اگر ہجے کی غلطی ہو یا سیاق و سباق میں بے معنی ہوں تو صرف وہی لفظ درست کریں۔
    - اگر لفظ درست ہو تو اسے ہرگز تبدیل نہ کریں۔
    3. سیاق و سباق کو سمجھ کر ممکنہ درست لفظ کا انتخاب کریں
     جملے کی ساخت، زمانہ، فعل یا اسلوب کو نہ چھیڑیں۔
    4. صرف punctuation (۔ ، ؛ ؟) درست کریں۔
    5. کسی بھی جملے کو دوبارہ نہ لکھیں۔
    6. کسی بھی لفظ کو مترادف یا بہتر شکل میں تبدیل نہ کریں۔
    7. آخر میں اصل متن اور درست شدہ متن کا اندرونی طور پر موازنہ کریں:
    - اگر کسی درست لفظ کو تبدیل کیا گیا ہو تو اسے واپس اصل شکل میں کر دیں۔
    8. صرف حتمی درست شدہ متن واپس کریں۔

    ###کیا نہ کریں###
    - جملے کو دوبارہ نہ لکھیں۔
    - فعل کی شکل نہ بدلیں۔
    - زمانہ تبدیل نہ کریں۔
    - وضاحت شامل نہ کریں۔
    - heading شامل نہ کریں۔
    - "درست شدہ متن:" دوبارہ نہ لکھیں۔
    - کوئی اضافی لائن شامل نہ کریں۔

    متن:
    {text}

    درست شدہ متن:
    """

    response = chatbot(
        prompt,
        max_new_tokens=300,
        do_sample=False
    )

    output = response[0]["generated_text"]

    # Extract corrected text
    if "درست شدہ متن:" in output:
        output = output.split("درست شدہ متن:")[-1].strip()

    return output

# ------------------------------
# Main Run
# ------------------------------
if __name__ == "__main__":
    line = "*" * 20

    print(f"{line}\nSpeech To Text")
    print(f"Type 'exit' to quit\n{line}")

    while True:
        audio_input = input("Enter Audio path :\n") 

        if audio_input.lower() == "exit":
            break

        if not os.path.exists(audio_input):
            print("Audio file not found.")
            continue
        
        print("Processing Audio File...")
        urdu_text = pipe_lora(audio_input, generate_kwargs={"language": "urdu"})['text']

        print(f"\nSpeech Detected in \'{audio_input}\':")
        print(correct_urdu_text(urdu_text))
        print("-" * 50)
