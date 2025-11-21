from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import config
from src import model, utils


def continuous_translate(tokenizer: PreTrainedTokenizerFast, model: model.Transformer):
    print("--- English to Vietnamese Translator ---")
    print("Type 'exit' or 'quit' to stop the program.\n")

    while True:
        # 1. Get input from user
        english_sentence = input("Enter English text: ")

        # 2. Check for exit condition
        if english_sentence.strip().lower() in ["exit", "quit"]:
            print("Exiting program. Goodbye!")
            break

        # 3. Skip empty input
        if not english_sentence.strip():
            continue

        # 4. Perform translation
        try:
            # Translate from English (src='en') to Vietnamese (dest='vi')
            vietnamese_translation = utils.translate(
                model,
                tokenizer,
                english_sentence,
                config.DEVICE,
                config.MAX_SEQ_LEN,
                config.SOS_TOKEN_ID,
                config.EOS_TOKEN_ID,
                config.PAD_TOKEN_ID,
            )

            # 5. Print the result
            print(f"-> Vietnamese meaning: {vietnamese_translation}")
            print("-" * 30)  # Separator line

        except Exception as e:
            print(f"Error: Could not translate. Reason: {e}")


def main():

    try:
        tokenizer: PreTrainedTokenizerFast = utils.load_tokenizer(config.TOKENIZER_PATH)

        print("Loading model for inference...")
        inference_model = model.load_trained_model(
            config, config.MODEL_SAVE_PATH, config.DEVICE
        )
        print(f"Successfully loaded trained weights from {config.MODEL_SAVE_PATH}")

        continuous_translate(tokenizer, inference_model)

    except Exception as e:
        print(
            f"Warning: Could not load model. Using RANDOMLY initialized model. Error: {e}"
        )
        print("   (Translations will be gibberish)")


if __name__ == "__main__":
    main()
