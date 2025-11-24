import streamlit as st
import time
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import config
import model
import utils


# ==========================================
# 1. ASSUMPTIONS
# ==========================================


@st.cache_resource
def load_artifacts():
    tokenizer: PreTrainedTokenizerFast = None
    transformer_model: model.Transformer = None

    try:
        tok_path = hf_hub_download(
            repo_id=config.REPO_ID, filename="iwslt_en-vi_tokenizer_32k.json"
        )
        tokenizer = utils.load_tokenizer(tok_path)

        print("Loading model for inference...")
        transformer_model = model.load_trained_model(
            config, config.MODEL_SAVE_PATH, config.DEVICE
        )

    except Exception as e:
        print(
            f"Warning: Could not load model. Using RANDOMLY initialized model. Error: {e}"
        )
        print("   (Translations will be gibberish)")

    return transformer_model, tokenizer


# ==========================================
# 2. UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="En-Vi Translator | AttentionIsAllYouBuild",
    page_icon="🤖",
    layout="centered",
    # layout="wide",
)

# Customize CSS to create beautiful interface
st.markdown(
    """
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 10px;
    }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #FF4B4B;
    }
    .source-text {
        color: #666;
        font-style: italic;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .translated-text {
        color: #333;
        font-size: 20px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MAIN APP LAYOUT
# ==========================================

# Header
st.title("🤖 AI Translator: English → Vietnamese")
st.markdown("### Project: *Attention Is All You Build*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Thông tin Model")
    st.info(
        """
        Đây là mô hình **Transformer (Encoder-Decoder)** được xây dựng "from scratch" bằng PyTorch.

        - **Kiến trúc**: Pre-LN Transformer
        - **Tokenizer**: BPE (32k vocab)
        - **Inference**: Greedy
        """
    )
    st.write("Created by [Lê Hồ Long]")

# Input Area
input_text = st.text_area(
    label="Nhập câu tiếng Anh:",
    placeholder="Example: Artificial intelligence is transforming the world...",
    height=150,
)

# ==========================================
# 4. INFERENCE LOGIC
# ==========================================

# Translation Button
if st.button("Dịch sang Tiếng Việt (Translate)"):
    if not input_text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung cần dịch!")
    else:
        # Display spinner while model is running
        with st.spinner("Wait a second... AI is thinking 🧠"):
            try:
                # Measure inference time
                start_time = time.time()

                # --- Call translate function ---
                transformer_model, tokenizer = load_artifacts()

                if utils and transformer_model and tokenizer:
                    translation = utils.translate(
                        transformer_model,
                        tokenizer,
                        sentence_en=input_text,
                        device=config.DEVICE,
                        max_len=config.MAX_SEQ_LEN,
                        sos_token_id=config.SOS_TOKEN_ID,
                        eos_token_id=config.EOS_TOKEN_ID,
                        pad_token_id=config.PAD_TOKEN_ID,
                    )

                else:
                    # Mockup output
                    time.sleep(1)  # Simulate latency
                    translation = "[DEMO OUTPUT] Hệ thống chưa load model thực tế. Đây là kết quả mẫu."

                end_time = time.time()
                inference_time = end_time - start_time

                # --- Display Result ---
                st.success(f"✅ Hoàn tất trong {inference_time:.2f}s")

                st.markdown("### Kết quả:")
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="source-text">Original: {input_text}</div>
                        <div class="translated-text">{translation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi trong quá trình dịch: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by PyTorch & Streamlit")
