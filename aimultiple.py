import time
import re
import streamlit as st
import datetime
from io import BytesIO
from PIL import Image
import google.generativeai as genai

TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image-preview"
MAX_PROMPT_CHARS = 3000
MAX_RETRIES = 5
INITIAL_DELAY = 3

st.set_page_config(page_title="Multi-AI App (robust)", layout="wide")

api_key = st.secrets.get("GEMINI_apikey")
if not api_key:
    st.error("API key missing. Put your key in `.streamlit/secrets.toml` as: GEMINI_apikey = \"YOUR_KEY\"")
    st.stop()

genai.configure(api_key=api_key)

if "history" not in st.session_state:
    st.session_state.history = []

page = st.sidebar.radio("Select Feature", ["Teaching", "Image Generator", "Math Solver", "History"], index=0)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    ClientError = genai.errors.ClientError
except Exception:
    ClientError = Exception

def extract_retry_delay_from_exception(exc):
    try:
        for arg in getattr(exc, "args", []):
            if isinstance(arg, dict):
                err = arg.get("error") or arg
                details = err.get("details", []) if isinstance(err, dict) else []
                for d in details:
                    if isinstance(d, dict) and d.get("@type", "").endswith("RetryInfo"):
                        retry = d.get("retryDelay")
                        if isinstance(retry, str):
                            m = re.search(r"(\d+(?:\.\d+)?)s", retry)
                            if m:
                                return int(float(m.group(1)))
                            m2 = re.search(r"(\d+):(\d+):(\d+)", retry)
                            if m2:
                                h, mm, s = map(int, m2.groups())
                                return h * 3600 + mm * 60 + s
        s = str(exc)
        m = re.search(r"retryDelay.? ?:? ?\"?(\d+(?:\.\d+)?)s\"?", s)
        if m:
            return int(float(m.group(1)))
    except Exception:
        pass
    return None

def generate_with_retry(model: str, contents, is_image: bool = False):
    delay = INITIAL_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = contents if not is_image else [contents]
            gm = genai.GenerativeModel(model)
            try:
                resp = gm.generate_content(payload)
            except TypeError:
                resp = gm.generate_content(contents=payload)
            return resp
        except ClientError as e:
            status = getattr(e, "status_code", None)
            msg = str(e).upper()
            retry_delay = extract_retry_delay_from_exception(e)
            if status == 429 or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg:
                wait = retry_delay if retry_delay is not None else delay
                st.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s before retrying...")
                time.sleep(wait)
                delay *= 2
                continue
            raise
        except Exception as e:
            st.error(f"Generation failed: {e}")
            raise
    raise Exception(f"Failed after {MAX_RETRIES} retries due to rate limits.")

def run_teaching_assistant():
    st.header("📘 AI Teaching Assistant")
    prompt = st.text_area("Ask your question (keep it short to save quota):", height=200, max_chars=MAX_PROMPT_CHARS, placeholder="Explain binary search in simple words...", key="teach_input")
    if st.button("Get Answer", key="teach_btn"):
        if not prompt.strip():
            st.warning("Please enter a question.")
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            st.warning(f"Prompt truncated to {MAX_PROMPT_CHARS} chars to reduce token usage.")
            prompt = prompt[:MAX_PROMPT_CHARS]
        with st.spinner("Generating..."):
            try:
                resp = generate_with_retry(TEXT_MODEL, prompt, is_image=False)
                text = getattr(resp, "text", None)
                if not text:
                    cand = getattr(resp, "candidates", None) or []
                    if cand:
                        first = cand[0]
                        text = getattr(first, "text", None)
                        if not text:
                            cont = getattr(first, "content", None)
                            if cont:
                                parts = getattr(cont, "parts", []) or []
                                text = " ".join([getattr(p, "text", "") for p in parts])
                text = text or "No text returned."
                st.write(text)
                st.session_state.history.append(("Teaching", prompt, text, now_str()))
            except Exception as e:
                st.error(f"Failed to generate text: {e}")

def run_image_generator():
    st.header("🖼️ Safe AI Image Generator")
    prompt = st.text_area("Describe the image (concise prompts use fewer tokens):", height=200, max_chars=MAX_PROMPT_CHARS, placeholder="A minimal flat-style logo of a white fox curled around a blue moon", key="img_input")
    num_images = st.slider("Variations", min_value=1, max_value=3, value=1, help="Generate 1–3 images (each uses quota).")
    fmt = st.selectbox("Download format", ["PNG", "WEBP", "JPEG"], index=0)
    if st.button("Generate Image(s)", key="img_btn"):
        if not prompt.strip():
            st.warning("Please enter a description.")
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            st.warning("Prompt too long; truncating to reduce token usage.")
            prompt = prompt[:MAX_PROMPT_CHARS]
        with st.spinner("Generating images..."):
            try:
                resp = generate_with_retry(IMAGE_MODEL, prompt, is_image=True)
                images = []
                try:
                    candidates = getattr(resp, "candidates", []) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        parts = getattr(content, "parts", []) or []
                        for p in parts:
                            inline = getattr(p, "inline_data", None)
                            if inline and getattr(inline, "data", None):
                                images.append(inline.data)
                    if not images and candidates:
                        parts = getattr(candidates[0].content, "parts", []) or []
                        if parts:
                            inline = getattr(parts[0], "inline_data", None)
                            if inline and getattr(inline, "data", None):
                                images.append(inline.data)
                except Exception:
                    images = []
                    try:
                        images = [resp.candidates[0].content.parts[0].inline_data.data]
                    except Exception:
                        images = []
                if not images:
                    st.error("No image bytes found in the model response. Try a simpler prompt or wait.")
                    text = getattr(resp, "text", None)
                    if text:
                        st.info(text)
                    return
                images = images[:num_images]
                for i, img_bytes in enumerate(images, start=1):
                    try:
                        img = Image.open(BytesIO(img_bytes))
                    except Exception as ex:
                        st.error(f"Could not decode image #{i}: {ex}")
                        continue
                    st.image(img, caption=f"Generated Image #{i}", use_container_width=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"gemini_image_{ts}_{i}.{fmt.lower()}"
                    buf = BytesIO()
                    if fmt == "JPEG":
                        img.convert("RGB").save(buf, format="JPEG", quality=95)
                    else:
                        img.save(buf, format=fmt)
                    st.download_button(f"⬇️ Download #{i}", data=buf.getvalue(), file_name=fname, mime=f"image/{fmt.lower()}")
                    st.session_state.history.append(("Image", prompt, fname, now_str()))
            except Exception as e:
                st.error(f"Image generation failed: {e}")

def run_math_mastermind():
    st.header("🧠 Math Mastermind")
    q = st.text_area("Enter a math problem (be concise):", height=200, max_chars=MAX_PROMPT_CHARS, key="math_input", placeholder="Integrate x^2 * e^x dx")
    if st.button("Solve", key="math_btn"):
        if not q.strip():
            st.warning("Please enter a math problem.")
            return
        if len(q) > MAX_PROMPT_CHARS:
            st.warning("Question truncated to reduce token usage.")
            q = q[:MAX_PROMPT_CHARS]
        with st.spinner("Solving..."):
            try:
                resp = generate_with_retry(TEXT_MODEL, q, is_image=False)
                text = getattr(resp, "text", None)
                if not text:
                    c = getattr(resp, "candidates", None) or []
                    if c:
                        cont = getattr(c[0], "content", None)
                        if cont:
                            parts = getattr(cont, "parts", []) or []
                            text = " ".join([getattr(p, "text", "") for p in parts])
                text = text or "No answer returned."
                st.write(text)
                st.session_state.history.append(("Math", q, text, now_str()))
            except Exception as e:
                st.error(f"Math solving failed: {e}")

def run_history():
    st.header("📜 History")
    if not st.session_state.history:
        st.info("No interactions yet.")
        return
    for idx, (mode, prompt, output, ts) in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{mode} #{idx} — {ts}"):
            st.markdown("**Prompt:**")
            st.write(prompt)
            st.markdown("**Output:**")
            st.write(output)

if page == "Teaching":
    run_teaching_assistant()
elif page == "Image Generator":
    run_image_generator()
elif page == "Math Solver":
    run_math_mastermind()
else:
    run_history()
