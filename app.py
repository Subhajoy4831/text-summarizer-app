import streamlit as st
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load model once and cache it."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

def format_summary(summary_text, tone):
    """Format summary based on tone."""
    if tone == "Bullet Points":
        sentences = [s.strip() for s in summary_text.replace('. ', '.\n').split('\n') if s.strip()]
        return '\n'.join([f"• {sentence}" for sentence in sentences])
    elif tone == "Casual":
        import random
        casual_intro = ["So basically, ", "Here's the deal: ", "In a nutshell, ", "Long story short, "]
        return random.choice(casual_intro) + summary_text.lower().capitalize()
    return summary_text

def main():
    # Header
    st.title("AI Text Summarizer")
    st.markdown("*Powered by Hugging Face Transformers*")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading AI model... (This may take a minute on first run)"):
        summarizer = load_model()
    
    st.success("Model loaded successfully!")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter your text:",
            height=300,
            placeholder="Paste your text here to get a summary...",
            help="Enter the text you want to summarize"
        )
        
        # Character count
        if text_input:
            st.caption(f"Characters: {len(text_input)}")
    
    with col2:
        st.subheader("Settings")
        
        tone = st.selectbox(
            "Tone:",
            ["Formal", "Casual", "Bullet Points"],
            help="Choose the style of your summary"
        )
        
        length = st.selectbox(
            "Length:",
            ["Brief", "Medium", "Detailed"],
            index=1,
            help="Choose how long the summary should be"
        )
        
        st.markdown("---")
        
        summarize_button = st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True
        )
    
    # Length parameters
    length_params = {
        "Brief": {"max_length": 50, "min_length": 20},
        "Medium": {"max_length": 130, "min_length": 50},
        "Detailed": {"max_length": 250, "min_length": 100}
    }
    
    # Process when button is clicked
    if summarize_button:
        if not text_input.strip():
            st.error("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    params = length_params[length]
                    result = summarizer(
                        text_input,
                        max_length=params["max_length"],
                        min_length=params["min_length"],
                        do_sample=False
                    )
                    
                    summary = result[0]['summary_text']
                    formatted = format_summary(summary, tone)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Summary")

                    if tone == "Bullet Points":
                        st.markdown(formatted.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        st.info(formatted)
                    
                    # Statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Original Length", f"{len(text_input)} chars")
                    with col_b:
                        st.metric("Summary Length", f"{len(formatted)} chars")
                    with col_c:
                        reduction = round((1 - len(formatted)/len(text_input)) * 100)
                        st.metric("Reduction", f"{reduction}%")
                    
                    # Download button
                    st.download_button(
                        label="Download Summary",
                        data=formatted,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Made with using Streamlit and Hugging Face</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()