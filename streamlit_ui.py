import streamlit as st
import os
import time
from datetime import datetime
import json
from pathlib import Path
import base64

# AI imports
import openai
import replicate
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import requests
import io

# Anthropic for SEO
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import tempfile for Streamlit Cloud compatibility
import tempfile

# Helper function for API keys (Streamlit Cloud compatibility)
def get_api_key(key_name, default=None):
    """Get API key from st.secrets (Streamlit Cloud) or environment variables (local)"""
    try:
        # Try Streamlit secrets first (for deployed apps)
        return st.secrets.get(key_name, os.getenv(key_name, default))
    except:
        # Fallback to environment variables (for local development)
        return os.getenv(key_name, default)

# Helper function to upload image to Replicate and get URL
def upload_image_to_replicate(image_path):
    """Upload image file to Replicate and return the URL"""
    try:
        # Use Replicate's file upload
        with open(image_path, "rb") as file:
            # Upload file using replicate client
            file_output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={"image": file}
            )
            # This won't work, we need a different approach
    except:
        pass
    
    # Alternative: Return the local path (some models accept local paths)
    # Or we need to host the file somewhere
    return image_path

# Helper function to convert image to data URI for Replicate
def image_to_data_uri(image_path):
    """Convert image file to data URI for Replicate API"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    # Detect image type
    if image_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith('.webp'):
        mime_type = "image/webp"
    else:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{encoded}"

# Retry decorator for rate limits
def retry_with_backoff(max_retries=3, initial_delay=10):
    """Decorator to retry Replicate calls with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "throttled" in error_msg.lower():
                        if attempt < max_retries - 1:
                            st.warning(f"⏳ Rate limit hit. Waiting {delay} seconds before retry... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            st.error("❌ Rate limit exceeded. Please wait a minute or add credits to your Replicate account.")
                            raise
                    else:
                        raise
        return wrapper
    return decorator

# Setup directories for Streamlit Cloud (uses temp directory)
temp_dir = tempfile.gettempdir()
OUTPUTS_DIR = Path(temp_dir) / "outputs"
UPLOADS_DIR = Path(temp_dir) / "uploads"

# Page config
st.set_page_config(
    page_title="AI Image Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Initialize session state for designs
if 'designs' not in st.session_state:
    st.session_state.designs = []

# Create directories (using temp directory for Streamlit Cloud)
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Add warning about ephemeral storage on Streamlit Cloud
if not os.path.exists('.env'):  # Likely running on Streamlit Cloud
    st.warning("⚠️ **Note:** This app uses temporary storage. Download your images immediately as they will be deleted when the app restarts!")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Make tabs scrollable horizontally */
    [data-baseweb="tab-list"] {
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    /* Ensure all tabs are visible */
    [data-baseweb="tab"] {
        flex-shrink: 0;
        min-width: fit-content;
    }
</style>
""", unsafe_allow_html=True)

# AI Service Functions
class AIService:
    @staticmethod
    def generate_prompt(style: str, niche: str, provider: str = "openai") -> str:
        """Generate detailed prompt for image generation"""
        
        system_prompt = """You are an expert at creating prompts for AI image generation for print-on-demand designs.
Create detailed, specific prompts that will result in commercially viable designs.
Focus on clear composition, professional quality, and printable aesthetics.
Keep prompts under 400 characters."""
        
        user_prompt = f"""Create an image generation prompt for a {style} style design in the {niche} niche.
The design should be suitable for print-on-demand products like t-shirts, mugs, and posters.
Make it visually appealing, commercially viable, and on-trend."""
        
        if provider == "openai" and get_api_key("OPENAI_API_KEY"):
            openai.api_key = get_api_key("OPENAI_API_KEY")
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        else:
            # Fallback prompt
            return f"A {style} style {niche} design, professional quality, suitable for print on demand, vector art style, clean composition, commercial appeal"
    
    @staticmethod
    def generate_image_openai(prompt: str) -> str:
        """Generate image using OpenAI DALL-E 3"""
        openai.api_key = get_api_key("OPENAI_API_KEY")
        
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        return response.data[0].url
    
    @staticmethod
    def generate_image_replicate(prompt: str, model: str = "sdxl") -> str:
        """Generate image using Replicate"""
        os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
        
        models = {
            "sdxl": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "flux-dev": "black-forest-labs/flux-dev",
            "flux-schnell": "black-forest-labs/flux-schnell",
        }
        
        model_version = models.get(model, models["sdxl"])
        
        output = replicate.run(
            model_version,
            input={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "num_outputs": 1
            }
        )
        
        if isinstance(output, list):
            return output[0]
        return str(output)

# SEO Metadata Generator
class SEOGenerator:
    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """Encode image to base64 for Anthropic API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def generate_seo_with_anthropic(image_path: str, style: str, niche: str, text_overlay: str = None) -> dict:
        """Generate SEO metadata using Anthropic Claude with vision"""
        
        if not ANTHROPIC_AVAILABLE:
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
        
        anthropic_key = get_api_key("ANTHROPIC_API_KEY")
        if not anthropic_key:
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
        
        try:
            client = anthropic.Anthropic(api_key=anthropic_key)
            
            # Encode image
            image_data = SEOGenerator.encode_image_base64(image_path)
            
            # Try multiple model versions in order of preference
            models_to_try = [
                "claude-3-sonnet-20240229",  # Claude 3 Sonnet with vision
                "claude-3-opus-20240229",    # Claude 3 Opus (if Sonnet unavailable)
                "claude-3-haiku-20240307"    # Claude 3 Haiku (fallback)
            ]
            
            response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    # Create prompt for SEO generation
                    prompt = f"""Analyze this print-on-demand design image and generate SEO-optimized metadata.

Design Context:
- Style: {style}
- Niche: {niche}
- Text on design: {text_overlay if text_overlay else "No text overlay"}

Generate the following in JSON format:

1. **title** (50-80 characters): A compelling, keyword-rich title that:
   - Includes primary keywords (style, niche)
   - Is descriptive and appealing
   - Works for Etsy, Redbubble, Amazon Merch
   - Example: "Minimalist Fitness Motivation Poster - Gym Wall Art Print"

2. **description** (150-250 characters): A detailed description that:
   - Describes what you see in the image
   - Includes relevant keywords naturally
   - Mentions use cases (t-shirt, mug, poster, wall art)
   - Highlights unique features
   - Example: "Eye-catching minimalist fitness design perfect for gym enthusiasts. Features bold typography and clean geometric elements. Ideal for motivational posters, workout apparel, or fitness studio decor. High-quality vector-style artwork."

3. **tags** (15-20 tags): Relevant keywords including:
   - Style descriptors (minimalist, vintage, modern, etc.)
   - Niche-specific terms
   - Product types (shirt, mug, poster, print, sticker)
   - Audience terms (gift for, lovers, enthusiasts)
   - Visual elements you see in the image
   - Trending search terms in this category
   - Example: ["minimalist", "fitness", "gym", "motivation", "workout", "athletic", "sports", "poster", "wall art", "t-shirt design", "fitness gift", "gym decor", "clean design", "modern", "typography"]

4. **keywords** (10-15 keywords): Primary search terms for SEO:
   - Most searchable combinations
   - Long-tail keywords (3-4 word phrases)
   - Example: ["fitness motivation poster", "gym wall art", "minimalist workout print", "athletic motivation", "fitness enthusiast gift"]

5. **alt_text** (125 characters max): Accessible description for screen readers and SEO
   - Describe exactly what's visible
   - Example: "Minimalist black and white fitness motivation poster with geometric shapes and bold text"

6. **search_terms** (comma-separated): Additional Etsy/Amazon search terms
   - Platform-specific keywords
   - Example: "fitness print, gym poster, workout motivation, athletic art, sports decor"

Return ONLY valid JSON with these exact keys: title, description, tags, keywords, alt_text, search_terms

Base your analysis on what you actually SEE in the image, not just the context provided."""

                    # Call Anthropic API with vision
                    response = client.messages.create(
                        model=model,
                        max_tokens=1500,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": image_data
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    )
                    
                    # If successful, break out of loop
                    break
                    
                except Exception as model_error:
                    last_error = model_error
                    print(f"Model {model} failed: {str(model_error)}")
                    continue
            
            # If no model worked
            if response is None:
                print(f"All Anthropic models failed. Last error: {str(last_error)}")
                return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
            
            # Parse response
            content = response.content[0].text
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                seo_data = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ['title', 'description', 'tags', 'keywords', 'alt_text', 'search_terms']
                if all(field in seo_data for field in required_fields):
                    return seo_data
            
            # If parsing failed, use fallback
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
            
        except Exception as e:
            print(f"Anthropic SEO generation failed: {str(e)}")
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
    
    @staticmethod
    def generate_seo_with_openai(style: str, niche: str, prompt: str, text_overlay: str = None) -> dict:
        """Generate SEO metadata using OpenAI (without vision)"""
        
        if not get_api_key("OPENAI_API_KEY"):
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
        
        try:
            openai.api_key = get_api_key("OPENAI_API_KEY")
            
            seo_prompt = f"""Generate SEO-optimized metadata for a print-on-demand design.

Design Details:
- Style: {style}
- Niche: {niche}
- AI Prompt Used: {prompt}
- Text Overlay: {text_overlay if text_overlay else "None"}

Generate JSON with these fields:
- title (50-80 chars): Keyword-rich product title
- description (150-250 chars): Detailed description for listings
- tags (15-20 items): Relevant search tags
- keywords (10-15 items): Primary SEO keywords and phrases
- alt_text (125 chars): Accessibility description
- search_terms (comma-separated): Platform-specific search terms

Return ONLY valid JSON."""

            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an SEO expert for e-commerce and print-on-demand platforms like Etsy, Redbubble, and Amazon Merch."},
                    {"role": "user", "content": seo_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                seo_data = json.loads(json_match.group())
                return seo_data
            
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
            
        except Exception as e:
            print(f"OpenAI SEO generation failed: {str(e)}")
            return SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
    
    @staticmethod
    def generate_seo_fallback(style: str, niche: str, text_overlay: str = None) -> dict:
        """Fallback SEO metadata when AI generation fails"""
        
        text_part = f" - {text_overlay}" if text_overlay else ""
        
        return {
            "title": f"{style.title()} {niche.title()} Design{text_part}",
            "description": f"Premium {style} style design perfect for {niche} enthusiasts. High-quality print-on-demand artwork ideal for t-shirts, mugs, posters, and wall art. Unique design featuring clean composition and professional aesthetics.",
            "tags": [
                style.lower(),
                niche.lower(),
                "pod design",
                "print on demand",
                "t-shirt design",
                "poster art",
                "wall art",
                "mug design",
                "sticker design",
                f"{niche} gift",
                f"{style} art",
                "custom design",
                "unique artwork",
                "digital art",
                "printable"
            ],
            "keywords": [
                f"{style} {niche}",
                f"{niche} design",
                f"{style} artwork",
                f"{niche} print",
                f"{style} poster",
                f"{niche} gift",
                "pod design",
                "print on demand",
                "custom artwork",
                f"{niche} enthusiast"
            ],
            "alt_text": f"{style.title()} style {niche} design with professional quality artwork",
            "search_terms": f"{style} design, {niche} art, pod artwork, print on demand, custom design, {niche} gift, {style} poster"
        }

# Image Processing Functions
class ImageProcessor:
    @staticmethod
    def download_image(url: str) -> bytes:
        """Download image from URL"""
        import requests
        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
        return response.content
    
    @staticmethod
    def upscale_image(image: Image.Image, scale: int = 2) -> Image.Image:
        """Upscale image"""
        new_width = image.width * scale
        new_height = image.height * scale
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def add_text_overlay(image: Image.Image, text: str) -> Image.Image:
        """Add text overlay to image"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center position
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2
        
        # Draw with stroke
        draw.text(
            (x, y),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255)
        )
        
        return img
    
    @staticmethod
    def process_design(image_url: str, design_id: str, text_overlay: str = None, upscale: bool = True):
        """Process the design"""
        # Download
        image_data = ImageProcessor.download_image(image_url)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Upscale
        if upscale:
            image = ImageProcessor.upscale_image(image, scale=2)
        
        # Add text
        if text_overlay:
            image = ImageProcessor.add_text_overlay(image, text_overlay)
        
        # Save
        output_path = str(OUTPUTS_DIR / f"{design_id}.png")
        image.save(output_path, "PNG", dpi=(300, 300), quality=100)
        
        return output_path

# Header
st.markdown('<div class="main-header">🏭 AI Image Factory</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Image Generation & Editing</div>', unsafe_allow_html=True)

# Check API keys (needed for tab content)
has_openai = bool(get_api_key("OPENAI_API_KEY"))
has_replicate = bool(get_api_key("REPLICATE_API_TOKEN"))
has_anthropic = bool(get_api_key("ANTHROPIC_API_KEY"))

# Show rate limit warning if using Replicate
if has_replicate and not has_openai:
    st.warning("⚠️ **Important**: If you have less than $5 credit on Replicate, you're limited to 6 requests/minute. Add credits at [replicate.com/account/billing](https://replicate.com/account/billing) for faster processing.")

# Main tabs
# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎨 Generate", 
    "✂️ Edit", 
    "💬 Caption",
    "🎌 Anime",
    "😊 Emoji",
    "✏️ Sketch",
    "🔧 Restore",
    "📚 Gallery"
])

# Tab 1: Image Generation
with tab1:
    st.header("🎨 Generate New Image")
    
    # Description
    st.markdown("""
    **Transform your ideas into stunning visuals with AI-powered image generation.**
    
    Create professional-quality images from simple text descriptions. Whether you're designing for social media, 
    marketing materials, or creative projects, our AI models bring your vision to life in seconds. Choose from 
    multiple state-of-the-art models including OpenAI DALL-E 3 and Replicate's advanced image generation models 
    for the perfect balance of quality and cost.
    """)
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Design Parameters")
        
        # Style selection with custom option
        style_options = ["minimalist", "vintage", "modern", "retro", "abstract", "geometric", 
                        "watercolor", "illustration", "grunge", "elegant", "🎨 Custom (Type Your Own)"]
        
        selected_style = st.selectbox(
            "Design Style",
            style_options
        )
        
        # If custom selected, show text input
        if selected_style == "🎨 Custom (Type Your Own)":
            style = st.text_input(
                "Enter Custom Style",
                placeholder="e.g., cyberpunk, steampunk, art deco, minimalist geometric...",
                help="Enter any design style you want!"
            )
            if not style:
                style = "modern"  # Fallback if empty
        else:
            style = selected_style
        
        # Niche selection with custom option
        niche_options = ["fitness", "motivation", "coffee", "cats", "dogs", "gaming", 
                        "travel", "nature", "yoga", "music", "coding", "cooking",
                        "✨ Custom (Type Your Own)"]
        
        selected_niche = st.selectbox(
            "Niche",
            niche_options
        )
        
        # If custom selected, show text input
        if selected_niche == "✨ Custom (Type Your Own)":
            niche = st.text_input(
                "Enter Custom Niche",
                placeholder="e.g., meditation, baking, gardening, astronomy, skateboarding...",
                help="Enter any niche or topic you want!"
            )
            if not niche:
                niche = "lifestyle"  # Fallback if empty
        else:
            niche = selected_niche
        
        text_overlay = st.text_input(
            "Text Overlay (Optional)",
            placeholder="Enter text to add to the design"
        )
    
    with col2:
        st.subheader("AI Provider")
        
        provider_options = []
        if has_openai:
            provider_options.append("openai")
        if has_replicate:
            provider_options.append("replicate")
        
        if not provider_options:
            st.error("Please configure at least one API key")
            ai_provider = None
        else:
            ai_provider = st.radio("Choose AI Provider", provider_options)
        
        if ai_provider == "replicate":
            replicate_model = st.selectbox(
                "Replicate Model",
                ["sdxl", "prunaai/p-image", "flux-schnell"]
            )
        else:
            replicate_model = "sdxl"
        
        upscale = st.checkbox("Upscale Image (2x)", value=True)
        
        # Cost estimation
        if ai_provider == "openai":
            estimated_cost = "$0.09"
        else:
            estimated_cost = "$0.003"
        
        st.info(f"💰 Estimated Cost: {estimated_cost}")
        
        # Examples for inspiration
        with st.expander("💡 Need Ideas? Click for Examples"):
            st.markdown("""
            **Popular Style + Niche Combinations:**
            
            🎨 **Styles to try:**
            - Art Deco, Cyberpunk, Steampunk, Bauhaus, Memphis Design
            - Line Art, Stippling, Halftone, Risograph, Screen Print
            - Vaporwave, Synthwave, Y2K, Brutalist, Maximalist
            
            ✨ **Niches to try:**
            - Mental Health, Sustainability, Plant Mom, Book Lover
            - Astronomy, Astrology, Tarot, Crystals, Meditation
            - Skateboarding, Surfing, Hiking, Rock Climbing
            - Podcasting, Journaling, Bullet Journal, Planner
            - Cottagecore, Dark Academia, Witchy, Boho
            
            🔥 **Trending combinations:**
            - "Vaporwave + Meditation"
            - "Line Art + Plant Mom"  
            - "Y2K + Astrology"
            - "Cottagecore + Book Lover"
            - "Synthwave + Gaming"
            """)
    
    # Generate button
    # Validate inputs
    can_generate = True
    error_message = None
    
    if selected_style == "🎨 Custom (Type Your Own)" and not style.strip():
        can_generate = False
        error_message = "⚠️ Please enter a custom style"
    
    if selected_niche == "✨ Custom (Type Your Own)" and not niche.strip():
        can_generate = False
        error_message = "⚠️ Please enter a custom niche"
    
    if error_message:
        st.warning(error_message)
    
    if st.button("🚀 Generate Design", type="primary", use_container_width=True, 
                 disabled=(ai_provider is None or not can_generate)):
        
        design_id = f"design_{int(time.time())}"
        
        with st.spinner("Generating your design... This may take 30-60 seconds..."):
            try:
                start_time = time.time()
                
                # Step 1: Generate prompt
                st.info("📝 Generating prompt...")
                prompt = AIService.generate_prompt(style, niche, ai_provider)
                
                # Step 2: Generate image
                st.info("🎨 Generating image...")
                if ai_provider == "openai":
                    image_url = AIService.generate_image_openai(prompt)
                    cost = 0.09
                else:
                    image_url = AIService.generate_image_replicate(prompt, replicate_model)
                    cost = 0.003
                
                # Step 3: Process image
                st.info("⚙️ Processing image...")
                output_path = ImageProcessor.process_design(
                    image_url, 
                    design_id, 
                    text_overlay if text_overlay else None,
                    upscale
                )
                
                # Step 4: Generate SEO metadata
                st.info("🏷️ Generating SEO metadata...")
                
                # Try Anthropic first (with vision - most accurate)
                if get_api_key("ANTHROPIC_API_KEY") and ANTHROPIC_AVAILABLE:
                    st.caption("Using Anthropic Claude (Vision) for SEO analysis...")
                    seo_data = SEOGenerator.generate_seo_with_anthropic(
                        output_path, style, niche, text_overlay
                    )
                # Fallback to OpenAI (without vision)
                elif get_api_key("OPENAI_API_KEY"):
                    st.caption("Using OpenAI GPT-4 for SEO generation...")
                    seo_data = SEOGenerator.generate_seo_with_openai(
                        style, niche, prompt, text_overlay
                    )
                # Use simple fallback
                else:
                    st.caption("Using template-based SEO...")
                    seo_data = SEOGenerator.generate_seo_fallback(style, niche, text_overlay)
                
                processing_time = time.time() - start_time
                
                # Save to session state
                design_data = {
                    "id": design_id,
                    "style": style,
                    "niche": niche,
                    "text_overlay": text_overlay,
                    "provider": ai_provider,
                    "prompt": prompt,
                    "image_url": image_url,
                    "output_path": output_path,
                    "processing_time": processing_time,
                    "cost": cost,
                    "created_at": datetime.now().isoformat(),
                    # SEO metadata
                    "seo_title": seo_data.get("title", ""),
                    "seo_description": seo_data.get("description", ""),
                    "seo_tags": seo_data.get("tags", []),
                    "seo_keywords": seo_data.get("keywords", []),
                    "seo_alt_text": seo_data.get("alt_text", ""),
                    "seo_search_terms": seo_data.get("search_terms", "")
                }
                
                st.session_state.designs.insert(0, design_data)
                
                st.success(f"✅ Design generated in {processing_time:.2f}s!")
                
                # Display result
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Generated Design")
                    st.image(output_path, use_container_width=True)
                    
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Design",
                            data=f.read(),
                            file_name=f"{design_id}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                with col2:
                    st.subheader("Details")
                    st.write(f"**Style:** {style}")
                    st.write(f"**Niche:** {niche}")
                    st.write(f"**Prompt:** {prompt}")
                    st.write(f"**Provider:** {ai_provider}")
                    st.write(f"**Processing Time:** {processing_time:.2f}s")
                    st.write(f"**Cost:** ${cost:.4f}")
                    
                    st.divider()
                    
                    # SEO Metadata Section
                    st.subheader("📊 SEO Metadata")
                    
                    # Title
                    st.write("**Title:**")
                    st.code(seo_data.get("title", ""), language=None)
                    
                    # Description
                    st.write("**Description:**")
                    st.text_area("Description text", seo_data.get("description", ""), height=100, key="desc_display", disabled=True, label_visibility="collapsed")
                    
                    # Alt Text
                    st.write("**Alt Text:**")
                    st.code(seo_data.get("alt_text", ""), language=None)
                    
                    # Tags
                    st.write("**Tags:**")
                    tags = seo_data.get("tags", [])
                    if tags:
                        st.code(", ".join(tags[:10]), language=None)
                        if len(tags) > 10:
                            with st.expander("Show all tags"):
                                st.code(", ".join(tags), language=None)
                    
                    # Keywords
                    st.write("**Keywords:**")
                    keywords = seo_data.get("keywords", [])
                    if keywords:
                        st.code(", ".join(keywords), language=None)
                    
                    # Search Terms
                    st.write("**Search Terms:**")
                    st.code(seo_data.get("search_terms", ""), language=None)
                    
                    # Download SEO as JSON
                    seo_json = json.dumps(seo_data, indent=2)
                    st.download_button(
                        "📥 Download SEO Data (JSON)",
                        data=seo_json,
                        file_name=f"{design_id}_seo.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    # Download SEO as Text
                    seo_text = f"""Title: {seo_data.get('title', '')}

Description:
{seo_data.get('description', '')}

Alt Text:
{seo_data.get('alt_text', '')}

Tags:
{', '.join(seo_data.get('tags', []))}

Keywords:
{', '.join(seo_data.get('keywords', []))}

Search Terms:
{seo_data.get('search_terms', '')}
"""
                    st.download_button(
                        "📄 Download SEO Data (TXT)",
                        data=seo_text,
                        file_name=f"{design_id}_seo.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Tab 2: Image Editing
with tab2:
    st.header("✂️ Image Editing")
    
    # Description
    st.markdown("""
    **Transform and enhance your images with AI-powered editing tools.**
    
    Remove backgrounds instantly, apply custom edits with natural language prompts, or transform image styles. 
    Our advanced AI models understand your instructions and deliver professional results in seconds. Perfect for 
    e-commerce product photos, social media content, and creative projects. No complex software needed—just 
    describe what you want and let AI do the work.
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image to edit",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload the image you want to edit"
        )
        
        if uploaded_file:
            # Display uploaded image
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
            
            # Save uploaded file
            upload_path = str(UPLOADS_DIR / f"upload_{int(time.time())}.png")
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.subheader("Editing Options")
        
        # Editing mode selection
        edit_mode = st.radio(
            "Choose editing mode",
            ["Remove Background", "Custom Edit (AI Prompt)"],
            help="Select what you want to do with the image"
        )
        
        if edit_mode == "Remove Background":
            st.info("💡 This will remove the background from your image, leaving only the main subject on a transparent background.")
            
            # Model selection for background removal
            # Get default model from env
            default_bg_model = os.getenv("DEFAULT_BG_REMOVAL_MODEL", "briaai/rmbg-1.4:606bd08d63f1d0c00c233c4ba2793801c93e1777e0f23cc7dc64e6a34b24d1e6")
            
            bg_removal_choice = st.selectbox(
                "Background Removal Model",
                ["Default Model (from config)", "Custom Model"],
                help="Use default model from .env or specify custom Replicate model",
                key="bg_removal_sel"
            )
            
            if bg_removal_choice == "Custom Model":
                bg_removal_model = st.text_input(
                    "Custom Replicate Model ID",
                    placeholder="owner/model-name:version-hash",
                    help="Enter full Replicate model ID for background removal",
                    key="bg_custom"
                )
                if not bg_removal_model:
                    bg_removal_model = default_bg_model
            else:
                bg_removal_model = default_bg_model
                st.info(f"📋 Using: `{bg_removal_model.split(':')[0]}`")
            
            custom_prompt = None
            
        else:  # Custom Edit
            st.info("💡 Describe how you want to edit your image using natural language")
            
            custom_prompt = st.text_area(
                "Image Editing Prompt",
                placeholder="Examples:\n- Make the image more vibrant and colorful\n- Add a soft blur effect\n- Enhance the contrast and brightness\n- Convert to black and white\n- Add a vintage film effect\n- Sharpen the details",
                height=150,
                help="Describe the edits you want in natural language"
            )
            
            # Get default model from env
            default_edit_model = os.getenv("DEFAULT_IMAGE_EDIT_MODEL", "timothybrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f")
            
            # Model selection for custom editing
            edit_model_choice = st.selectbox(
                "Image Editing Model",
                ["Default Model (from config)", "Custom Model"],
                help="Use default model from .env or specify custom Replicate model",
                key="edit_model_sel"
            )
            
            if edit_model_choice == "Custom Model":
                edit_model = st.text_input(
                    "Custom Replicate Model ID",
                    placeholder="owner/model-name:version-hash",
                    help="Enter full Replicate model ID for image editing",
                    key="edit_custom"
                )
                if not edit_model:
                    edit_model = default_edit_model
            else:
                edit_model = default_edit_model
                st.info(f"📋 Using: `{edit_model.split(':')[0]}`")
            
            # Strength parameter for custom edits
            edit_strength = st.slider(
                "Edit Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="How much to modify the image (0.0 = minimal, 1.0 = maximum)"
            )
    
    with col2:
        st.subheader("Results")
        
        if uploaded_file:
            # Process button
            if st.button("🚀 Process Image", type="primary", use_container_width=True):
                
                with st.spinner(f"Processing with AI... This may take 30-60 seconds..."):
                    try:
                        start_time = time.time()
                        
                        if edit_mode == "Remove Background":
                            st.info(f"🔧 Removing background...")
                            
                            # Process with Replicate
                            os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                            
                            # Convert image to data URI
                            image_uri = image_to_data_uri(upload_path)
                            
                            output = replicate.run(
                                bg_removal_model,
                                input={"image": image_uri}
                            )
                            
                            # Download result
                            if isinstance(output, str):
                                result_url = output
                            elif isinstance(output, list):
                                result_url = output[0]
                            else:
                                result_url = str(output)
                            
                            result_data = requests.get(result_url).content
                            result_path = str(OUTPUTS_DIR / f"edited_{int(time.time())}.png")
                            
                            with open(result_path, "wb") as f:
                                f.write(result_data)
                            
                            processing_time = time.time() - start_time
                            
                            st.success(f"✅ Background removed in {processing_time:.2f}s!")
                            st.image(result_path, caption="Result - Background Removed", use_container_width=True)
                            
                            # Download button
                            with open(result_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Result",
                                    data=f.read(),
                                    file_name="background_removed.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            
                            st.info(f"💰 Estimated cost: $0.005")
                            
                        else:  # Custom Edit
                            if not custom_prompt:
                                st.warning("⚠️ Please enter an editing prompt")
                            else:
                                st.info(f"✨ Editing image with: '{custom_prompt}'")
                                
                                # Process with Replicate
                                os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                                
                                # Open file - Replicate SDK will handle the upload
                                with open(upload_path, "rb") as image_file:
                                    try:
                                        # Format for p-image-edit and similar models
                                        output = replicate.run(
                                            edit_model,
                                            input={
                                                "images": [image_file],  # Array with file handle
                                                "prompt": custom_prompt,
                                                "turbo": True,
                                                "aspect_ratio": "1:1"
                                            }
                                        )
                                    except Exception as e1:
                                        st.warning(f"Trying alternative format... ({str(e1)[:50]})")
                                        # Try data URI format
                                        image_uri = image_to_data_uri(upload_path)
                                        try:
                                            output = replicate.run(
                                                edit_model,
                                                input={
                                                    "images": [image_uri],
                                                    "prompt": custom_prompt,
                                                    "turbo": True
                                                }
                                            )
                                        except Exception as e2:
                                            # Standard InstructPix2Pix format
                                            output = replicate.run(
                                                edit_model,
                                                input={
                                                    "image": image_uri,
                                                    "prompt": custom_prompt,
                                                    "num_inference_steps": 20,
                                                    "guidance_scale": 7.5,
                                                    "image_guidance_scale": edit_strength
                                                }
                                            )
                                
                                # Download result
                                if isinstance(output, str):
                                    result_url = output
                                elif isinstance(output, list):
                                    result_url = output[0]
                                else:
                                    result_url = str(output)
                                
                                result_data = requests.get(result_url).content
                                result_path = str(OUTPUTS_DIR / f"edited_{int(time.time())}.png")
                                
                                with open(result_path, "wb") as f:
                                    f.write(result_data)
                                
                                processing_time = time.time() - start_time
                                
                                st.success(f"✅ Image edited in {processing_time:.2f}s!")
                                st.image(result_path, caption=f"Result - {custom_prompt}", use_container_width=True)
                                
                                # Download button
                                with open(result_path, "rb") as f:
                                    st.download_button(
                                        "⬇️ Download Result",
                                        data=f.read(),
                                        file_name="edited_image.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                                
                                st.info(f"💰 Estimated cost: $0.01")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        else:
            st.info("👆 Upload an image to get started")

# Tab 3: Caption Images
with tab3:
    st.header("💬 Caption Images")
    
    # Description
    st.markdown("""
    **Generate text descriptions and captions from images using advanced multimodal AI.**
    
    Our AI uses large multimodal transformers trained on millions of image-text pairs to understand visual concepts 
    and generate accurate descriptions. Perfect for creating alt text for accessibility, indexing your photo library, 
    answering questions about image content, or generating detailed prompts for AI art generation. Simply upload an 
    image and let our AI analyze and describe it in natural language.
    
    **Key capabilities:**
    - 📝 **Image Captioning**: Produce relevant captions summarizing image contents and context
    - ❓ **Visual Q&A**: Ask questions and get natural language answers about your images  
    - ✨ **Prompt Generation**: Create detailed prompts matching image style and content
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        caption_file = st.file_uploader(
            "Choose an image to caption",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload the image you want to get captions for",
            key="caption_upload"
        )
        
        if caption_file:
            st.image(caption_file, caption="Image to Caption", use_container_width=True)
            
            upload_path = str(UPLOADS_DIR / f"caption_{int(time.time())}.png")
            with open(upload_path, "wb") as f:
                f.write(caption_file.getbuffer())
        
        st.subheader("Caption Options")
        
        caption_mode = st.radio(
            "Choose mode",
            ["Auto Caption", "Visual Q&A", "Generate Text Prompt"],
            help="Select what you want to generate from the image"
        )
        
        if caption_mode == "Visual Q&A":
            user_question = st.text_input(
                "Ask a question about the image",
                placeholder="What is in this image? What color is...? How many...?",
                help="Ask any question about the image content"
            )
        else:
            user_question = None
        
        # Get default model from env
        default_caption_model = os.getenv("DEFAULT_CAPTION_MODEL", "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb")
        
        caption_model_choice = st.selectbox(
            "Model Selection",
            ["Default Model (from config)", "Custom Model"],
            help="Use default model from .env or specify custom Replicate model"
        )
        
        if caption_model_choice == "Custom Model":
            caption_model = st.text_input(
                "Custom Replicate Model ID",
                placeholder="owner/model-name:version-hash",
                help="Enter full Replicate model ID (e.g., yorickvp/llava-13b:hash...)"
            )
            if not caption_model:
                caption_model = default_caption_model
        else:
            caption_model = default_caption_model
            st.info(f"📋 Using: `{caption_model.split(':')[0]}`")
    
    with col2:
        st.subheader("Results")
        
        if caption_file:
            if st.button("🚀 Generate Caption", type="primary", use_container_width=True, key="caption_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                        
                        if caption_mode == "Visual Q&A" and user_question:
                            prompt = user_question
                        elif caption_mode == "Generate Text Prompt":
                            prompt = "Describe this image in detail, focusing on the style, composition, colors, mood, and visual elements. Create a detailed prompt that could be used to generate a similar image."
                        else:
                            prompt = "Describe this image in detail."
                        
                        # Convert image to data URI
                        image_uri = image_to_data_uri(upload_path)
                        
                        output = replicate.run(
                            caption_model,
                            input={
                                "image": image_uri,
                                "prompt": prompt
                            }
                        )
                        
                        result = "".join(output) if hasattr(output, '__iter__') else str(output)
                        
                        st.success("✅ Caption generated!")
                        
                        if caption_mode == "Auto Caption":
                            st.markdown("**📝 Image Caption:**")
                        elif caption_mode == "Visual Q&A":
                            st.markdown(f"**❓ Q:** {user_question}")
                            st.markdown("**💡 A:**")
                        else:
                            st.markdown("**✨ Generated Prompt:**")
                        
                        st.write(result)
                        
                        # Download as text
                        st.download_button(
                            "📥 Download Caption",
                            data=result,
                            file_name="caption.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                        st.info("💰 Cost: ~$0.01")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload an image to get started")

# Tab 4: Anime-Style Images
with tab4:
    st.header("🎌 Generate Anime-Style Images")
    
    # Description
    st.markdown("""
    **Create stunning anime and manga-style artwork with specialized AI models.**
    
    Our AI excels at generating images and videos in the authentic style of anime. Whether you're designing characters, 
    exploring new artistic styles, or transforming existing concepts, these models help you produce polished anime 
    visuals with minimal setup. Perfect for character design, fan art, game development, storytelling, and creative 
    projects. Choose from Studio Ghibli aesthetics, modern anime styles, or classic manga looks.
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generation Settings")
        
        anime_prompt = st.text_area(
            "Describe your anime image",
            placeholder="A girl with blue hair in a magical forest, Studio Ghibli style\nAn anime warrior with glowing sword, dynamic pose\nCute anime cat character, kawaii style",
            height=100,
            help="Describe the anime-style image you want to create"
        )
        
        # Get default model from env
        default_anime_model = os.getenv("DEFAULT_ANIME_MODEL", "aaronaftab/mirage-ghibli:166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f")
        
        anime_model_choice = st.selectbox(
            "Anime Model",
            ["Default Model (from config)", "Custom Model"],
            help="Use default model from .env or specify custom Replicate model",
            key="anime_model_sel"
        )
        
        if anime_model_choice == "Custom Model":
            anime_model = st.text_input(
                "Custom Replicate Model ID",
                placeholder="owner/model-name:version-hash",
                help="Enter full Replicate model ID for anime generation"
            )
            if not anime_model:
                anime_model = default_anime_model
        else:
            anime_model = default_anime_model
            st.info(f"📋 Using: `{anime_model.split(':')[0]}`")
        
        anime_negative = st.text_input(
            "Negative Prompt (Optional)",
            placeholder="low quality, blurry, distorted",
            help="Things you don't want in the image"
        )
    
    with col2:
        st.subheader("Results")
        
        if st.button("🚀 Generate Anime Image", type="primary", use_container_width=True, key="anime_btn"):
            if not anime_prompt:
                st.warning("⚠️ Please enter a prompt")
            else:
                with st.spinner("Creating anime artwork..."):
                    try:
                        os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                        
                        output = replicate.run(
                            anime_model,
                            input={
                                "prompt": anime_prompt,
                                "negative_prompt": anime_negative if anime_negative else "low quality, blurry"
                            }
                        )
                        
                        result_url = output[0] if isinstance(output, list) else str(output)
                        result_data = requests.get(result_url).content
                        result_path = str(OUTPUTS_DIR / f"anime_{int(time.time())}.png")
                        
                        with open(result_path, "wb") as f:
                            f.write(result_data)
                        
                        st.success("✅ Anime image created!")
                        st.image(result_path, use_container_width=True)
                        
                        with open(result_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name="anime_image.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        st.info("💰 Cost: ~$0.01")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# Tab 5: Generate Emojis
with tab5:
    st.header("😊 Generate Emojis")
    
    # Description
    st.markdown("""
    **Create custom emojis from text descriptions or transform images into emoji style.**
    
    Our AI models can generate brand new emoji-style graphics from simple text prompts, or convert real photos into 
    Apple-style emoji characters. The collection includes fine-tuned diffusion models, image-to-emoji converters, 
    and flexible icon generators. Perfect for designing custom emoji sets, creating personalized emoji avatars, 
    making stickers for chat apps, or building fun visuals for social media and branding.
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Emoji Settings")
        
        emoji_mode = st.radio(
            "Generation Mode",
            ["Text to Emoji", "Image to Emoji"],
            help="Create emoji from text description or convert image"
        )
        
        if emoji_mode == "Text to Emoji":
            emoji_prompt = st.text_input(
                "Describe your emoji",
                placeholder="Happy robot, smiling cat, crying laughing face, cool sunglasses",
                help="Describe the emoji you want to create"
            )
            emoji_file = None
        else:
            emoji_file = st.file_uploader(
                "Upload image to convert to emoji",
                type=['png', 'jpg', 'jpeg'],
                key="emoji_upload"
            )
            emoji_prompt = "Convert to emoji style"
            
            if emoji_file:
                st.image(emoji_file, caption="Image to convert", width=200)
        
        # Get default model from env
        default_emoji_model = os.getenv("DEFAULT_EMOJI_MODEL", "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e")
        
        emoji_model_choice = st.selectbox(
            "Emoji Model",
            ["Default Model (from config)", "Custom Model"],
            help="Use default model from .env or specify custom Replicate model",
            key="emoji_model_sel"
        )
        
        if emoji_model_choice == "Custom Model":
            emoji_model = st.text_input(
                "Custom Replicate Model ID",
                placeholder="owner/model-name:version-hash",
                help="Enter full Replicate model ID for emoji generation",
                key="emoji_custom"
            )
            if not emoji_model:
                emoji_model = default_emoji_model
        else:
            emoji_model = default_emoji_model
            st.info(f"📋 Using: `{emoji_model.split(':')[0]}`")
    
    with col2:
        st.subheader("Results")
        
        should_generate = (emoji_mode == "Text to Emoji" and emoji_prompt) or (emoji_mode == "Image to Emoji" and emoji_file)
        
        if st.button("🚀 Generate Emoji", type="primary", use_container_width=True, disabled=not should_generate, key="emoji_btn"):
            with st.spinner("Creating emoji..."):
                try:
                    os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                    
                    if emoji_mode == "Image to Emoji" and emoji_file:
                        upload_path = str(UPLOADS_DIR / f"emoji_{int(time.time())}.png")
                        with open(upload_path, "wb") as f:
                            f.write(emoji_file.getbuffer())
                        
                        # Open file for Replicate to upload
                        with open(upload_path, "rb") as image_file:
                            output = replicate.run(
                                emoji_model,
                                input={
                                    "prompt": emoji_prompt if emoji_prompt else "Turn this image into the emoji style of Apple iOS system",
                                    "input_image": image_file,  # Correct parameter name
                                    "aspect_ratio": "match_input_image",
                                    "lora_strength": 1,
                                    "output_format": "webp"
                                }
                            )
                    else:
                        # Text to Emoji mode - different models support this
                        # flux-kontext is primarily for image-to-emoji
                        # For text-to-emoji, use SDXL-emoji or similar
                        
                        # Check if using flux-kontext model
                        if "flux-kontext" in emoji_model or "kontext-emoji" in emoji_model:
                            # This model requires an image, so use sdxl-emoji for text mode
                            text_emoji_model = "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e"
                            st.info("💡 Using SDXL-Emoji for text-to-emoji (flux-kontext requires an image)")
                            
                            output = replicate.run(
                                text_emoji_model,
                                input={
                                    "prompt": f"A {emoji_prompt} emoji, high quality",
                                    "apply_watermark": False
                                }
                            )
                        else:
                            # Use the selected model
                            output = replicate.run(
                                emoji_model,
                                input={
                                    "prompt": emoji_prompt,
                                    "output_format": "webp"
                                }
                            )
                    
                    result_url = output[0] if isinstance(output, list) else str(output)
                    result_data = requests.get(result_url).content
                    result_path = str(OUTPUTS_DIR / f"emoji_{int(time.time())}.png")
                    
                    with open(result_path, "wb") as f:
                        f.write(result_data)
                    
                    st.success("✅ Emoji created!")
                    st.image(result_path, width=200)
                    
                    with open(result_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            data=f.read(),
                            file_name="custom_emoji.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    st.info("💰 Cost: ~$0.01")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 6: Sketches to Images
with tab6:
    st.header("✏️ Turn Sketches into Images")
    
    # Description
    st.markdown("""
    **Transform your rough sketches and line drawings into polished, professional visuals.**
    
    Turn simple sketches into lifelike art, concept renderings, or detailed illustrations. Our AI models accept 
    images of your sketches as input and transform them into detailed, polished output images according to your 
    text prompt. Perfect for concept artists, designers, architects, game developers, and anyone who wants to 
    quickly visualize ideas. Maintains your original composition while adding realistic details, colors, and 
    professional finishing touches.
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Sketch")
        
        sketch_file = st.file_uploader(
            "Upload your sketch",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a sketch or line drawing",
            key="sketch_upload"
        )
        
        if sketch_file:
            st.image(sketch_file, caption="Your Sketch", use_container_width=True)
        
        sketch_prompt = st.text_area(
            "Describe the final image",
            placeholder="A realistic landscape with mountains and lake\nA fantasy character with armor and sword\nA modern building with glass facade",
            height=100,
            help="Describe how you want the final polished image to look"
        )
        
        # Get default model from env
        default_sketch_model = os.getenv("DEFAULT_SKETCH_MODEL", "jagilley/controlnet-scribble:435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117")
        
        sketch_model_choice = st.selectbox(
            "Model Selection",
            ["Default Model (from config)", "Custom Model"],
            help="Use default model from .env or specify custom Replicate model",
            key="sketch_model_sel"
        )
        
        if sketch_model_choice == "Custom Model":
            sketch_model = st.text_input(
                "Custom Replicate Model ID",
                placeholder="owner/model-name:version-hash",
                help="Enter full Replicate model ID for sketch transformation",
                key="sketch_custom"
            )
            if not sketch_model:
                sketch_model = default_sketch_model
        else:
            sketch_model = default_sketch_model
            st.info(f"📋 Using: `{sketch_model.split(':')[0]}`")
    
    with col2:
        st.subheader("Results")
        
        if sketch_file and sketch_prompt:
            if st.button("🚀 Transform Sketch", type="primary", use_container_width=True, key="sketch_btn"):
                with st.spinner("Transforming sketch..."):
                    try:
                        upload_path = str(UPLOADS_DIR / f"sketch_{int(time.time())}.png")
                        with open(upload_path, "wb") as f:
                            f.write(sketch_file.getbuffer())
                        
                        os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                        
                        # Convert to data URI
                        image_uri = image_to_data_uri(upload_path)
                        
                        output = replicate.run(
                            sketch_model,
                            input={
                                "image": image_uri,
                                "prompt": sketch_prompt,
                                "num_samples": "1",
                                "image_resolution": "512"
                            }
                        )
                        
                        result_url = output[0] if isinstance(output, list) else str(output)
                        result_data = requests.get(result_url).content
                        result_path = str(OUTPUTS_DIR / f"sketch_result_{int(time.time())}.png")
                        
                        with open(result_path, "wb") as f:
                            f.write(result_data)
                        
                        st.success("✅ Sketch transformed!")
                        st.image(result_path, caption="Transformed Result", use_container_width=True)
                        
                        with open(result_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name="transformed_sketch.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        st.info("💰 Cost: ~$0.01")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload a sketch and add a description to get started")

# Tab 7: Restore Images
with tab7:
    st.header("🔧 Restore Images")
    
    # Description
    st.markdown("""
    **Restore and improve images by fixing defects like blur, noise, and missing colors.**
    
    Our AI models can breathe new life into old, damaged, or low-quality images. Choose from multiple restoration 
    capabilities:
    
    - 🎨 **Colorization**: Add realistic, natural-looking color to black and white photos
    - 🔍 **Deblurring**: Sharpen blurry images by reversing blur effects—perfect for old or motion-blurred photos  
    - ✨ **Denoising**: Remove grain, artifacts, and noise while preserving important details
    
    Perfect for digitizing family photos, restoring historical images, improving scanned documents, or enhancing 
    any image that needs quality improvements.
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        restore_file = st.file_uploader(
            "Upload image to restore",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a blurry, noisy, or black & white image",
            key="restore_upload"
        )
        
        if restore_file:
            st.image(restore_file, caption="Original Image", use_container_width=True)
        
        restore_mode = st.radio(
            "Restoration Type",
            ["Colorization", "Deblurring", "Denoising"],
            help="Select the type of restoration"
        )
        
        # Advanced options based on mode
        if restore_mode == "Deblurring":
            deblur_quality = st.selectbox(
                "Quality Level",
                ["Large (Best Quality)", "Medium (Faster)"],
                help="Higher quality takes longer but produces better results"
            )
        elif restore_mode == "Denoising":
            denoise_type = st.selectbox(
                "Image Type",
                ["Color Image", "Grayscale Image"],
                help="Select based on your image type"
            )
        
        # Get default models from env based on mode
        if restore_mode == "Colorization":
            st.info("💡 Add realistic color to black and white photos")
            default_model = os.getenv("DEFAULT_COLORIZE_MODEL", "piddnad/ddcolor:ca494ba129e44e45f661d6ece83c4c98a9a7c774309beca01429b58fce8aa695")
        elif restore_mode == "Deblurring":
            st.info("💡 Sharpen blurry images and enhance resolution")
            default_model = os.getenv("DEFAULT_DEBLUR_MODEL", "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a")
        else:
            st.info("💡 Remove grain and artifacts from photos")
            default_model = os.getenv("DEFAULT_DENOISE_MODEL", "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a")
        
        restore_model_choice = st.selectbox(
            "Model Selection",
            ["Default Model (from config)", "Custom Model"],
            help="Use default model from .env or specify custom Replicate model",
            key="restore_model_sel"
        )
        
        if restore_model_choice == "Custom Model":
            restore_model = st.text_input(
                "Custom Replicate Model ID",
                placeholder="owner/model-name:version-hash",
                help="Enter full Replicate model ID for image restoration",
                key="restore_custom"
            )
            if not restore_model:
                restore_model = default_model
        else:
            restore_model = default_model
            st.info(f"📋 Using: `{restore_model.split(':')[0]}`")
    
    with col2:
        st.subheader("Results")
        
        if restore_file:
            if st.button("🚀 Restore Image", type="primary", use_container_width=True, key="restore_btn"):
                with st.spinner(f"Applying {restore_mode.lower()}..."):
                    try:
                        upload_path = str(UPLOADS_DIR / f"restore_{int(time.time())}.png")
                        with open(upload_path, "wb") as f:
                            f.write(restore_file.getbuffer())
                        
                        os.environ["REPLICATE_API_TOKEN"] = get_api_key("REPLICATE_API_TOKEN", "")
                        
                        # Convert to data URI
                        image_uri = image_to_data_uri(upload_path)
                        
                        if restore_mode == "Colorization":
                            output = replicate.run(
                                restore_model,
                                input={"image": image_uri}
                            )
                        elif restore_mode == "Deblurring":
                            # Determine task type based on quality selection
                            if "Large" in deblur_quality:
                                task = "Real-World Image Super-Resolution-Large"
                            else:
                                task = "Real-World Image Super-Resolution-Medium"
                            
                            output = replicate.run(
                                restore_model,
                                input={
                                    "image": image_uri,
                                    "task_type": task
                                }
                            )
                        else:  # Denoising
                            # Determine task type based on image type
                            if "Color" in denoise_type:
                                task = "Color Image Denoising"
                            else:
                                task = "Grayscale Image Denoising"
                            
                            output = replicate.run(
                                restore_model,
                                input={
                                    "image": image_uri,
                                    "task_type": task
                                }
                            )
                        
                        result_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) else str(output))
                        result_data = requests.get(result_url).content
                        result_path = str(OUTPUTS_DIR / f"restored_{int(time.time())}.png")
                        
                        with open(result_path, "wb") as f:
                            f.write(result_data)
                        
                        st.success(f"✅ {restore_mode} complete!")
                        st.image(result_path, caption="Restored Image", use_container_width=True)
                        
                        with open(result_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name="restored_image.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        st.info("💰 Cost: ~$0.01")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload an image to restore")

# Tab 8: Gallery
with tab8:
    st.header("📚 Gallery")
    
    # Description
    st.markdown("""
    **View and manage all your AI-generated creations in one place.**
    
    Browse through your image generation history, download your creations, and access SEO metadata. Each image 
    is saved with complete details including the AI model used, generation cost, timestamps, and any SEO data 
    like titles, descriptions, and tags. Perfect for organizing your creative work, tracking your projects, 
    and managing your AI-generated assets.
    """)
    st.divider()
    
    # Load all images from outputs folder
    import glob
    from pathlib import Path
    
    output_images = []
    
    # Get all images from outputs folder
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        output_images.extend(glob.glob(str(OUTPUTS_DIR / ext)))
    
    # Sort by modification time (newest first)
    output_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if output_images or st.session_state.designs:
        st.success(f"📁 Found {len(output_images)} images in outputs folder")
        
        cols = st.columns(3)
        
        # Show images from outputs folder
        for idx, img_path in enumerate(output_images):
            with cols[idx % 3]:
                try:
                    st.image(img_path, use_container_width=True)
                    
                    # Get filename and info
                    filename = Path(img_path).name
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(img_path))
                    
                    st.write(f"**{filename[:30]}...**" if len(filename) > 30 else f"**{filename}**")
                    st.caption(f"📅 {mod_time.strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"💾 {file_size:.1f} KB")
                    
                    # Download button
                    with open(img_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            data=f.read(),
                            file_name=filename,
                            mime="image/png",
                            key=f"download_{idx}_{filename}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error loading {img_path}: {str(e)}")
    else:
        st.info("📭 No images yet. Create your first image using any of the features above!")
        st.markdown("""
        **Get started:**
        - 🎨 **Generate** - Create images from text
        - ✂️ **Edit** - Remove backgrounds or edit images
        - 💬 **Caption** - Generate captions from images
        - 🎌 **Anime** - Create anime-style artwork
        - 😊 **Emoji** - Make custom emojis
        - ✏️ **Sketch** - Transform sketches to images
        - 🔧 **Restore** - Colorize, deblur, or denoise photos
        """)

# Footer
st.divider()
st.caption("AI Image Factory v1.0 • Powered by FastAPI & Streamlit • AI: OpenAI, Replicate & Anthropic")
