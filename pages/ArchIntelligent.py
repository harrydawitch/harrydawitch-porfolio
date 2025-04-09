import streamlit as st 
import pandas as pd
from PIL import Image

st.set_page_config(
                    page_title="Projects | ArchIntelligent",
                    layout= 'wide',
                    )

@st.cache_data
def load_image(path):
    return Image.open(path)

@st.cache_data
def load_building_data():
    data = {
        "Buildings": ["Building 1", "Building 2", "Building 3"],
        "image_url": [
            "*img1_url, img2_url, img3_url…",
            "*img1_url, img2_url, img3_url…",
            "*img1_url, img2_url, img3_url…"
        ],
        "article_texts": [
            "Sand and Soil, located in Jogjakarta…",
            "The villa's architecture is thoughtfully…",
            "Inside the highlands of eastern Antioquia…"
        ],
        "Functional": ["Apartment", "Villa", "Office"],
        "Style": ["Sustainable", "Minimalism", "Modern"]
    }
    return pd.DataFrame(data)

st.title(":red[Arch]Intelligent")
st.markdown(
    """
    <a href="https://github.com/harrydawitch/ArchIntelligent" target="_blank">
        <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub Repo" style="vertical-align:middle; margin-right:10px;"/>
        View on GitHub
    </a>
    """,
    unsafe_allow_html=True
)
st.divider()


# Custom CSS to style headers and emojis
st.markdown("""
    <style>
        h1, h2, h3 {
            color: #ff4b4b;
        }
        .emoji {
            font-size: 1.4em;
        }
    </style>
""", unsafe_allow_html=True)

with st.container(border= True):
    # Overview
    st.markdown("## **Overview** 📌")
    st.markdown("""
    **ArchIntelligent** is an end-to-end Generative AI solution for designing both **exterior** and **interior** of buildings.  
    It assists architects throughout their workflow, helping them save time, explore new ideas faster, and boost their creativity.
    """)

    overview_img = load_image("assets/images/meme1.png")
    st.image(overview_img, use_container_width=False, width= 700)

c1, c2= st.columns([0.4,0.6], border= True)

with c1:
    # Motivation
    st.markdown("### **Motivation** ✨")
    st.markdown("""
    Because I’m the guy who always strives to apply what I’ve learned into real world problems.  
    So I started asking people I knew, if they were facing any challenges in their work?

    It turned out that one of my high-school friends, who is majoring in architecture, came and said to me that...
    He was tired of paying for expensive AI architecture tools through monthly subscriptions on the internet...

    He asked if I could build a similar thing. If I could, he’d market my work to other students who also struggle financially.

    At that moment, I was honestly scared to accept the challenge. I had no idea where to start.
    But like I said, I love solving problems. So from that very night, I jumped straight into this project.
    """)

with c2:
    # Problem & Solution
    st.markdown("### The Problem & Solution 💡")
    st.markdown("""
    Imagine without AI today, how much work would an architect have to deal with every single day?  
    The answer? A ton.

    Architects handle so many tasks on a daily basis. click the button below to see:
    """)

    st.image(image="assets/images/architect_workflow.png", width= 550)

    st.markdown("""
    And the list goes on...

    The task that takes the most time-consuming in an architect’s workflow?  
    ➡️ **The Conceptual Design phase**

    That’s where an **AI architecture model** comes in handy — it generates multiple design variations in just **minutes**,
    dramatically speeding up the iteration process and making room for more creativity and less burnout.

    *To Do: Add a diagram of the architect workflow and client + solution*
    """)

st.text("\n")
st.text("\n")
st.text("\n")

with st.container(border= True):
    st.title("⚒️ ArchIntelligent’s Workflows")

    st.markdown("---")

    st.write("""
    To be honest, when I started this project, my knowledge of **Image Generation Models** was basically **0%…**

    Fortunately, I’d already built a strong foundation in machine/deep learning concepts through my previous project, **DeepNum**. I had the tools—I just needed to figure out how to use them to build something new.
    """)

    st.subheader("1. Research the Problem")

    st.write("""
    I began by searching and reading a lot of papers on AI architecture on the internet. But none of those came close to addressing my problem. Through this process, I discovered there are two main types of models used in image generation:
    """)

    st.markdown("""
    1. **Generative Adversarial Network (GAN)**  
    2. **Diffusion Models**
    """)

    st.write("""
    While reading papers, I also spent a week studying the 
    [GAN Specialization by DeepLearning.AI](https://www.coursera.org/specializations/generative-adversarial-networks-gans), 
    watching YouTube explanations, and reading blog posts—only to find out that GANs are quite outdated for my use case.

    GANs suffer from an issue called [**Mode Collapse**](https://en.wikipedia.org/wiki/Mode_collapse), where the model fails to generate diverse outputs and ends up producing very similar or even identical results.
    """)

    st.write("**So I decided to narrow my focus:**")

    st.markdown("""
    - ❌ Drop **interior design** (my friend only needed exterior)
    - ❌ Drop **GANs**
    - ✅ Focus solely on **exterior design** using **Diffusion Models**
    """)

    st.markdown("""
                And that’s when I stumbled across a paper that really helped me move in the right direction:

    📄 [**Generative AI-powered architectural exterior conceptual design based on the design intent**](https://academic.oup.com/jcde/article/11/5/125/7749580)
    """)

    st.write("""
    I wouldn’t call it the *holy grail*, but it introduced me to a whole world of concepts I’d never encountered before—things like **LoRA**, **ControlNet**, **DreamBooth**, and techniques for collecting and annotating datasets.

    It was a massive unlock.
    """)
    
    st.markdown("---")

    st.subheader("📦 2. Collecting, Preprocessing & Annotating Data")

    st.write("""
    Cool! I finally had a direction. Now what?

    (Ah yes, **data**. Without it, I basically couldn’t move on to the next phase.)

    But then came more questions:
    """)

    st.markdown("""
    - ❓ How do I collect images of buildings?  
    - ❓ What features should I consider when collecting them?  
    - ❓ Once I have the data, how do I preprocess and annotate it?  
    """)

    st.write("That’s a lot of questions that needed to be answered…")

    # Question 1
    st.markdown("##### 🔍 Question 1: How do I collect images of buildings?")

    st.write("""
    The paper I mentioned earlier briefly stated that I’d need to **crawl websites** to collect building images. Not exactly detailed guidance, but at least a starting point\
    After some more research, I discovered three popular architecture magazines offering high-quality images:
    """)

    st.markdown("""
    - 🌐 [ArchDaily](https://www.archdaily.com/)
    - 🌐 [Dezeen](https://www.dezeen.com/)
    - 🌐 [Dwell](https://www.dwell.com/)
    """)

    st.write("""
    I chose **ArchDaily** because it’s the most popular and updated daily. \
    But then I hit the next roadblock: *How do I crawl the site?* \
    The paper didn’t mention any specific tools. So, more research led me to a Python library called **Scrapy**. \
        Damn! I had to pause everything again and learn Scrapy.
    """)


    st.write("""
    I followed a [FreeCodeCamp tutorial](https://www.freecodecamp.org/news/use-scrapy-for-web-scraping-in-python/) and eventually figured out how to crawl ArchDaily and extract building images along with article text.
    But... because I only had a surface-level understanding of Scrapy, the website **blocked my IP** after a while 😓.

    To get around this, I switched to a visual scraping tool:  
    🔧 [**Octoparse**](https://www.octoparse.com/) — an absolute **life saver**.

    It let me scrape as many images as I needed **without getting blocked** and even ran scraping jobs on the **cloud**, so I could pull data in parallel. It saved me a lot of time and stress.
    """)
    st.caption("TODO: Insert meme image here")
    st.text("Hooray🎉🍾 I got the data, could I jump straight into the training phase? \nNOPE !! The data that I got at that time, looked messy and unrelated.")


    # Question 2
    st.markdown("##### 📐 Question 2: What features should I consider when collecting them?")

    st.write("""
    Before preprocessing, I had to understand which features were important and which weren’t.

    I narrowed them down into two main categories:
    """)

    st.markdown("""
    - 🏛️ **Architectural styles** (e.g. Modern, Classical, Art Deco, Baroque, Gothic)  
    - 🏠 **Building functions** (e.g. Residential, Apartment, Villa, Office, Restaurant)  
    """)

    # Question 3
    st.markdown("##### 🛠️ Question 3: How do I preprocess and annotate it?")

    st.write("""
    My dataset had around **15,000 images**, representing roughly **4,000 different buildings**.
    At this point, I got stuck for over **1.5 months**—trying and failing with every annotation method I found online.
    """)

    st.write("""
    If I had gone manual, it would’ve taken at least **6 months**, not counting all the possible errors along the way.
    So I went back into research mode—digging, testing, failing, and iterating. Eventually, I built a **data pipeline** that could automatically preprocess and annotate images.
    It saved me **a massive amount of time and effort**.\n
    """)

    # Placeholder for next part
    st.caption("👉 The dataset was a raw `.csv` file output from the scraping tool, containing the following:")
    
    # Load and cache the data
    df = load_building_data()
    st.dataframe(df, use_container_width=True)
    
    st.markdown("""
    I scraped **image URL links** because I didn’t pay for the premium version of Octoparse. But honestly, the workaround was pretty straightforward. I just wrote a small Python script using the **`requests`** library to automatically download all the images. ✅

    Now, for the **article texts**, the paper suggested that I should extract **keyword information** from two main categories:

    - **Client Needs** – keywords related to the client’s requirements, goals, or vision for the building described in the article.
    - **Architecture Language (AL)** – keywords tied to architectural design principles, terms, or stylistic language relevant to the building.

    **Note that these keywords from 2 categories will eventually merge into 1 line of text along with general image description (which i will cover later), will create a label for a single building.*

    But here’s the thing. H**ow was I supposed to extract those keywords just by reading architecture articles?** I’m not an architect. I don’t have the domain knowledge to know what counts as a "Client Need" or "Architecture Language" term.

    Those were the exact questions I had at the time… But I found that I could use ChatGPT API to automatically extract those keywords for me. Because ChatGPT has been trained on billions (if not trillions) of text samples. All I had to do was feed the article texts into the API and ask it to extract the relevant keywords for me. That solved the problem beautifully and only cost me about **$5 (~130,000 VND)** in API credits.

    For the images that i had downloaded. I needed to extract general description from those images. The authors of the paper used DreamBooth to extract image description. I did some research and found that the authors were wrong. DreamBooth is used for **fine-tuning** Stable Diffusion models on specific styles or subjects. It doesn’t extract descriptions from images.

    Fortunately, the **Hugging Face** community came through and I found the [**CLIP Interrogator**](https://huggingface.co/spaces/fffiloni/CLIP-Interrogator-2). It’s a model that takes in an image and **generates a detailed description** of what's in it. If you're curious, [this video explains it well](https://www.youtube.com/watch?v=KcSXcpluDe4&t=243s).

    ##### **The Final Label Structure**
    So, for each building image, I created a label made up of **three components**:

    1. **General Image Description** – generated by CLIP  
    2. **Architecture Language** – extracted from article text via ChatGPT  
    3. **Client Needs** – also extracted via ChatGPT  
    4. Functional  - from Dataframe  
    5. Styles - from Dataframe

    Each building folder containing multiple images from different angles and environments. While the **Architecture Language** and **Client Needs** stayed the same across images of the same building, the **General Description** varied depending on the image angle, lighting, and surroundings.
    """)