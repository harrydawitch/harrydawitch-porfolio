import streamlit as st 
from PIL import Image

st.set_page_config(
                    page_title="Projects | ArchIntelligent",
                    layout= 'wide',
                    )
@st.cache_data
def load_image(path):
    return Image.open(path)

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

    Architects juggle so many tasks on a daily basis. Let’s take a look:
    """)

    st.image(image=load_image("assets/images/architect_workflow.png"), width= 550)

    st.markdown("""
    And the list goes on...

    The task that takes the most time-consuming in an architect’s workflow?  
    ➡️ **The Conceptual Design phase**

    That’s where an **AI architecture model** comes in handy — it generates multiple design variations in just **minutes**,
    dramatically speeding up the iteration process and making room for more creativity and less burnout.

    *To Do: Add a diagram of the architect workflow and client + solution*
    """)

    

