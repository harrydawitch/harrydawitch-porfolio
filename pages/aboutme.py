import streamlit as st 
from streamlit_image_comparison import image_comparison

from PIL import Image


st.set_page_config(
                    page_title="HarryDaWitch",
                    layout= 'wide',
                    )
@st.cache_data
def load_image(path):
    return Image.open(path)

st.title(":blue[About Me]")

col1, col2= st.columns([0.5, 0.8], gap='small', vertical_alignment= 'center')

with col1:
    st.image(image= load_image("assets/images/harrydawitch.png"), use_container_width= True)
    
with col2:
    st.title("Chu Hoang Thien Long")
    st.markdown(
        "I'm passionate about solving problems with AI-driven solutions and am working toward my goal of becoming an AI engineer. \
         I bring a strong foundation in **deep learning, machine learning, and AI algorithms**, along with hands-on skills in scraping, \
         cleaning, and preprocessing data to support intelligent systems. I'm actively building my knowledge through personal projects \
         and self-studying related courses, which have deepened my understanding of AI and data. With solid programming knowledge, \
         including a good grasp of data structures and algorithms, I approach challenges with a data-driven mindset. As I start my career \
         in data-related roles, I'm eager to contribute to impactful projects and grow into a skilled AI engineer."
                )

        
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
#________________________________________________________Skills______________________________________________________________________________________________________

st.markdown("<h1 style='font-size: 2.5rem;'>💻 Skills & Expertise</h1>", unsafe_allow_html=True)
st.caption("My technical & other skills")

# ----- Custom CSS -----
st.markdown("""
    <style>
    .category-box {
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        background-color: var(--background-color);
    }
    .category-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--text-color);
    }
    .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .tag {
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid var(--text-color);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.95rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


# ----- Skill Data -----
skills = {
    "AI & Machine Learning": {
        "icon": "🧠",
        "tags": ["Neural Networks", "Deep Learning", "Generative AI", "Computer Vision", "LLM Fine-Tuning", "Causal Inference"]
    },
    "Programming": {
        "icon": "👨‍💻",
        "tags": ["Python", "Java",  "PyTorch", "Tensorflow", "LangChain", "Hugging Face", "Scikit-Learn", "CUDA", "Pandas", "NumPy"]
    },
    "Cloud Architecture & MLOps": {
        "icon": "☁️",
        "tags": ["AWS Services", "Docker/Kubernetes", "CloudFormation", "Vector DBs", "GPU Acceleration", "CI/CD", "Git"]
    },
    "Data Science": {
        "icon": "📊",
        "tags": ["Feature Engineering", "Dimensionality Reduction", "Clustering", "Statistical Modeling", "Data Visualization", "Data Analytics"]
    }
}

# ----- Layout: 2 Columns -----
col1, col2 = st.columns(2)

# loop over the skills and assign left/right alternately
for i, (category, data) in enumerate(skills.items()):
    with (col1 if i % 2 == 0 else col2):
        st.markdown(f"<div class='category-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='category-title'>{data['icon']} {category}</div>", unsafe_allow_html=True)
        st.markdown("<div class='tags'>" + "".join(
            [f"<span class='tag'>{tag}</span>" for tag in data["tags"]]
        ) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")

        
#________________________________________________________________________________________________________________________________________________________________________________
st.title("**Projects** ⚒️")
st.caption("My independent projects & contributions")
tab1, tab2= st.tabs(["ArchIntelligent", "DeepNum"])
tools= ["Python", "Pytorch", "Pandas", "Numpy", "Scrapy", "CLIP", "GPT-4 API", "Diffusers", "Huggingface", "LoRA", "ControlNet", "Streamlit"]


with tab1:
    st.write("""
                This is the end to end Generative AI solution project that was inspired by my best friend, who majoring in architecture. Tired of paying for design tool subscriptions,
                one day he then asked me if i can build such a tool that could help him and many architect students to have a free AI model for generating design concepts more efficiently — without paying monthly fees.\
                I took the idea and then started working on this project by researching the problem, reading papers, learning new tools. And after 4 months of hard-working, this is the result.
                """)
    p1_1, p1_2 = st.columns([0.5, 0.5], vertical_alignment= "top", border= True)
    
    with p1_1:
        image_comparison(
                        img1=load_image("assets/images/image_1.jpg"),
                        img2=load_image("assets/images/image_2.png"),
                        label1="Before",
                        label2="After",
                        width= 512
                         )

        st.subheader("🛠️ **Tools & Technologies**")
        st.pills(
                label= "Tools",
                label_visibility= "collapsed",
                options= tools,
                disabled= False,
                selection_mode= "multi",
                default= tools,
                   
                 )        
        st.write("\n")
        st.subheader("🎯 **Impact for me & people**")
        st.markdown("""
                    -  My friend and other architecture students no longer need to rely on expensive subscription-based model. My Generative AI model gives them creative freedom, for free.
                    -  Sharpened my skills in data scraping, data preprocessing, model fine-tuning, API integration, and deployment. Expanding both my technical and practical toolkit.
                    -  Real-world project experience — this wasn't just a toy project. It was a practical project, I'm finally providing value for people.
                    -  Learning how **Stable Diffusion**, **LoRA**, and **ControlNet** work under the hood, not just at surface level.
                    """)
    
    with p1_2:
        
        st.subheader("🧠 **Key Features & Workflow**")
        st.markdown("""
                    - **🖼️ Web Scraping Architectural Images**  
                    Scraped thousands of architectural design references using **Scrapy**, covering various styles and perspectives.

                    - **🧹 Data Cleaning & Preprocessing**  
                    Cleaned and preprocessed image-text datasets using **OpenCV**, **NumPy**, and **Pandas** to prepare for training.

                    - **⚙️ Automated Annotation with GPT & CLIP**  
                    Built a semi-automated pipeline using the **OpenAI GPT API** and **CLIP model** to annotate and describe images — reducing manual labeling by hand efforts.

                    - **🎨 Fine-Tuning Stable Diffusion XL via LoRA and integrated ControlNet**  
                    Used **LoRA** to fine-tune **Stable Diffusion XL** on architecture-specific datasets for better control and relevance. Added **ControlNet** to enable sketch-to-image functionality, allowing users to upload rough architectural sketches and generate complete visuals.

                    - **☁️ Deployment**  
                    I originally planned for AWS, but due to budget constraints, i deployed the model on **Hugging Face Spaces** with a CPU-based environment — althought CPU still quite slow, people can also clone my repo and generate image through Google Colab T4 GPU (Free version)

                    - **🖥️ Streamlit UI**  
                    Built an interactive **Streamlit** app that enables users to input text prompts or sketches and receive AI-generated design concepts in real time.
                    """)
    if st.button("# **View project in detail**",
                 type='primary', 
                 use_container_width= True):
        st.switch_page(page="pages/ArchIntelligent.py")
        
with tab2:
    st.header("TO DO: Write an overview of DeepNum")
        
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
#_______________________________________CERTIFICATIONS__________________________________________________________

certifications = [
    {
        "name": "Deep Learning Specialization",
        "issuer": "DeepLearning.AI",
        "logo": "https://remoteworksource.com/wp-content/uploads/2022/01/DeepLearning-AI.jpg",
        "date": "Oct 2024",
        "url": "https://coursera.org/share/56cc1377cb18f3f1f0be27099160d1f7"
    },
    {
        "name": "IBM Data Science Professional Certificate",
        "issuer": "IBM",
        "logo": "https://www.ibm.com/brand/experience-guides/developer/b1db1ae501d522a1a4b49613fe07c9f1/01_8-bar-positive.svg",
        "date": "March 2024",
        "url": "https://coursera.org/share/b4fc66aed12dc163d831dce582a65c65"
    },
    {
        "name": "Machine Learning Specialization",
        "issuer": "Stanford University",
        "logo": "https://identity.stanford.edu/wp-content/uploads/sites/3/2020/07/block-s-right.png",
        "date": "June  2024",
        "url": "https://coursera.org/share/4d937915d8e6f031861f44c8956f451e"
    },
    {
        "name": "AI & Machine Learning, Data Science Bootcamp",
        "issuer": "Zero to Mastery Academy",
        "logo": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoqj--xVXR-ebpeGz9YnC59DyHKGaqzA7G4Q&s",
        "date": "July 2024",
        "url": "https://www.udemy.com/certificate/UC-bb24d9e2-6ee9-4ded-8d5a-cc47c28ec897/  "
    },
    {
        "name": "Data Structures and Algorithms Specialization",
        "issuer": "UC San Diego",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Seal_of_the_University_of_California%2C_San_Diego.svg/800px-Seal_of_the_University_of_California%2C_San_Diego.svg.png",
        "date": "July 2024",
        "url": "https://coursera.org/share/ef3f9e12210a8c527b64a999e5b86b82"
    },
    {
        "name": "Mathematic for Machine Learning and Data Science Specialization",
        "issuer": "DeepLearning.AI",
        "logo": "https://remoteworksource.com/wp-content/uploads/2022/01/DeepLearning-AI.jpg",
        "date": "April 2024",
        "url": "https://coursera.org/share/1c924cbfaed99345d5a62a345da201ff"
    }
    
]

st.markdown("""
<style>
.cert-container {
    border: 1px solid #444;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    background-color: #111;
    display: flex;
    align-items: center;
    gap: 15px;
}
.cert-logo {
    width: 50px;
    height: 50px;
    object-fit: contain;
    border-radius: 6px;
    background-color: white;
    padding: 4px;
}
.cert-content {
    flex: 1;
}
.cert-title {
    font-size: 1.1rem;
    font-weight: bold;
}
.cert-meta {
    font-size: 0.9rem;
    color: #aaa;
}
.cert-link a {
    color: #1f77b4;
    text-decoration: none;
    font-size: 0.9rem;
}
.cert-link a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.title("Certifications 📜")

with st.expander(" ", expanded= True):
    for cert in certifications:
        st.markdown(f"""
        <div class="cert-container">
            <img src="{cert['logo']}" class="cert-logo">
            <div class="cert-content">
                <div class="cert-title">🏅 {cert['name']}</div>
                <div class="cert-meta">🎓 {cert['issuer']} &nbsp;&nbsp; 📅 {cert['date']}</div>
                <div class="cert-link">🔗 <a href="{cert['url']}" target="_blank">View Certificate</a></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")


#___________________________________________Education____________________________________________________________________
st.markdown("---")

st.markdown("## 🎓 Education")
st.caption("My journey in the academic")

st.write("\n")
st.write("\n")
st.write("\n")

# Load university logo
iuh_logo = load_image("assets/images/Logo-IUH.jpg")  # Replace with the correct path to your logo image

# --- Column layout for university ---
col1, col2 = st.columns([1, 8])
with col1:
    st.image(iuh_logo, width=80)
with col2:
    st.markdown("**Industrial University of Ho Chi Minh City (IUH)**")
    st.markdown("*B.Eng. in Electrical Engineering and Automation (Incomplete)*")
    st.markdown("📍 *Ho Chi Minh City, Vietnam* &nbsp;&nbsp;&nbsp; 📅 *2021 – 2023*")
    st.markdown("_Paused university studies in the 3rd year due to financial hardship to support my family. Since then, I've actively pursued self-education in AI and data Science._")

# --- Spacer ---
st.markdown("---")

# --- Online Learning ---
col3, col4 = st.columns([1, 8])
with col3:
    st.image(load_image("assets/images/online_study.jpg"), width=80)  # Add an icon or placeholder image
with col4:
    st.markdown("**Online Learning**")
    st.markdown("*Self-taught in AI, ML, and Data Engineering*")
    st.markdown("📍 *Remote (Coursera, edX, etc.)* &nbsp;&nbsp;&nbsp; 📅 *2023 – Present*")
    st.markdown("_Completed specializations from Stanford, DeepLearning.AI, and AWS._")
