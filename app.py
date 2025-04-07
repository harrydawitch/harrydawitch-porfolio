import streamlit as st



aboutme_page = st.Page(
                        page= "pages/aboutme.py",
                        title= "About me",
                        icon=":material/person:",
                        default= True
                       )

arch_intel_page= st.Page(
                        page= "pages/ArchIntelligent.py",
                        title= "ArchIntelligent",
                        icon= ":material/villa:"
                         )

deep_num_page= st.Page(
                        page= "pages/deepnum.py",
                        title= "DeepNum",
                        icon= ":material/grid_guides:"
                       )

pages = st.navigation({
                        "Home": [aboutme_page],
                        "Projects": [arch_intel_page, deep_num_page]
                       }
                      )

pages.run()