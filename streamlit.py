import streamlit as st
import json
import random
import base64
import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from chatbot import NMCCChatbot

st.set_page_config(
    page_title="NMCC Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "Home"


@st.cache_data
def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()



logo_base64 = get_base64_image("logo.png")
image_base64 = get_base64_image("Nesamony.jpg")
pages = ["Home", "About", "Chatbot", "Contact"]

if hasattr(st, "query_params"):
    query_page = st.query_params.get("page")
else:
    query_page = st.experimental_get_query_params().get("page", [None])

if isinstance(query_page, list):
    query_page = query_page[0] if query_page else None

if query_page in pages:
    st.session_state.page = query_page

current_page = st.session_state.page if st.session_state.page in pages else "Home"
st.session_state.page = current_page

menu_links = []
for label in pages:
    active_class = "active" if label == current_page else ""
    menu_links.append(
        f'<a class="nav-link {active_class}" href="?page={label}" target="_self">{label}</a>'
    )

nav_links_html = " ".join(menu_links)

# ================= HEADER + MENU BAR =================
st.markdown(f"""
<style>
/* Remove Streamlit UI */
header {{visibility: hidden;}}
footer {{visibility: hidden;}}
.block-container {{padding-top: 0rem;}}

/* FONT */
.stApp {{
    font-family: "Segoe UI", sans-serif;
}}

/* ================= MAIN HEADER ================= */
.utility-bar {{
    background-color: #1f9e9a;
    height: 30px;
    width: 100%;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 14px;
    padding: 0 60px;
    font-size: 13px;
    color: white;
    box-sizing: border-box;
}}

.utility-bar a {{
    color: white;
    text-decoration: none;
    font-weight: 600;
}}

.utility-bar .sep {{
    opacity: 0.7;
}}

.main-header {{
    background-color: #27b3a7;
    padding: 32px 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
}}

.header-left {{
    display: flex;
    align-items: center;
}}

.header-left img {{
    width: 90px;
    margin-right: 20px;
}}

.college-name {{
    color: white;
    font-size: 30px;
    font-weight: 800;
}}

.college-sub {{
    color: white;
    font-size: 14px;
    margin-top: 6px;
}}

/* Ribbon */
.ribbon {{
    background-color: #f2b91e;
    padding: 14px 24px;
    font-size: 14px;
    font-weight: 400;
    color: black;
    clip-path: polygon(6% 0, 100% 0, 100% 100%, 0 100%);
    overflow: hidden;
    white-space: nowrap;
}}

.header-right {{
    position: absolute;
    right: 60px;
    bottom: 20px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
}}

.profile-photo {{
    width: 80px;
    height: 80px;
    border-radius: 50%;
    object-fit: cover;
    border: none;
    border: 2px solid transparent;
    display: block;
    
}}

.ribbon-text {{
    display: inline-block;
    white-space: nowrap;
    animation: ribbon-scroll 20s linear infinite;
}}

@keyframes ribbon-scroll {{
    0% {{
        transform: translateX(100%);
    }}
    100% {{
        transform: translateX(-100%);
    }}
}}

/* Nav bar under header (same setup as upper bar) */
.nav-bar {{
    background-color: #1f9e9a;
    height: 30px;
    width: 100%;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 14px;
    padding: 0 60px;
    font-size: 15px;
    color: white;
    box-sizing: border-box;
}}

.nav-link {{
    color: white;
    text-decoration: none !important;
    font-weight: 600;
    padding: 0 6px;
    outline: none;
    border-bottom: none;
    box-shadow: none;
}}

.nav-link:link,
.nav-link:visited,
.nav-link:hover,
.nav-link:active,
.nav-link:focus,
.nav-link.active {{
    color: white;
    text-decoration: none !important;
    outline: none;
    border-bottom: none;
    box-shadow: none;
}}
</style>

<div class="utility-bar"></div>
<div class="main-header">
    <div class="header-left">
        <img src="data:image/png;base64,{logo_base64}">
        <div>
            <div class="college-name">NESAMONY MEMORIAL CHRISTIAN COLLEGE</div>
            <div class="college-sub">
                [ESTD:1964, Administrated by CSI Kanniyakumari Diocese]<br>
                Affiliated with Manonmaniam Sundaranar University
            </div>
        </div>
    </div>
    <div class="header-right">
        <img class="profile-photo" src="data:image/png;base64,{image_base64}">
        <div class="ribbon">
            <span class="ribbon-text">NMCC Ranked 63rd by NIRF All India Ranking 2025, Govt. of India.</span>
        </div>
    </div>
</div>
<div class="nav-bar">
    {nav_links_html}
</div>
""", unsafe_allow_html=True)

page = current_page

@st.cache_resource
def load_bot():
    return NMCCChatbot()
if "bot" not in st.session_state:
    st.session_state.bot = load_bot()
    st.session_state.chat_history = []
    st.session_state.chat_input = ""


def process_message():
    user_input = st.session_state.chat_input
    if user_input.strip():
        reply = st.session_state.bot.get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", reply))
        st.session_state.chat_input = ""  # clear after sending


def reset_chat():
    st.session_state.chat_history = []
    st.session_state.chat_input = ""


if page == "Chatbot":
    st.title("NMCC College Chatbot")

    # Display chat history
    for sender, message in st.session_state.chat_history:
        bubble_color = "#f0f0f0" if sender == "You" else "#27b3a7"
        text_color = "black" if sender == "You" else "white"
        # align = "right" if sender == "You" else "left"
        align = "left"
        st.markdown(
            f"<div style='background-color:{bubble_color}; color:{text_color}; "
            f"padding:10px; border-radius:8px; margin-bottom:5px; text-align:{align};'>"
            f"<b>{sender}:</b> {message}</div>",
            unsafe_allow_html=True
        )
         # Input box with Enter support
    st.text_input("Ask something:", key="chat_input", on_change=process_message)

    # Reset chat
    st.button("🔄", on_click=reset_chat)




elif page == "About":
    st.title("About NMCC")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
    <div style="text-align: justify; font-size:16px; line-height:1.6; margin:0; padding:0;">
    <b>Nesamony Memorial Christian College (NMCC)</b> is a prestigious academic and research institution
    that fosters both academic and personal development by providing a highly professional and
    value-based education. The college, located on a lush green campus, is equipped with excellent
    infrastructure including a modern library, indoor and outdoor stadiums, hostels, and internet facilities.
    It offers programs in science, humanities, commerce, and management, supported by well-equipped
    department libraries and classrooms. Dedicated and erudite faculty members guide students to become
    world-class citizens in today’s globally competitive environment.

    NMCC traces its origins to land purchased by Rev. James Emlyn of the London Missionary Society,
    later developed by Rev. Robert Sinclair in 1910 with a bungalow and boarding school. Following India’s
    independence, the Kanniyakumari Diocese of the Church of South India recognized the need for higher
    education in Marthandam and, in 1964, established the college with 32 acres of endowment land.
    An ad-hoc committee led by Mr. N. Dennis, Ex. MP, raised funds with strong support from the local
    community, parents, students, and staff.

    The college began with a Pre-University Class in 1964 under the leadership of its first Principal,
    Dr. John D.K. Sundar Singh, and was inaugurated by Bishop Rt. Rev. I.R.H. Gnanadason. Initially a men’s
    institution, it became co-educational in 1977. Undergraduate programs in Mathematics, History, and
    Economics were introduced in 1965, and postgraduate studies commenced in 1980–81 with M.Sc. Physics.
    A Ph.D. Research Centre in History was established in 1997. Today, the college offers a wide range of
    programs: 20 undergraduate, 15 postgraduate, 12 M.Phil., 12 Ph.D., and 11 certificate courses, serving
    over 4,000 students with nearly 280 staff members.

    Affiliated first with Madurai University in 1966, the college later joined Manonmaniam Sundaranar
    University in 1991. In 1984, it was renamed Nesamony Memorial Christian College in honor of
    Thiru. A. Nesamony, whose contributions to the Diocese were significant. The institution was accredited
    with an ‘A’ grade by NAAC for 2014–2019 and continues to pursue innovative programs aimed at uplifting
    the local community. Guided by faith and public support, NMCC remains a distinguished center of learning
    in the region.
    </div>
    """, unsafe_allow_html=True)
        
    with col2:
        st.image(
            "Screenshot 2026-02-02 164229.png",
            caption="Nesamony",
            width=350  # 👈 adjust this value (300–500 looks good)
        )

elif page == "Contact":
    st.header("Contact Nesamony Memorial Christian College")
    st.write("""
    📍 **Address:** Marthandam – 629165,Kanyakumari District,Tamil Nadu, INDIA  
    📞 **Phone:** 9443370257  
    ✉️ **Email:** principalnmcc2014@gmail.com  
    🌐 **Website:** [www.nmcc.ac.in](https://nmcc.ac.in/Default.aspx)  
    📌 **Location:** [View on Google Maps](https://www.google.com/maps/place/Nesamony+Memorial+Christian+College/@8.308128,77.221224,14z/data=!4m6!3m5!1s0x3b045519d8dab465:0xda2ed8db101afe90!8m2!3d8.3081275!4d77.2212235!16zL20vMDc2NTgx?hl=en&entry=ttu&g_ep=EgoyMDI2MDEyOC4wIKXMDSoASAFQAw%3D%3D)  
    """)

else:
    st.title("Welcome to NMCC")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            "clg.webp",
            caption="Nesamony Memorial Christian College",
            width=880
        )

    with col2:
        st.image(
            "Screenshot 2026-02-02 180018.png",
            width=450
        )

st.markdown("</div>", unsafe_allow_html=True)







