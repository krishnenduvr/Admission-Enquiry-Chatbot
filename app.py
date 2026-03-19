import base64
import html
from pathlib import Path
from typing import Dict

import streamlit as st  # type: ignore

import py2

st.set_page_config(
    page_title="NMCC Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ASSET_DIR = Path(__file__).resolve().parent
PAGES = ["Home", "About", "Chatbot", "Contact"]
WELCOME_MESSAGE = "Hello! Ask me anything about the NMCC handbook."

if "page" not in st.session_state:
    st.session_state.page = "Home"


@st.cache_data(show_spinner=False)
def get_base64_image(path: str) -> str:
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


@st.cache_resource(show_spinner=False)
def load_base_state() -> Dict[str, object]:
    chunks = py2.load_data()
    source_text = py2.load_source_text(chunks)
    heading_sections = py2.build_heading_sections(source_text)
    subheading_sections = py2.build_subheading_sections(source_text, heading_sections)
    heading_index = py2.build_heading_index_from_text(source_text)
    intents = py2.load_intents()
    return {
        "chunks": chunks,
        "source_text": source_text,
        "heading_sections": heading_sections,
        "subheading_sections": subheading_sections,
        "heading_index": heading_index,
        "intents": intents,
    }


def answer_query(query: str, state: Dict[str, object]) -> str:
    return py2.answer_query(
        query=query,
        source_text=state["source_text"],
        heading_sections=state["heading_sections"],
        subheading_sections=state["subheading_sections"],
        heading_index=state["heading_index"],
        intents=state["intents"],
    )


def resolve_asset(name: str) -> str:
    path = ASSET_DIR / name
    return str(path)


logo_base64 = get_base64_image(resolve_asset("logo.png"))
image_base64 = get_base64_image(resolve_asset("Nesamony.jpg"))

if hasattr(st, "query_params"):
    query_page = st.query_params.get("page")
else:
    query_page = st.experimental_get_query_params().get("page", [None])

if isinstance(query_page, list):
    query_page = query_page[0] if query_page else None

if query_page in PAGES:
    st.session_state.page = query_page

current_page = st.session_state.page if st.session_state.page in PAGES else "Home"
st.session_state.page = current_page

menu_links = []
for label in PAGES:
    active_class = "active" if label == current_page else ""
    menu_links.append(
        f'<a class="nav-link {active_class}" href="?page={label}" target="_self">{label}</a>'
    )
nav_links_html = " ".join(menu_links)

st.markdown(
    f"""
    <style>
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 0rem; padding-left: 0rem; padding-right: 0rem;}}
    .stApp {{
        font-family: "Segoe UI", sans-serif;
        background: #ffffff;
    }}
    .main-content {{
        padding: 28px 60px 40px 60px;
    }}
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
        border: 2px solid transparent;
        display: block;
    }}
    .ribbon {{
        background-color: #f2b91e;
        padding: 14px 24px;
        font-size: 14px;
        font-weight: 400;
        color: black;
        clip-path: polygon(6% 0, 100% 0, 100% 100%, 0 100%);
        overflow: hidden;
        white-space: nowrap;
        max-width: 420px;
    }}
    .ribbon-text {{
        display: inline-block;
        white-space: nowrap;
        animation: ribbon-scroll 20s linear infinite;
    }}
    @keyframes ribbon-scroll {{
        0% {{ transform: translateX(100%); }}
        100% {{ transform: translateX(-100%); }}
    }}
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
    .chat-wrap {{
        background: #f7fbfb;
        border: 1px solid #d7ecea;
        border-radius: 14px;
        padding: 12px;
        margin-top: 0;
        margin-bottom: 0.5rem;
        max-height: 280px;
        overflow-y: auto;
    }}
    .chat-bubble {{
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        text-align: left;
        line-height: 1.6;
    }}
    .chat-bubble.user {{
        background-color: #f0f0f0;
        color: black;
    }}
    .chat-bubble.bot {{
        background-color: #27b3a7;
        color: white;
    }}
    .chat-actions {{
        margin-top: 4px;
    }}
    .stButton > button {{
        background-color: #27b3a7;
        color: white;
        border: none;
        border-radius: 8px;
    }}
    .stButton > button:hover {{
        background-color: #1f9e9a;
        color: white;
        border: none;
    }}
    div[data-testid="stHorizontalBlock"] {{
        gap: 0.5rem;
        margin-bottom: 0.25rem;
        align-items: flex-start;
    }}
    div[data-testid="stTextInput"] {{
        margin-bottom: 0;
    }}
    div[data-testid="stTextInput"] > div {{
        margin-bottom: 0;
    }}
    div[data-testid="stTextInput"] > div > div {{
        background: #ffffff;
        border: 1px solid #cfe1df;
        border-radius: 10px;
        box-shadow: none;
    }}
    div[data-testid="stTextInput"] [data-baseweb="base-input"] {{
        background: #ffffff !important;
        border: 1px solid #cfe1df !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }}
    div[data-testid="stTextInput"] [data-baseweb="base-input"] > div {{
        background: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }}
    div[data-testid="stTextInput"] [data-baseweb="input"] {{
        background: #ffffff !important;
    }}
    div[data-testid="stTextInput"] input {{
        background: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }}
    div[data-testid="stTextInput"] label {{
        display: none;
    }}
    div[data-testid="column"] {{
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }}
    .reset-row .stButton > button {{
        width: 120px;
    }}
    @media (max-width: 900px) {{
        .main-header {{
            padding: 20px 20px 90px 20px;
            align-items: flex-start;
        }}
        .utility-bar, .nav-bar {{
            padding: 0 20px;
        }}
        .header-right {{
            right: 20px;
            bottom: 12px;
        }}
        .main-content {{
            padding: 20px;
        }}
        .college-name {{
            font-size: 22px;
        }}
        .header-left img {{
            width: 64px;
        }}
        .ribbon {{
            max-width: 260px;
            padding: 10px 18px;
            font-size: 12px;
        }}
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
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
def process_message(user_input: str) -> None:
    user_input = user_input.strip()
    if not user_input:
        return
    state = load_base_state()
    try:
        reply = answer_query(user_input, state)
    except Exception as exc:
        reply = f"Error: {type(exc).__name__}"
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", reply))


def reset_chat() -> None:
    st.session_state.chat_history = []


st.markdown('<div class="main-content">', unsafe_allow_html=True)

if current_page == "Chatbot":
    title_col, action_col = st.columns([6, 1])
    with title_col:
        st.markdown(
            "<h1 style='margin: 0 0 8px 0; font-size: 2.1rem;'>NMCC College Chatbot</h1>",
            unsafe_allow_html=True,
        )
    with action_col:
        st.markdown('<div class="reset-row">', unsafe_allow_html=True)
        st.button("Reset Chat", on_click=reset_chat, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for sender, message in st.session_state.chat_history:
            bubble_class = "user" if sender == "You" else "bot"
            safe_message = html.escape(message).replace("\n", "<br>")
            st.markdown(
                f"<div class='chat-bubble {bubble_class}'><b>{sender}:</b> {safe_message}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    prompt = st.chat_input("Ask something...")
    if prompt:
        process_message(prompt)
        st.rerun()

elif current_page == "About":
    st.title("About NMCC")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            """
            <div style="text-align: justify; font-size:16px; line-height:1.6; margin:0; padding:0;">
            <b>Nesamony Memorial Christian College (NMCC)</b> is a prestigious academic and research institution
            that fosters both academic and personal development by providing a highly professional and
            value-based education. The college, located on a lush green campus, is equipped with excellent
            infrastructure including a modern library, indoor and outdoor stadiums, hostels, and internet facilities.
            It offers programs in science, humanities, commerce, and management, supported by well-equipped
            department libraries and classrooms. Dedicated and erudite faculty members guide students to become
            world-class citizens in today's globally competitive environment.

            NMCC traces its origins to land purchased by Rev. James Emlyn of the London Missionary Society,
            later developed by Rev. Robert Sinclair in 1910 with a bungalow and boarding school. Following India's
            independence, the Kanniyakumari Diocese of the Church of South India recognized the need for higher
            education in Marthandam and, in 1964, established the college with 32 acres of endowment land.
            An ad-hoc committee led by Mr. N. Dennis, Ex. MP, raised funds with strong support from the local
            community, parents, students, and staff.

            The college began with a Pre-University Class in 1964 under the leadership of its first Principal,
            Dr. John D.K. Sundar Singh, and was inaugurated by Bishop Rt. Rev. I.R.H. Gnanadason. Initially a men's
            institution, it became co-educational in 1977. Undergraduate programs in Mathematics, History, and
            Economics were introduced in 1965, and postgraduate studies commenced in 1980-81 with M.Sc. Physics.
            A Ph.D. Research Centre in History was established in 1997. Today, the college offers a wide range of
            programs: 20 undergraduate, 15 postgraduate, 12 M.Phil., 12 Ph.D., and 11 certificate courses, serving
            over 4,000 students with nearly 280 staff members.

            Affiliated first with Madurai University in 1966, the college later joined Manonmaniam Sundaranar
            University in 1991. In 1984, it was renamed Nesamony Memorial Christian College in honor of
            Thiru. A. Nesamony, whose contributions to the Diocese were significant. The institution was accredited
            with an 'A' grade by NAAC for 2014-2019 and continues to pursue innovative programs aimed at uplifting
            the local community. Guided by faith and public support, NMCC remains a distinguished center of learning
            in the region.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.image(resolve_asset("Screenshot 2026-02-02 164229.png"), caption="Nesamony", width=350)

elif current_page == "Contact":
    st.header("Contact Nesamony Memorial Christian College")
    st.write(
        """
        **Address:** Marthandam - 629165, Kanyakumari District, Tamil Nadu, India

        **Phone:** 9443370257

        **Email:** principalnmcc2014@gmail.com

        **Website:** [www.nmcc.ac.in](https://nmcc.ac.in/Default.aspx)

        **Location:** [View on Google Maps](https://www.google.com/maps/place/Nesamony+Memorial+Christian+College/@8.308128,77.221224,14z/data=!4m6!3m5!1s0x3b045519d8dab465:0xda2ed8db101afe90!8m2!3d8.3081275!4d77.2212235!16zL20vMDc2NTgx)
        """
    )

else:
    st.title("Welcome to NMCC")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            resolve_asset("clg.webp"),
            caption="Nesamony Memorial Christian College",
            use_container_width=True,
        )

    with col2:
        st.image(resolve_asset("Screenshot 2026-02-02 180018.png"), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
