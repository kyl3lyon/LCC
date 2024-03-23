import streamlit as st
from app.pages import overview, clouds, precipitation, lightning, analytics

# Define the pages
PAGES = {
    "Overview": overview,
    "Clouds": clouds,
    "Precipitation": precipitation,
    "Lightning": lightning,
    "Analytics": analytics
}

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Go to", list(PAGES.keys()))

    # Call the module associated with the chosen page
    page = PAGES[choice]
    page.show()

if __name__ == "__main__":
    main()
