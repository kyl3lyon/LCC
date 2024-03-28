import streamlit as st
from app.pages import analytics, nimlet

# Define the pages
PAGES = {
    "Nimlet-1": nimlet,
    "Analytics": analytics
}

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Call the module associated with the chosen page
    page = PAGES[choice]
    page.show()

if __name__ == "__main__":
    main()
