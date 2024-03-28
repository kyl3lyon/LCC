import streamlit as st
from app.pages import nimbus, analytics

# Define the pages
PAGES = {
    "Nimbus-1": nimbus,
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
