import streamlit as st

# Title
st.title("Universal URL Research Tool")

# Step 1: Input number of URLs
num_urls = st.number_input("How many URLs do you want to index?", min_value=1, max_value=20, step=1)

# Step 2: Dynamic URL input fields
urls = []
for i in range(num_urls):
    url = st.text_input(f"URL {i + 1}")
    urls.append(url)

# Step 3: Index Sources Button
if st.button("Index Sources"):
    if all(urls):
        st.success("URLs are ready to be indexed!")
        # Placeholder for indexing logic
    else:
        st.error("Please fill in all URL fields before indexing.")

# Step 4: Question Input
question = st.text_input("Ask a question based on the indexed URLs")

# Step 5: Results Display
if question:
    st.write("Answer:", "[Placeholder for answer]")
    st.write("Sources:", "[Placeholder for sources]")