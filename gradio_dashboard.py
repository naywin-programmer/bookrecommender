import pandas as pd
import numpy as np
import chromadb
import gradio as gr

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

books = pd.read_csv("books_with_emotion_data.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isnull(), "cover-not-found.jpg", books["large_thumbnail"])

raw_documents = TextLoader("tagged_description.txt", encoding="utf8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

db_path = "Production_BookRecommenderDB"
client = chromadb.PersistentClient(path=db_path)

# Before creating vector database, run this for the first time
# db_books = Chroma.from_documents(documents, embedding=OllamaEmbeddings(model="nomic-embed-text:latest"), client=client, persist_directory=db_path)

# After creating vector database, run this
db_books = Chroma(persist_directory=db_path, embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"))

emotion_selection = ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


def retrieve_sematic_recommendations(
    query: str,
    category: str = None,
    emotion: str = None,
    inital_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    search_results = db_books.similarity_search(query, k=inital_top_k)
    books_isbn_list = [int(each.page_content.strip('"').split(":")[0]) for each in search_results]
    book_results = books[books["isbn13"].isin(books_isbn_list)].head(final_top_k)
    
    if category.lower() == "all":
        book_results = book_results.head(final_top_k)
    else:
        book_results = book_results[book_results["simple_categories"] == category].head(final_top_k)
    
    # "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
    if emotion == emotion_selection[0]:
        book_results.sort_values(by="joy", ascending=False, inplace=True)
    elif emotion == emotion_selection[1]:
        book_results.sort_values(by="surprise", ascending=False, inplace=True)
    elif emotion == emotion_selection[2]:
        book_results.sort_values(by="anger", ascending=False, inplace=True)
    elif emotion == emotion_selection[3]:
        book_results.sort_values(by="fear", ascending=False, inplace=True)
    elif emotion == emotion_selection[4]:
        book_results.sort_values(by="sadness", ascending=False, inplace=True)
    
    return book_results


def recommend_books(
    query: str,
    category: str,
    emotion: str
):
    recommendations = retrieve_sematic_recommendations(query, category, emotion)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."
        
        authors_arr = row["authors"].split(";")
        if len(authors_arr) == 2:
            authors_str = f"{authors_arr[0]} and {authors_arr[1]}"
        elif len(authors_arr) > 2:
            authors_str = f"{', '.join(authors_arr[:-1])}, and {authors_arr[-1]}"
        else:
            authors_str = row["authors"]
            
        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    
    return results


# Gradio UI
categories = ["All"] + sorted(books["simple_categories"].unique())
emotions = ["All"] + emotion_selection

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommendater")
    
    with gr.Row():
        user_query = gr.Textbox(label="Search Book:", placeholder="eg. A story about the fun journey")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        emotion_dropdown = gr.Dropdown(choices=emotions, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")
        
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", rows=2, columns=4)
    
    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, emotion_dropdown],
        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
    # dashboard.launch(share=True)