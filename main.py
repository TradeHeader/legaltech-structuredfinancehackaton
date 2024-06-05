import base64
from io import BytesIO

import streamlit as st
from py_avataaars import PyAvataaar, AvatarStyle, SkinColor, HairColor, FacialHairType, TopType, ClotheType, Color

from src.embedding.multiModalEmbeddingWrapper import MultimodalEmbeddingWrapper
from src.query.QueryProcessor import QueryProcessor
from src.store.pgVector.multiModalPGVectorStore import MultimodalPGVectorStore


def initialize_query_processor():
    # Initialize your PGVectorStore and QueryProcessor
    connection_string = "postgresql+psycopg2://..."
    collection_name = "fpml-documents"
    embedding_wrapper = MultimodalEmbeddingWrapper(chunk_size=1000)
    pg_vector_store = MultimodalPGVectorStore(
        connection_string=connection_string,
        embedding_wrapper=embedding_wrapper,
        collection_name=collection_name
    )
    query_processor = QueryProcessor(pg_vector_store)
    return query_processor


def generate_avatar(style, skin_color, hair_color, facial_hair_type, top_type, clothe_type, clothe_color):
    avatar = PyAvataaar(
        style=style,
        skin_color=skin_color,
        hair_color=hair_color,
        facial_hair_type=facial_hair_type,
        top_type=top_type,
        clothe_type=clothe_type,
        clothe_color=clothe_color
    )
    img = BytesIO()
    avatar.render_png_file(img)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def main():
    st.title("Chatbot Interface")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = []

    if 'queries' not in st.session_state:
        st.session_state['queries'] = []

    if 'user_avatar' not in st.session_state:
        st.session_state['user_avatar'] = generate_avatar(
            AvatarStyle.CIRCLE, SkinColor.LIGHT, HairColor.BROWN,
            FacialHairType.DEFAULT, TopType.SHORT_HAIR_SHORT_FLAT,
            ClotheType.GRAPHIC_SHIRT, Color.HEATHER)

    if 'bot_avatar' not in st.session_state:
        st.session_state['bot_avatar'] = generate_avatar(
            AvatarStyle.CIRCLE, SkinColor.DARK_BROWN, HairColor.BLACK,
            FacialHairType.MOUSTACHE_MAGNUM, TopType.SHORT_HAIR_DREADS_02,
            ClotheType.BLAZER_SWEATER, Color.PASTEL_BLUE)

    query_processor = initialize_query_processor()

    def generate_response(query):
        try:
            with st.spinner("Bot is typing..."):
                response = query_processor.generate_response(query)
            return response
        except Exception as e:
            st.error(f"Error: {e}")
            return "Sorry, I couldn't process your query."

    # Form for user input
    with st.form(key='query_form'):
        user_input = st.text_input("", key="input")
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            st.session_state['queries'].append(user_input)
            response = generate_response(user_input)
            st.session_state['responses'].append(response)
            st.experimental_rerun()  # Use st.experimental_rerun to reset the input field

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i in range(len(st.session_state['responses'])):
            # Display user message
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"<div style='text-align: right;'>{st.session_state['queries'][i]}</div>",
                            unsafe_allow_html=True)
            with col2:
                st.image(f"data:image/png;base64,{st.session_state['user_avatar']}", width=40)

            # Display bot response
            col3, col4 = st.columns([1, 5])
            with col3:
                st.image(f"data:image/png;base64,{st.session_state['bot_avatar']}", width=40)
            with col4:
                st.markdown(f"<div style='margin-left: -50px;'>{st.session_state['responses'][i]}</div>",
                            unsafe_allow_html=True)


if __name__ == "__main__":
    main()