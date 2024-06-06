import os
import streamlit as st
from io import BytesIO
import mmlib as mvl  # 해당 함수들을 포함한 모듈이어야 합니다.
from PIL import Image

# 전역 변수 초기화
streaming_text = ""
faiss_index = None

def streaming_callback(chunk):
    global streaming_text
    streaming_text += chunk
    placeholder.write(streaming_text)

# 제목
st.title("PDF Search and Query with FAISS and Sonnet")

# Step 1: PDF 파일 업로드 및 색인
with st.container():
    st.header("파일 업로드 및 색인")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_pdf is not None:
        save_dir = "images"
        st.write("Processing PDF...")
        #save_dir, metadatas = mvl.save_images_and_text_from_pdf(uploaded_pdf, save_dir, st_container=st)
        save_dir, metadatas = mvl.save_page_images_from_pdf(uploaded_pdf, save_dir, st_container=st)
        st.write(f"PDF processed. Extracted {len(metadatas)} metadata entries.")
        
        if 'faiss_index' not in st.session_state:
            st.write("Creating FAISS index...")
            faiss_index = mvl.get_index(metadatas, st_container=st)
            if faiss_index is None:
                st.error("Failed to create FAISS index.")
                st.stop()  # 중지 함수로 애플리케이션 종료
            st.write("FAISS index created successfully.")
            st.session_state.faiss_index = faiss_index
        else:
            st.write("FAISS index already exists.")

# Step 2: 검색어 입력 및 결과 표시
with st.container():
    st.header("검색어 입력 및 결과 표시")
    search_query = st.text_input("Enter search query", "")
    
    if search_query and 'faiss_index' in st.session_state:
        st.write(f"Searching for '{search_query}' in FAISS index...")
        query_vector = mvl.get_text_vector(search_query)
        
        # Step 3: FAISS 검색 수행
        try:
            faiss_index = st.session_state.faiss_index
            search_results = faiss_index.similarity_search_by_vector(query_vector)
            st.write(f"Found {len(search_results)} results.")
            
            # 검색된 이미지와 텍스트 먼저 보여줌
            for result in search_results:
                #st.write(result.metadata["page_text"])
                image_path = result.metadata["image_path"]
                image = Image.open(image_path)
                st.image(image, caption=image_path)
                
            # Sonnet에 쿼리
            images = [BytesIO(open(result.metadata["image_path"], "rb").read()) for result in search_results]
            sonnet_data = [result.metadata for result in search_results]
            st.write(f"Querying Sonnet with {len(images)} images.")
            
            # 스트리밍 콜백을 위한 전역 placeholder 초기화
            global placeholder
            placeholder = st.empty()
                
            # Sonnet 쿼리 스트리밍 실행
            mvl.query_sonnet_with_images_and_text_with_streaming(images, search_query, sonnet_data, streaming_callback=streaming_callback)
            
        except Exception as e:
            st.error(f"Error during FAISS search or Sonnet query: {str(e)}")
