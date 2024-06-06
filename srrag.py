import streamlit as st
from PIL import Image
from io import BytesIO
import mmlib as mvl

st.title("PDF 파일 업로드 및 텍스트 벡터 생성")

# 상태 초기화
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'indexing_done' not in st.session_state:
    st.session_state.indexing_done = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

def process_pdf(uploaded_file):
    log_container = st.empty()
    with st.spinner('파일 처리 중...'):
        save_dir, metadata = mvl.save_images_and_text_from_pdf(uploaded_file, st_container=log_container)

    print(metadata)
    
    st.success(f"PDF 파일에서 이미지와 텍스트를 '{save_dir}' 디렉토리에 저장했습니다.")
    
    faiss_log_container = st.empty()
    with st.spinner('FAISS 인덱스 생성 중...'):
        index = mvl.get_index(metadata, st_container=faiss_log_container)
    
    st.success("텍스트 벡터를 생성하고 FAISS 인덱스에 저장했습니다.")
    
    # 상태 업데이트
    st.session_state.faiss_index = index
    st.session_state.metadata = metadata
    st.session_state.indexing_done = True

# PDF 파일 업로드
if not st.session_state.indexing_done:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        process_pdf(uploaded_file)

# 업로드된 파일과 인덱스가 있는 경우에만 검색어 입력 허용
if st.session_state.indexing_done and st.session_state.faiss_index is not None and st.session_state.metadata is not None:
    st.header("FAISS 인덱스가 생성되었습니다. 이제 검색을 수행할 수 있습니다.")
    
    search_query = st.text_input("검색어를 입력하세요")
    
    if st.button("검색 실행"):
        if search_query:
            try:
                if st.session_state.faiss_index is None:
                    raise ValueError("FAISS 인덱스가 제대로 초기화되지 않았습니다.")
                # FAISS 검색 수행
                search_results = st.session_state.faiss_index.similarity_search(search_query)
                st.session_state.search_results = search_results

                print(search_results)
                

            except Exception as e:
                st.error(f"FAISS 검색 중 오류 발생: {e}, {search_query}")

# 검색된 결과를 Sonnet에 쿼리
if st.session_state.search_results:
    if st.button("Sonnet에 이미지와 검색어 보내기"):
        if search_query:
            search_results = st.session_state.search_results
            images = [BytesIO(open(result.metadata["image_path"], "rb").read()) for result in search_results]
            try:
                sonnet_result = mvl.query_sonnet_with_images_and_text(images, search_query, [result.metadata for result in search_results])
                st.write("Sonnet 결과:", sonnet_result)
                
                # 검색된 이미지를 다시 출력
                st.write("검색된 이미지:")
                for image in images:
                    img = Image.open(image)
                    st.image(img, use_column_width=True, caption="검색된 이미지")
            except Exception as e:
                st.error(f"Sonnet 호출 중 오류 발생: {e}")
