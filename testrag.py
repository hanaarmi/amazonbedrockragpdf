import os
import json
import base64
from io import BytesIO
from PIL import Image
import mmlib as mvl


def debug_print(message):
    print(f"[DEBUG] {message}")

def main():
    pdf_path = "srretail.pdf"
    search_query = "fhd uhd qhd"

    # Step 1: PDF 처리 및 이미지 추출
    debug_print("PDF 파일 처리를 시작합니다.")
    save_dir = "images"
    #save_dir, metadatas = mvl.save_images_and_text_from_pdf(open(pdf_path, "rb"), save_dir)
    save_dir, metadatas = mvl.save_page_images_from_pdf(open(pdf_path, "rb"), save_dir)
    debug_print(f"PDF 파일 처리 완료. {len(metadatas)}개의 메타데이터를 추출했습니다.")
    
    # Step 2: FAISS 인덱스 생성
    debug_print("FAISS 인덱스를 생성합니다.")
    index = mvl.get_index(metadatas)
    #index = mvl.get_index_test()
    if index is None:
        debug_print("FAISS 인덱스를 생성하는 데 실패했습니다.")
        return
    debug_print(f"FAISS 인덱스 생성 완료.")
    
    # Step 3: 검색어를 사용하여 FAISS 검색 수행
    debug_print(f"검색어 '{search_query}'로 FAISS 검색을 수행합니다.")
    try:
        query_vector = mvl.get_text_vector(search_query)
        search_results = index.similarity_search_by_vector(query_vector, k=10)
        debug_print(f"검색 결과: {len(search_results)}개의 결과를 찾았습니다.")
        for result in search_results:
            debug_print(f"Result Metadata: {result.metadata}")
    except Exception as e:
        debug_print(f"FAISS 검색 중 오류 발생: {str(e)}")
        return
    
    
    # Step 4: 검색된 결과의 이미지를 Sonnet에 쿼리
    images = [BytesIO(open(result.metadata["image_path"], "rb").read()) for result in search_results]
    try:
        sonnet_data = [result.metadata for result in search_results]
        debug_print(f"Sonnet에 쿼리를 보냅니다. {len(images)}개의 이미지를 포함합니다.")
        #sonnet_result = mvl.query_sonnet_with_images_and_text(images, search_query, sonnet_data)
        mvl.query_sonnet_with_images_and_text_with_streaming(images, search_query, sonnet_data)
        debug_print("Sonnet 쿼리 완료.")
        #print("Sonnet 결과:", sonnet_result)
    except Exception as e:
        debug_print(f"Sonnet 쿼리 중 오류 발생: {str(e)}

if __name__ == "__main__":
    main()
