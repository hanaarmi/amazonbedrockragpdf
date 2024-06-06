import os
import boto3
import json
import base64
from io import BytesIO
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def get_text_vector(input_text):

    if not input_text or len(input_text.strip()) == 0:
        raise ValueError("Input text cannot be empty or blank")
    
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    request_body = {
        "inputText": input_text
    }
    
    body = json.dumps(request_body)
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.titan-embed-text-v2:0",  # Titan Text v2 모델
        accept="application/json", 
        contentType="application/json"
    )
    
    response_body = json.loads(response.get('body').read())

    
    embedding = response_body.get("embedding")
    return embedding

def get_index(metadata_list, st_container=None):
    text_embeddings = []
    metadatas = []
    processed_texts = ""
    num_items = len(metadata_list)

    if st_container:
        progress = st_container.progress(0)

    for idx, metadata in enumerate(metadata_list):
        page_text = metadata["page_text"]
        processed_texts += page_text + "\n\n"
        if st_container:
            # st_container.markdown(f"**현재 페이지**: {idx + 1}/{num_items}\n\n" + processed_texts)
            progress_value = (idx + 1) / num_items
            progress.progress(progress_value)
        try:
            embedding = get_text_vector(page_text)
            text_embeddings.append((page_text, embedding))  # 변경된 부분: 튜플 형태로 저장
            metadatas.append({"image_path": metadata["image_path"], "page_text": metadata["page_text"]})
        except ValueError as e:
            print(f"Ignored empty or invalid text: {e}")
    
    if text_embeddings:
        index = FAISS.from_embeddings(
            text_embeddings=text_embeddings,  # 확인: (텍스트, 임베딩)의 형태로 전달
            embedding=None,
            metadatas=metadatas
        )
        #print(f"[DEBUG] 임베딩 샘플: {text_embeddings[:2]}")
    else:
        print("[DEBUG] FAISS 인덱스를 생성할 수 있는 데이터가 없습니다.")
        index = None

    if st_container:
        st_container.write("FAISS 인덱스 생성 완료")
    
    return index


def get_index_test():
    loader = TextLoader("./abc.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embedding=None)

    return db.index
    

def query_sonnet_with_images_and_text(images, search_text, metadata, additional_prompt="한국어로만 대답해줘."):
    session = boto3.Session()
    sonnet = session.client(service_name='bedrock-runtime')
    
    #print("Before making prompt")

    contents = []
    for idx, (image, meta) in enumerate(zip(images, metadata)):
        #print(meta)
        image_base64 = base64.b64encode(image.getvalue()).decode('utf-8')
        
        # 이미지와 텍스트를 각각 content에 추가
        contents.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",  # 이미지는 JPEG 또는 PNG 형식을 사용합니다.
                "data": image_base64
            }
        })
        contents.append({
            "type": "text",
            "text": meta["page_text"]
        })

    # 마지막에 검색어를 텍스트로 추가
    contents.append({
        "type": "text",
        "text": search_text
    })

    # 추가 프롬프트를 포함
    contents.append({
        "type": "text",
        "text": additional_prompt
    })

    # 전체 메시지 구조
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": contents
            }
        ]
    }

    try:

        serialized_body = json.dumps(body)

        response = sonnet.invoke_model(
            body=serialized_body,
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        # 응답 본문을 전체 출력
        response_body = json.loads(response['body'].read())
        print("Response Body:", json.dumps(response_body, indent=2))

        # 올바른 내용을 추출하도록 코드 수정
        if ('content' in response_body and
            isinstance(response_body['content'], list) and
            len(response_body['content']) > 0 and
            'text' in response_body['content'][0]):
            result = response_body['content'][0]['text']
        else:
            result = "결과를 가져오지 못했습니다."
    except Exception as e:
        result = str(e)

    return result

# PDF파일에서 이미지와 텍스트 저장. Streamlit에서 실시간으로 가져옵니다.
def save_images_and_text_from_pdf(uploaded_pdf, save_dir="images", min_size=1024, st_container=None):
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.makedirs(save_dir)

    pdf = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    
    metadatas = []
    all_text = ""  # 모든 텍스트를 저장할 변수 추가

    for i in range(len(pdf)):
        # if i > 8:  # 최대 페이지 수 제한
        #     break
        page = pdf.load_page(i)
        text = page.get_text("text")
        images = page.get_images(full=True)
        
        if st_container:
            all_text += f"\n\n페이지 {i}:\n{text}\n"
            # st_container.text_area("텍스트 추출 과정", value=all_text, height=300, key=f"text_area_{i}")

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            # 이미지 크기 필터링
            if len(image_bytes) > min_size:
                image_path = os.path.join(save_dir, f"page_{i}_image_{img_index}.png")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # 이미지와 텍스트 메타데이터 저장
                metadata = {
                    "image_path": image_path,
                    "page_text": text
                }
                metadatas.append(metadata)
                #print("MetaData: " + str(metadata))
    return save_dir, metadatas


def save_page_images_from_pdf(uploaded_pdf, save_dir="images", st_container=None):
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.makedirs(save_dir)

    pdf = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    
    metadatas = []  # 메타데이터를 저장할 리스트 초기화

    for i in range(len(pdf)):
        # if i > 8:  # 최대 페이지 수 제한
        #     break
        page = pdf.load_page(i)

        # 페이지를 이미지로 변환
        pix = page.get_pixmap()
        image_path = os.path.join(save_dir, f"page_{i}.png")
        pix.save(image_path)
        
        # 페이지 텍스트 추출
        page_text = page.get_text("text")

        # 메타데이터 생성 및 추가
        metadata = {
            "image_path": image_path,
            "page_text": page_text
        }
        metadatas.append(metadata)

        # Streamlit 컨테이너에 이미지 표시
        # if st_container:
            # st_container.image(image_path, caption=f"Page {i}")
    
    return save_dir, metadatas

def chunk_handler(chunk):
    """스트리밍된 데이터를 처리하는 기본 핸들러"""
    print(chunk, end='')

def get_streaming_response(prompt, streaming_callback):

    # Bedrock 클라이언트 생성
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    """스트리밍 응답을 처리하는 함수"""
    response = bedrock.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0", 
        body=prompt,
        accept='application/json',
        contentType='application/json'
    )

    for event in response.get('body'):
        chunk = json.loads(event['chunk']['bytes'])
        if chunk['type'] == 'content_block_delta':
            if chunk['delta']['type'] == 'text_delta':
                streaming_callback(chunk['delta']['text'])

def query_sonnet_with_images_and_text_with_streaming(images, search_text, metadata, additional_prompt="한국어로 대답해", streaming_callback=chunk_handler):
    contents = []
    for idx, (image, meta) in enumerate(zip(images, metadata)):
        image_base64 = base64.b64encode(image.getvalue()).decode('utf-8')

        contents.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64
            }
        })
        contents.append({
            "type": "text",
            "text": meta["page_text"]
        })
    
    contents.append({
        "type": "text",
        "text": search_text + "\n\n"
    })

    contents.append({
        "type": "text",
        # "text": additional_prompt + "\n\n" + "이 문서에는 여러 개의 표가 있습니다. 각 표는 서로 다른 수의 행과 열을 가지고 있으며, 일부 셀이 병합되어 있거나 포함된 데이터가 없을 수 있습니다. 표는 문서 내 여러 위치에 있습니다. 첫번째 또는 두번째 줄 그리고 첫번째열 또는 두번째 열에 데이터 유형이 표시될 수 있습니다. 데이터 유형에 병합된 셀이 있는 경우, 밑의 데이터는 여러 줄이 해당 유형에 포함될 수 있습니다. 예를 들어 OLED 모델에 해당하는 데이터는 83DC90, 77SD95뿐만 아니라 77SD90, 65SD95 도 해당합니다." + "\n\n"
        "text": additional_prompt + "\n\n"
    })

    # 전체 프롬프트 구성
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": contents
            }
        ]
    }

    # JSON 직렬화
    serialized_body = json.dumps(prompt)
    
    # 스트리밍 응답 호출
    get_streaming_response(serialized_body, streaming_callback)
