import os
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tkinter import *
import tkinter.ttk as ttk
import shutil

# 1. 환경 설정 및 API 초기화
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 2. DB 초기화 및 데이터 인덱싱 함수
def prepare_database(file_path):
    db_path = "./chroma_db"
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)  # 기존 DB 삭제 
        print("기존 DB가 삭제되었습니다.")

    dbclient = chromadb.PersistentClient(path=db_path)

    collection = dbclient.create_collection(name="sogang_collection")

    # 엑셀 로드 및 청킹
    df = pd.read_excel(file_path)
    all_text = "\n\n".join(df['Content'].astype(str).tolist())
    
    # 400자 단위 분할 (Overlap 50)
    chunks = [all_text[i:i+400] for i in range(0, len(all_text), 350)]

    print(f"{len(chunks)}개의 지식 조각을 DB에 저장 중...")
    for idx, chunk in enumerate(chunks):
        # 임베딩 생성
        response = client.embeddings.create(input=[chunk], model="text-embedding-3-large")
        embedding = response.data[0].embedding
        
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": "namu_wiki", "index": idx}],
            ids=[f"sogang_{idx}"]
        )
    return collection

# 3. RAG 답변 생성 로직
def generate_answer(query, collection):
    # 질문 임베딩 및 검색
    query_embedding = client.embeddings.create(input=[query], model="text-embedding-3-large").data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    context = "\n\n".join(results["documents"][0])
    
    # 프롬프트 구성
    system_prompt = "당신은 서강대학교 나무위키 정보를 바탕으로 답변하는 어시스턴트입니다. 문서 내용에 기반하여 친절하게 답하세요." \
    "문서에 업근된 내용이 아니라면 답변을 지어내지 말고 '죄송합니다, 해당 정보는 찾을 수 없습니다.'라고 답하세요."

    user_prompt = f"문서 내용:\n{context}\n\n질문: {query}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# 4. GUI 구성 (Tkinter)
def start_gui(collection):
    root = Tk()
    root.title('서강대 지식 챗봇 (RAG)')
    root.geometry('500x700')
    root.configure(bg="pink")

    def process_query():
        query = text_input.get("1.0", END).strip()
        if query:
            label_status.config(text="서강대 DB 검색 중...", foreground="blue")
            root.update_idletasks()
            
            answer = generate_answer(query, collection)
            
            text_output.config(state="normal")
            text_output.delete("1.0", END)
            text_output.insert(END, answer)
            text_output.config(state="disabled")
            label_status.config(text="답변 완료", foreground="green")
        else:
            label_status.config(text="질문을 입력하세요.", foreground="red")

    # 입력창
    frame_input = Frame(root, padx=10, pady=10)
    frame_input.pack(fill="x")
    Label(frame_input, text="질문 입력", font=("Microsoft YaHei", 12, "bold")).pack(anchor="w")
    text_input = Text(frame_input, height=5, font=("Microsoft YaHei", 11))
    text_input.pack(pady=5)
    ttk.Button(frame_input, text="전송", command=process_query).pack()
    label_status = ttk.Label(frame_input, text="")
    label_status.pack()

    # 출력창
    frame_output = Frame(root, padx=10, pady=10)
    frame_output.pack(fill="both", expand=True)
    Label(frame_output, text="답변", font=("Microsoft YaHei", 12, "bold")).pack(anchor="w")
    text_output = Text(frame_output, wrap="word", font=("Microsoft YaHei", 11), state="disabled", bg="#f9f9f9")
    text_output.pack(side="left", fill="both", expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    # 1. DB 준비 
    target_collection = prepare_database("sogang_wiki.xlsx")
    # 2. GUI 실행
    start_gui(target_collection)