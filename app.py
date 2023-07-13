from langchain.document_loaders import TextLoader
import os
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, auth
import pyrebase
from pyrebase import initialize_app
import json
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')
active_loop_token = os.getenv('ACTIVELOOP_TOKEN')
deeplake_account_name = os.getenv('DEEPLAKE_ACCOUNT_NAME')

# setting up firebase
cred = credentials.Certificate("apollov1-6ca8c-firebase-adminsdk-6ejxf-449afd9ae4.json")
firebase_admin.initialize_app(cred)
pb = pyrebase.initialize_app(json.load(open('firebase_config.json')))

root_dir = "./data"

dataset_path = "hub://medicchatbot/text_embedding"

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".docx") or file.endswith('.csv'):
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
print(f"{len(docs)}")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")

embeddings = OpenAIEmbeddings()
embeddings

db = DeepLake.from_documents(
    texts, embeddings, dataset_path=f"hub://medicchatbot/text_embedding"
)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 20
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 20

model = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=400) 
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

app=FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )
class QuestionAnswer(BaseModel):
    question: str


@app.post("/answer")
async def answer_question(question_answer: QuestionAnswer):
    try:    
        question = question_answer.question
        chat_history = []

        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))

        return {"question": question, "answer": result["answer"]}
    except Exception as e:
        print("Error in question answering:", str(e))
        return fallback_chat_completion(question, chat_history)
    

def fallback_chat_completion(question, chat_history):
    model = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=400)
    result = model({"messages": chat_history + [(question, "")]})
    answer = result["choices"][0]["message"]["content"]

    return {"question": question, "answer": answer}    



@app.get('/bioc')
def get_bioc_data(format: str = "xml", id: str = None, encoding: str = "unicode"):
    if not id:
        raise HTTPException(status_code=400, detail="You must provide an ID (PubMed ID or PMC ID).")

    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_{format}/{id}/{encoding}"
    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Unable to fetch data from NLM API.")

    return response.content



# User management

class CreateUserRequest(BaseModel):
    email: str
    password: str
    firstName: str
    lastName: str
    licenseLevel: str
    country: str
    state: str
    localProtocol: str
    subscriptionInfo: str


class UpdateUserRequest(BaseModel):
    email: str = None
    password: str = None
    userProfile: dict = None


@app.post('/user/{id}/thumbsUp')
def thumbs_up(id: str):
    try:
        user = db.child('users').child(id).get()
        if user.exists():
            user_ref = db.child('users').child(id)
            user_ref.update({'thumbsUp': user.val().get('thumbsUp', 0) + 1})
            return '', 204
        else:
            raise HTTPException(status_code=404, detail='User not found')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while thumbs up: {e}")


@app.post('/user/{id}/thumbsDown')
def thumbs_down(id: str):
    try:
        user = db.child('users').child(id).get()
        if user.exists():
            user_ref = db.child('users').child(id)
            user_ref.update({'thumbsDown': user.val().get('thumbsDown', 0) + 1})
            return '', 204
        else:
            raise HTTPException(status_code=404, detail='User not found')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while thumbs down: {e}")


@app.post('/user')
def create_user(request: CreateUserRequest):
    try:
        user = auth.create_user_with_email_and_password(
            email=request.email,
            password=request.password
        )

        user_profile = {
            'firstName': request.firstName,
            'lastName': request.lastName,
            'licenseLevel': request.licenseLevel,
            'country': request.country,
            'state': request.state,
            'localProtocol': request.localProtocol,
            'subscriptionInfo': request.subscriptionInfo
        }

        db.child('users').child(user['localId']).set(user_profile)

        return {'userId': user['localId']}, 201
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while creating user: {e}")


@app.post('/login')
def login(request: dict):
    try:
        email = request.get('email')
        password = request.get('password')

        user = auth.sign_in_with_email_and_password(email, password)

        # Get the user ID and generate a JWT token if needed
        user_id = user['localId']
        token = auth.create_custom_token(user_id)

        return {'userId': user_id, 'token': token}, 200
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while logging in: {e}")


@app.delete('/user/{id}')
def delete_user(id: str):
    try:
        auth.delete_user(id)
        db.child('users').child(id).remove()
        return '', 204
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while deleting user: {e}")


@app.post('/sessionLogin')
def session_login(request: dict):
    try:
        id_token = request.get('idToken')
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token.get('uid')
        custom_claims = decoded_token.get('claims', {})
        return {'uid': uid, 'customClaims': custom_claims}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while logging in: {e}")


@app.post('/user/{id}/claims')
def set_custom_user_claims(id: str, request: dict):
    try:
        claims = request.get('claims', {})
        auth.set_custom_user_claims(id, claims)
        return '', 204
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while setting custom user claims: {e}")

@app.get('/users')
def list_users():
    try:
        users = auth.list_users().iterate_all()
        users_list = [user.uid for user in users]
        return users_list
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while listing users: {e}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


