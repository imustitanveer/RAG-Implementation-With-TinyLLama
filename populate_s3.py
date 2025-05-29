import argparse
import os
import json
import uuid
import boto3
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding import get_embedding_function
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
S3_BUCKET_NAME = "dam-chatbot-saas"
S3_PREFIX = "embeddings/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete all S3 data under the prefix.")
    args = parser.parse_args()

    if args.reset:
        print("🧨 Clearing S3 bucket contents...")
        clear_s3_prefix()

    documents = load_documents()
    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)
    embed_and_upload(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks


def embed_and_upload(chunks: list[Document]):
    embedder = get_embedding_function()

    print(f"🚀 Uploading {len(chunks)} chunks to S3 with embeddings...")
    for chunk in chunks:
        try:
            embedding = embedder.embed_query(chunk.page_content)
            data = {
                "id": chunk.metadata["id"],
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "content": chunk.page_content,
                "embedding": embedding,
            }

            filename = f"{S3_PREFIX}{uuid.uuid4()}.json"
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=filename,
                Body=json.dumps(data),
                ContentType="application/json"
            )
        except Exception as e:
            print(f"❌ Failed to upload chunk {chunk.metadata['id']}: {e}")

    print("✅ Done uploading all chunks.")


def clear_s3_prefix():
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)

    to_delete = []
    for page in pages:
        for obj in page.get("Contents", []):
            to_delete.append({"Key": obj["Key"]})

    if to_delete:
        s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={"Objects": to_delete})
        print(f"🗑️ Deleted {len(to_delete)} objects.")
    else:
        print("📭 No objects to delete.")


if __name__ == "__main__":
    main()